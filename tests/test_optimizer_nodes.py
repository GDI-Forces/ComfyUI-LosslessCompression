import asyncio

import pytest
import torch

import comfy.model_management
import nodes
from comfy.ldm.modules import attention as comfy_attention
from conftest import REPO_ROOT, sample
from optimizer_nodes import CompactVRAMBeforeLoad, CompressModel, FreeVRAM, LoadCheckpointFP8, SpeedUpModel


def load(ckpt_name, weight_dtype="default"):
    return LoadCheckpointFP8.execute(ckpt_name, weight_dtype).args[0]


def speed_up(model, attention="keep", fp16_accumulation=False, step_cache=0.0, torch_compile=False):
    return SpeedUpModel.execute(model, attention, fp16_accumulation, step_cache, torch_compile).args[0]


def compress(model, weight_dtype="fp8_e4m3fn"):
    return CompressModel.execute(model, weight_dtype).args[0]


def with_lora(model):
    """A clone with a LoRA-style weight patch big enough to visibly change the output."""
    patched = model.clone()
    weight = patched.model.state_dict()["diffusion_model.out.2.weight"]
    assert patched.add_patches({"diffusion_model.out.2.weight": (torch.full_like(weight, 0.05),)})
    return patched


def compact(model, unload_other_models=True):
    return CompactVRAMBeforeLoad.execute(model, unload_other_models).args[0]


@pytest.fixture
def fake_sage(monkeypatch):
    """Registers a stand-in "sage" backend that counts its calls and runs PyTorch attention."""
    calls = []
    pytorch_attention = comfy_attention.get_attention_function("pytorch")

    def sage(*args, **kwargs):
        calls.append(1)
        return pytorch_attention(*args, **kwargs)

    monkeypatch.setitem(comfy_attention.REGISTERED_ATTENTION_FUNCTIONS, "sage", sage)
    return calls


def test_pack_loads_as_a_custom_node():
    assert asyncio.run(nodes.load_custom_node(REPO_ROOT))
    for node_id in ["OptimizerSpeedUpModel", "OptimizerLoadCheckpointFP8", "OptimizerCompressModel",
                    "OptimizerCompactVRAM", "OptimizerFreeVRAM", "LosslessCompressModelFile",
                    "LosslessLoadDiffusionModel", "LosslessLoadCheckpoint", "LosslessLoadCLIP", "LosslessLoadLora",
                    "LosslessLoadVAE", "LosslessKeepModelCompressed"]:
        info = nodes.NODE_CLASS_MAPPINGS[node_id].GET_NODE_INFO_V1()
        assert info["category"].startswith("optimization")


def test_fp8_checkpoint_stores_weights_in_fp8(tiny_checkpoint):
    default = load(tiny_checkpoint)
    fp8 = load(tiny_checkpoint, "fp8_e4m3fn")

    assert fp8.model_dtype() == torch.float8_e4m3fn
    assert fp8.model_size() < default.model_size() / 2
    assert torch.isfinite(sample(fp8)).all()


def test_auto_attention_uses_sage_when_installed(tiny_checkpoint, fake_sage):
    sample(speed_up(load(tiny_checkpoint), attention="auto"), steps=2)
    assert len(fake_sage) > 0


def test_missing_attention_backend_leaves_attention_alone(tiny_checkpoint, monkeypatch):
    for name in ["sage", "flash"]:
        monkeypatch.delitem(comfy_attention.REGISTERED_ATTENTION_FUNCTIONS, name, raising=False)
    model = load(tiny_checkpoint)

    for choice in ["auto", "flash", "keep"]:
        patched = speed_up(model, attention=choice)
        assert "optimized_attention_override" not in patched.model_options["transformer_options"]


def test_speed_up_does_not_change_the_input_model(tiny_checkpoint, fake_sage):
    model = load(tiny_checkpoint)
    speed_up(model, attention="sage", fp16_accumulation=True, step_cache=0.2)

    sample(model, steps=2)
    assert fake_sage == []
    assert model.model_options["transformer_options"].get("easycache") is None
    assert model.get_all_wrappers("apply_model") == []


def test_fp16_accumulation_is_on_only_while_the_model_runs(tiny_checkpoint, watch_forward):
    if not hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
        pytest.skip("needs PyTorch 2.7+")
    model = speed_up(load(tiny_checkpoint), fp16_accumulation=True)
    calls = watch_forward(model)

    sample(model, steps=3)

    assert calls == [True] * 3
    assert torch.backends.cuda.matmul.allow_fp16_accumulation is False


def test_step_cache_skips_model_passes(tiny_checkpoint, watch_forward):
    model = load(tiny_checkpoint)
    calls = watch_forward(model)
    baseline = sample(model, steps=12)
    full_passes = len(calls)

    calls.clear()
    cached = sample(speed_up(model, step_cache=0.5), steps=12)

    assert len(calls) < full_passes
    assert torch.isfinite(cached).all()
    assert cached.shape == baseline.shape


def test_torch_compile_matches_eager(tiny_checkpoint):
    model = load(tiny_checkpoint)
    eager = sample(model, steps=2)
    compiled = sample(speed_up(model, torch_compile=True), steps=2)
    assert torch.allclose(eager, compiled, atol=1e-3)


@pytest.fixture(params=["checkpoint", "diffusion_model"])
def loaded_model(request):
    """The tiny model from Load Checkpoint, then from Load Diffusion Model."""
    if request.param == "checkpoint":
        return load(request.getfixturevalue("tiny_checkpoint"))
    return nodes.UNETLoader().load_unet(request.getfixturevalue("tiny_diffusion_model"), "default")[0]


def test_compress_model_stores_weights_in_fp8_and_leaves_the_input_alone(loaded_model):
    model = loaded_model
    compressed = compress(model)

    assert compressed.model_dtype() == torch.float8_e4m3fn
    assert model.model_dtype() != torch.float8_e4m3fn
    assert torch.isfinite(sample(compressed)).all()
    sample(model)
    assert 0 < compressed.loaded_size() <= model.loaded_size() / 2


def test_compress_model_keeps_loras_and_options(tiny_checkpoint):
    model = speed_up(load(tiny_checkpoint), step_cache=0.2)
    patched = with_lora(model)

    compressed = compress(patched)
    assert compressed.patches.keys() == patched.patches.keys()
    assert compressed.model_options["transformer_options"]["easycache"] is not None
    assert not torch.allclose(sample(compressed, steps=2), sample(compress(model), steps=2), atol=1e-2)


def test_compress_model_passes_fp8_models_through(tiny_checkpoint):
    fp8 = load(tiny_checkpoint, "fp8_e4m3fn")
    assert compress(fp8) is fp8


def test_compress_model_must_come_before_torch_compile(tiny_checkpoint):
    with pytest.raises(ValueError, match="before torch.compile"):
        compress(speed_up(load(tiny_checkpoint), torch_compile=True))


def test_compressed_model_reloads_as_fp8(loaded_model):
    # torch.compile and multi-GPU copies reload the model this way
    assert compress(loaded_model).clone(force_deepcopy=True).model_dtype() == torch.float8_e4m3fn


def test_compact_vram_unloads_the_uncompressed_original(tiny_checkpoint):
    model = load(tiny_checkpoint)
    comfy.model_management.load_models_gpu([model])

    sample(compact(compress(model)), steps=1)
    assert model not in comfy.model_management.loaded_models()


def test_compact_vram_unloads_other_models_before_loading(tiny_checkpoint):
    other = load(tiny_checkpoint)
    comfy.model_management.load_models_gpu([other])
    model = load(tiny_checkpoint)

    sample(model, steps=1)
    assert other in comfy.model_management.loaded_models()

    sample(compact(model), steps=1)
    assert other not in comfy.model_management.loaded_models()


def test_compact_vram_keeps_the_model_it_is_loading(tiny_checkpoint, monkeypatch):
    model = load(tiny_checkpoint)
    comfy.model_management.load_models_gpu([model])
    unloaded = []
    model_unload = comfy.model_management.LoadedModel.model_unload

    def record_unload(self, *args, **kwargs):
        unloaded.append(self.model)
        return model_unload(self, *args, **kwargs)

    monkeypatch.setattr(comfy.model_management.LoadedModel, "model_unload", record_unload)
    sample(compact(model), steps=1)
    assert model not in unloaded


def test_compact_vram_can_leave_other_models_loaded(tiny_checkpoint):
    other = load(tiny_checkpoint)
    comfy.model_management.load_models_gpu([other])

    sample(compact(load(tiny_checkpoint), unload_other_models=False), steps=1)
    assert other in comfy.model_management.loaded_models()


def test_free_vram_unloads_models_and_passes_value_through(tiny_checkpoint):
    model = load(tiny_checkpoint)
    comfy.model_management.load_models_gpu([model])
    assert model in comfy.model_management.loaded_models()

    value = {"samples": torch.ones(1, 4, 8, 8)}
    assert FreeVRAM.execute(value, True).args[0] is value
    assert model not in comfy.model_management.loaded_models()
