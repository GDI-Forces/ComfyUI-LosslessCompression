import json
import os
import subprocess
import sys

import pytest
import safetensors
import safetensors.torch
import torch

import comfy.model_management
import comfy.patcher_extension
import comfy.sampler_helpers
import comfy.utils
import folder_paths
import nodes
from conftest import REPO_ROOT, sample
from lossless import codec, fileformat, kernels, memory
from comfy.quant_ops import QuantizedTensor
from lossless.nodes import (CompressModelFile, KeepModelCompressed, LoadCheckpointLossless, LoadCLIPLossless,
                            LoadDiffusionModelLossless, LoadLoraLossless, LoadVAELossless)
from optimizer_nodes import CompressModel, LoadCheckpointFP8


def same_bits(a, b):
    return (a.dtype == b.dtype and a.shape == b.shape
            and torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)))


def weights(shape, dtype):
    """Random weights as they look in a model file of that dtype: small floats, or int8 spread over most of its
    range the way per-channel int8 quantization leaves them."""
    values = torch.randn(shape)
    if dtype == torch.int8:
        return (values * 30).round().clamp(-127, 127).to(torch.int8)
    return (values * 0.02).to(dtype)


def every_bit_pattern(dtype):
    view = codec.FORMATS[dtype][0]
    if view == torch.uint8:
        return torch.arange(256, dtype=torch.int32).to(torch.uint8).view(dtype)
    if view == torch.int16:
        return torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(dtype)
    specials = torch.tensor([0, -2**31, 2**31 - 1, 0x7F800000, -8388608, 0x7FC00001, 1], dtype=torch.int32)
    return torch.cat([torch.randint(-2**31, 2**31 - 1, (20000,), dtype=torch.int32), specials]).view(dtype)


@pytest.mark.parametrize("dtype", list(codec.FORMATS), ids=str)
@pytest.mark.parametrize("chunk", [codec.CHUNK, 64], ids=["one_chunk", "many_chunks"])
def test_every_bit_pattern_round_trips(dtype, chunk, monkeypatch):
    # NaN payloads, infinities, -0.0 and subnormals, mixed into ordinary weights so the tensor gets compressed
    monkeypatch.setattr(codec, "CHUNK", chunk)
    torch.manual_seed(0)
    patterns = every_bit_pattern(dtype)
    values = weights(10 * patterns.numel() + 5, dtype)
    tensor = torch.cat([values, patterns])[torch.randperm(values.numel() + patterns.numel())].view(-1, 1)

    blob, info = codec.encode(tensor)
    assert same_bits(codec.decode(blob, info), tensor)


@pytest.mark.parametrize("dtype, saving", [(torch.bfloat16, 0.25), (torch.float32, 0.12), (torch.float16, 0.08),
                                           (torch.float8_e4m3fn, 0.15), (torch.float8_e5m2, 0.15), (torch.int8, 0.06)], ids=str)
def test_weights_get_smaller(dtype, saving):
    torch.manual_seed(0)
    tensor = weights((512, 1024), dtype)
    blob, _ = codec.encode(tensor)
    assert blob.numel() < tensor.nbytes * (1 - saving)


def test_tensors_that_would_not_shrink_are_left_alone():
    assert codec.encode(torch.tensor([0.5], dtype=torch.bfloat16)) is None
    assert codec.encode(torch.arange(10000)) is None
    assert codec.encode(torch.randint(-127, 128, (512, 1024), dtype=torch.int8)) is None  # spread evenly over its range


def test_int8_picks_how_to_split_its_bits_per_tensor():
    torch.manual_seed(0)
    spread = weights((512, 1024), torch.int8)                   # magnitudes over most of the range
    few_values = torch.randint(-2, 3, (512, 1024), dtype=torch.int8)  # only the sign and the lowest bits ever vary
    (blob, info), (blob_few, info_few) = codec.encode(spread), codec.encode(few_values)

    assert info["split"] != info_few["split"]
    assert blob_few.numel() < few_values.nbytes * 0.5
    assert same_bits(codec.decode(blob, info), spread) and same_bits(codec.decode(blob_few, info_few), few_values)


def test_decoding_makes_no_tensors_from_python_lists(monkeypatch):
    # torch.tensor(list, device="cuda") copies to the GPU and waits for it: for every layer, at every step
    blob, info = codec.encode(weights((512, 1024), torch.bfloat16))
    codec.decode(blob, info)  # the lookup tables are made on the first call
    made = []
    make = torch.tensor
    monkeypatch.setattr(torch, "tensor", lambda *args, **kwargs: made.append(1) or make(*args, **kwargs))
    codec.decode(blob, info)
    assert made == []


def peak_cpu_bytes(work):
    """Most bytes of CPU tensors alive at once while work() runs, from the profiler's allocation events."""
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        work()
    try:
        events = sorted((e for e in prof.profiler.kineto_results.events() if e.name() == "[memory]"), key=lambda e: e.start_ns())
        sizes = [e.nbytes() for e in events]
    except AttributeError:
        pytest.skip("this PyTorch doesn't expose the profiler's allocation events")
    live = peak = 0
    for size in sizes:
        live += size
        peak = max(peak, live)
    return peak


@pytest.mark.parametrize("chunk", [codec.CHUNK, 1 << 16], ids=["one_chunk", "many_chunks"])
@pytest.mark.parametrize("dtype", list(codec.FORMATS), ids=str)
def test_decoding_stays_within_its_memory_estimate(dtype, chunk):
    # The estimate is what ComfyUI is told to keep free, so decoding must never need more.
    torch.manual_seed(0)
    blob, info = codec.encode(weights(1 << 21, dtype), chunk=chunk)
    codec.decode(blob, info)
    assert peak_cpu_bytes(lambda: codec.decode(blob, info)) <= codec.decode_memory(blob, info) - blob.nbytes


@pytest.fixture
def model_file(tmp_path):
    torch.manual_seed(0)
    tensors = {
        "blocks.0.weight": (torch.randn(256, 300) * 0.02).to(torch.bfloat16),
        "blocks.0.fp8": (torch.randn(128, 64) * 0.05).to(torch.float8_e4m3fn),
        "blocks.0.int8": weights((128, 64), torch.int8),
        "blocks.0.bias": torch.randn(300, dtype=torch.float16),
        "blocks.0.scale": torch.tensor(0.5),
        "position_ids": torch.arange(77),
        "mask": torch.tensor([True, False]),
    }
    path = tmp_path / "model.safetensors"
    safetensors.torch.save_file(tensors, str(path), metadata={"modelspec.title": "test"})
    return tensors, str(path)


def test_compressed_file_restores_the_exact_original(model_file, tmp_path):
    tensors, path = model_file
    compressed = str(tmp_path / "model.lossless.safetensors")
    original_bytes, written = fileformat.compress_file(path, compressed)

    assert written < original_bytes
    assert fileformat.verify(path, compressed) == []
    state_dict, metadata = fileformat.load(compressed)
    assert state_dict.keys() == tensors.keys()
    assert all(same_bits(state_dict[k], tensors[k]) for k in tensors)
    assert metadata == {"modelspec.title": "test"}

    restored = str(tmp_path / "restored.safetensors")
    fileformat.decompress_file(compressed, restored)
    assert all(same_bits(t, tensors[k]) for k, t in safetensors.torch.load_file(restored).items())


def test_verify_catches_a_corrupted_file(model_file, tmp_path):
    _, path = model_file
    compressed = tmp_path / "model.lossless.safetensors"
    fileformat.compress_file(path, str(compressed))
    data = bytearray(compressed.read_bytes())
    data[-100] ^= 0x01
    compressed.write_bytes(bytes(data))
    assert fileformat.verify(path, str(compressed)) != []


def test_pickle_checkpoints_can_be_compressed(model_file, tmp_path):
    tensors, _ = model_file
    ckpt = str(tmp_path / "model.ckpt")
    torch.save({"state_dict": tensors}, ckpt)
    compressed = str(tmp_path / "model.lossless.safetensors")
    fileformat.compress_file(ckpt, compressed)
    assert fileformat.verify(ckpt, compressed) == []


def test_written_tensors_are_aligned(model_file, tmp_path):
    _, path = model_file
    compressed = tmp_path / "model.lossless.safetensors"
    fileformat.compress_file(path, str(compressed))
    data = compressed.read_bytes()
    header = json.loads(data[8:8 + int.from_bytes(data[:8], "little")])
    sizes = {"F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "I8": 1, "I64": 8, "U8": 1, "BOOL": 1}
    assert all(entry["data_offsets"][0] % sizes[entry["dtype"]] == 0 for k, entry in header.items() if k != "__metadata__")


def compress_in_comfyui(folder, name):
    CompressModelFile.execute(f"{folder}/{name}")
    compressed = name.replace(".safetensors", ".lossless.safetensors")
    assert compressed in folder_paths.get_filename_list(folder)
    return compressed


def test_lossless_diffusion_model_samples_exactly_like_the_original(tiny_diffusion_model):
    compressed = compress_in_comfyui("diffusion_models", tiny_diffusion_model)
    original = nodes.UNETLoader().load_unet(tiny_diffusion_model, "default")[0]
    loaded = LoadDiffusionModelLossless.execute(compressed).args[0]
    assert torch.equal(sample(loaded), sample(original))


def test_lossless_checkpoint_samples_exactly_like_the_original(tiny_checkpoint):
    compressed = compress_in_comfyui("checkpoints", tiny_checkpoint)
    original = LoadCheckpointFP8.execute(tiny_checkpoint, "default").args[0]
    loaded = LoadCheckpointLossless.execute(compressed).args[0]
    assert torch.equal(sample(loaded), sample(original))


def test_lossless_text_encoder_encodes_exactly_like_the_original(tiny_text_encoder):
    compressed = compress_in_comfyui("text_encoders", tiny_text_encoder)
    original = nodes.CLIPLoader().load_clip(tiny_text_encoder, "stable_diffusion")[0]
    loaded = LoadCLIPLossless.execute(compressed, "stable_diffusion").args[0]

    def encode(clip):
        return clip.encode_from_tokens(clip.tokenize("a lighthouse at dusk"), return_pooled=True)
    (cond, pooled), (cond_original, pooled_original) = encode(loaded), encode(original)
    assert torch.isfinite(cond_original).all()
    assert torch.equal(cond, cond_original) and torch.equal(pooled, pooled_original)


def encode_prompt(clip):
    return clip.encode_from_tokens(clip.tokenize("a lighthouse at dusk"), return_pooled=True)[0]


def test_lossless_lora_applies_exactly_like_the_original(tiny_diffusion_model, tiny_text_encoder, tiny_lora):
    compressed = compress_in_comfyui("loras", tiny_lora)
    assert json.loads(safetensors.safe_open(folder_paths.get_full_path("loras", compressed), "pt").metadata()[fileformat.TENSORS_KEY])
    model = nodes.UNETLoader().load_unet(tiny_diffusion_model, "default")[0]
    clip = nodes.CLIPLoader().load_clip(tiny_text_encoder, "stable_diffusion")[0]

    model_original, clip_original = nodes.LoraLoader().load_lora(model, clip, tiny_lora, 0.8, 0.6)
    model_lossless, clip_lossless = LoadLoraLossless.execute(model, compressed, 0.8, 0.6, clip=clip).args

    assert torch.equal(sample(model_lossless, steps=3), sample(model_original, steps=3))
    assert not torch.equal(sample(model_lossless, steps=3), sample(model, steps=3))
    assert torch.equal(encode_prompt(clip_lossless), encode_prompt(clip_original))
    assert not torch.equal(encode_prompt(clip_lossless), encode_prompt(clip))


def test_lossless_lora_can_change_only_the_model(tiny_diffusion_model, tiny_lora):
    compressed = compress_in_comfyui("loras", tiny_lora)
    model = nodes.UNETLoader().load_unet(tiny_diffusion_model, "default")[0]
    model_original = nodes.LoraLoaderModelOnly().load_lora_model_only(model, tiny_lora, 1.0)[0]
    model_lossless, clip = LoadLoraLossless.execute(model, compressed, 1.0, 1.0).args
    assert clip is None
    assert torch.equal(sample(model_lossless, steps=3), sample(model_original, steps=3))


def test_lossless_vae_decodes_exactly_like_the_original(tiny_vae):
    compressed = compress_in_comfyui("vae", tiny_vae)
    original = nodes.VAELoader().load_vae(tiny_vae)[0]
    loaded = LoadVAELossless.execute(compressed).args[0]

    latent = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(0))
    image = original.decode(latent)
    assert torch.isfinite(image).all()
    assert torch.equal(loaded.decode(latent), image)
    assert loaded.patcher.cached_patcher_init is not None


def test_lossless_models_can_be_compressed_to_fp8(tiny_diffusion_model):
    compressed = compress_in_comfyui("diffusion_models", tiny_diffusion_model)
    fp8 = CompressModel.execute(LoadDiffusionModelLossless.execute(compressed).args[0], "fp8_e4m3fn").args[0]
    assert fp8.model_dtype() == torch.float8_e4m3fn
    assert torch.isfinite(sample(fp8, steps=2)).all()


def inner_layouts(model):
    """The ComfyUI quantized layouts of the model's compressed fp8 and int8 layers."""
    return {p._params.inner for p in model.model.diffusion_model.parameters() if memory.is_compressed(p) and p._params.inner}


def compressed_weights(model):
    """How many of the model's weights are stored losslessly compressed."""
    from comfy.quant_ops import QuantizedTensor
    return sum(isinstance(p, QuantizedTensor) and p._layout_cls == memory.LAYOUT for p in model.model.diffusion_model.parameters())


def with_lora(model):
    patched = model.clone()
    weight = patched.model.state_dict()["diffusion_model.out.2.weight"]
    assert patched.add_patches({"diffusion_model.out.2.weight": (torch.full(weight.shape, 0.05),)})
    return patched


def test_kept_compressed_model_samples_exactly_like_the_original_in_less_memory(tiny_diffusion_model):
    compressed = compress_in_comfyui("diffusion_models", tiny_diffusion_model)
    original = nodes.UNETLoader().load_unet(tiny_diffusion_model, "default")[0]
    kept = LoadDiffusionModelLossless.execute(compressed, True).args[0]

    assert torch.equal(sample(kept), sample(original))
    assert compressed_weights(kept) > 0
    assert kept.loaded_size() < original.loaded_size() * 0.9


def test_kept_compressed_model_applies_loras_exactly(tiny_diffusion_model):
    compressed = compress_in_comfyui("diffusion_models", tiny_diffusion_model)
    original = nodes.UNETLoader().load_unet(tiny_diffusion_model, "default")[0]
    kept = LoadDiffusionModelLossless.execute(compressed, True).args[0]

    unpatched = compressed_weights(kept)
    lora = with_lora(kept)
    with_lora_output = sample(lora, steps=3)
    assert compressed_weights(lora) == unpatched  # the LoRA-patched weight is stored compressed too
    assert torch.equal(with_lora_output, sample(with_lora(original), steps=3))
    assert not torch.equal(with_lora_output, sample(kept, steps=3))
    assert compressed_weights(kept) == unpatched


def test_kept_compressed_checkpoint_samples_exactly_like_the_original(tiny_checkpoint):
    compressed = compress_in_comfyui("checkpoints", tiny_checkpoint)
    original = LoadCheckpointFP8.execute(tiny_checkpoint, "default").args[0]
    kept = LoadCheckpointLossless.execute(compressed, True).args[0]
    assert compressed_weights(kept) > 0
    assert torch.equal(sample(kept), sample(original))


def test_models_that_cannot_load_compressed_are_compressed_after_loading(tiny_diffusion_model, monkeypatch):
    # Simulates layers that can't take a compressed weight while loading: they load decoded, then get compressed.
    monkeypatch.setattr(memory.CompressedWeight, "_load_from_state_dict", torch.nn.Module._load_from_state_dict)
    monkeypatch.setattr(memory.LoadingTensor, "dequantize", lambda self: (_ for _ in ()).throw(RuntimeError("can't decode here")))
    compressed = compress_in_comfyui("diffusion_models", tiny_diffusion_model)
    original = nodes.UNETLoader().load_unet(tiny_diffusion_model, "default")[0]
    loaded = LoadDiffusionModelLossless.execute(compressed, True).args[0]
    assert compressed_weights(loaded) > 0
    assert torch.equal(sample(loaded), sample(original))


def test_fp8_compression_refuses_kept_compressed_models(tiny_diffusion_model):
    compressed = compress_in_comfyui("diffusion_models", tiny_diffusion_model)
    kept = LoadDiffusionModelLossless.execute(compressed, True).args[0]
    with pytest.raises(ValueError, match="already kept compressed"):
        CompressModel.execute(kept, "fp8_e4m3fn")


def test_kept_compressed_models_reserve_memory_for_decoding(tiny_diffusion_model, monkeypatch):
    compressed = compress_in_comfyui("diffusion_models", tiny_diffusion_model)
    original = nodes.UNETLoader().load_unet(tiny_diffusion_model, "default")[0]
    kept = LoadDiffusionModelLossless.execute(compressed, True).args[0]
    asked = []
    load_models_gpu = comfy.model_management.load_models_gpu
    monkeypatch.setattr(comfy.model_management, "load_models_gpu",
                        lambda models, memory_required=0, *args, **kwargs: (asked.append(memory_required),
                                                                             load_models_gpu(models, memory_required, *args, **kwargs))[1])

    def required(model):
        asked.clear()
        sample(model, steps=1)
        return asked[0]

    reserved = kept.model.lossless_decode_memory
    assert reserved > 0
    assert required(kept) - required(original) == pytest.approx(reserved)  # what ComfyUI is told to keep free while sampling
    assert "memory_required" not in vars(kept.model)  # the estimate is only raised while the model loads
    assert not hasattr(original.model, "lossless_decode_memory")
    assert kept.clone().model.lossless_decode_memory == reserved


def test_decoded_models_reserve_nothing(tiny_diffusion_model):
    compressed = compress_in_comfyui("diffusion_models", tiny_diffusion_model)
    loaded = LoadDiffusionModelLossless.execute(compressed).args[0]
    assert not hasattr(loaded.model, "lossless_decode_memory")
    assert not loaded.get_wrappers(comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING, memory.RESERVE_KEY)


def test_compressed_weights_use_the_triton_decoder_when_they_carry_its_tables(monkeypatch):
    torch.manual_seed(0)
    weight = weights((512, 1024), torch.bfloat16)
    blob, info = codec.encode(weight)
    with_tables = dict(info, tables=blob.numel())
    launched = []
    monkeypatch.setattr(kernels, "usable", lambda device: True)
    monkeypatch.setattr(kernels, "decode", lambda blob, info: launched.append(1) or codec.decode(blob, info))

    assert same_bits(memory.decode(blob, info), weight) and launched == []  # no tables: the PyTorch decoder
    assert same_bits(memory.decode(blob, with_tables), weight) and launched == [1]

    monkeypatch.setattr(kernels, "decode", lambda blob, info: None)  # a kernel that can't run hands over to PyTorch
    assert same_bits(memory.decode(blob, with_tables), weight)


def test_loading_prepares_the_triton_tables_when_the_kernel_can_run(monkeypatch):
    torch.manual_seed(0)
    weight = weights((512, 1024), torch.bfloat16)
    blob, info = codec.encode(weight)
    assert "tables" not in memory.compressed_tensor(blob, info)._params.info  # nothing to prepare without a GPU

    monkeypatch.setattr(kernels, "usable", lambda device: True)
    monkeypatch.setattr(kernels, "decode", lambda blob, info: None)
    tensor = memory.compressed_tensor(blob, info, torch.device("cpu"))
    assert tensor._qdata.numel() > blob.numel() and tensor._params.info["tables"] >= blob.numel()
    assert same_bits(tensor.dequantize(), weight)  # the tables after the sections don't matter to the PyTorch decoder
    assert memory.decode_workspace([tensor], triton=True) == weight.nbytes
    assert memory.decode_workspace([tensor], triton=False) > weight.nbytes


def test_triton_decoder_is_exact_for_every_format():
    pytest.importorskip("triton")
    env = dict(os.environ)
    if not torch.cuda.is_available():
        env["TRITON_INTERPRET"] = "1"  # Triton's CPU interpreter
    result = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "tests", "triton_check.py")],
                            env=env, capture_output=True, text=True, timeout=1200)
    assert result.returncode == 0, result.stdout + result.stderr


def test_lossless_int8_convrot_file_loads_exactly(tiny_int8_convrot_model):
    compressed = compress_in_comfyui("diffusion_models", tiny_int8_convrot_model)
    sizes = [os.path.getsize(folder_paths.get_full_path("diffusion_models", name)) for name in (tiny_int8_convrot_model, compressed)]
    assert sizes[1] < sizes[0]
    original = nodes.UNETLoader().load_unet(tiny_int8_convrot_model, "default")[0]
    loaded = LoadDiffusionModelLossless.execute(compressed).args[0]
    assert torch.equal(sample(loaded), sample(original))

    # Its int8 layers stay int8 layers whose int8 data is kept compressed
    kept = LoadDiffusionModelLossless.execute(compressed, True).args[0]
    assert inner_layouts(kept) == {"TensorWiseINT8Layout"}
    assert torch.equal(sample(kept), sample(original))


def test_lossless_models_can_be_compressed_to_int8(tiny_diffusion_model):
    compressed = compress_in_comfyui("diffusion_models", tiny_diffusion_model)
    int8 = CompressModel.execute(LoadDiffusionModelLossless.execute(compressed).args[0], "int8_convrot").args[0]
    layouts = {p._layout_cls for p in int8.model.diffusion_model.parameters() if hasattr(p, "_layout_cls")}
    assert layouts == {"TensorWiseINT8Layout"}
    assert torch.isfinite(sample(int8, steps=2)).all()


def test_int8_compression_refuses_kept_compressed_models(tiny_diffusion_model):
    compressed = compress_in_comfyui("diffusion_models", tiny_diffusion_model)
    kept = LoadDiffusionModelLossless.execute(compressed, True).args[0]
    with pytest.raises(ValueError, match="already kept compressed"):
        CompressModel.execute(kept, "int8_convrot")


@pytest.fixture(params=["mixed", "scaled"])
def fp8_model(request):
    """ComfyUI's two fp8 file formats: mixed precision (comfy_quant), and the older "scaled" one."""
    return request.getfixturevalue(f"tiny_fp8_{request.param}_model")


def quantized_layer(model, layout):
    return next(m for m in model.model.diffusion_model.modules()
                if memory.is_compressed(getattr(m, "weight", None)) and m.weight._params.inner == layout)


def use_fp8_matmuls(model):
    """Makes fp8 layers run their fp8 matmul as on a GPU that has one (on the CPU they compute in full precision)."""
    for module in model.model.diffusion_model.modules():
        if hasattr(module, "_full_precision_mm"):
            module._full_precision_mm = False


def test_kept_compressed_fp8_file_keeps_its_fp8_layers_compressed(fp8_model, monkeypatch):
    compressed = compress_in_comfyui("diffusion_models", fp8_model)
    original = nodes.UNETLoader().load_unet(fp8_model, "default")[0]
    kept = LoadDiffusionModelLossless.execute(compressed, True).args[0]

    assert inner_layouts(kept) == {"TensorCoreFP8E4M3Layout"}
    assert kept.model_size() < original.model_size() * 0.9
    assert torch.equal(sample(kept), sample(original))
    # The fp8 matmul gets the layer's own fp8 weight back, and computes exactly the same.
    use_fp8_matmuls(original)
    use_fp8_matmuls(kept)
    layer_class = type(quantized_layer(kept, "TensorCoreFP8E4M3Layout")).__mro__[2]
    forward, seen = layer_class._forward, set()

    def watch(self, input, weight, bias):
        seen.add(weight._layout_cls if hasattr(weight, "_layout_cls") else None)
        return forward(self, input, weight, bias)
    monkeypatch.setattr(layer_class, "_forward", watch)
    assert torch.equal(sample(kept), sample(original))
    assert seen == {"TensorCoreFP8E4M3Layout"}


def test_kept_compressed_fp8_layers_apply_loras_exactly(tiny_fp8_mixed_model):
    compressed = compress_in_comfyui("diffusion_models", tiny_fp8_mixed_model)
    original = nodes.UNETLoader().load_unet(tiny_fp8_mixed_model, "default")[0]
    kept = LoadDiffusionModelLossless.execute(compressed, True).args[0]
    key = "diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"

    def lora(model):
        patched = model.clone()
        assert patched.add_patches({key: (torch.full((320, 320), 0.05),)})
        return patched
    lora_kept = lora(kept)
    output = sample(lora_kept, steps=3)
    assert torch.equal(output, sample(lora(original), steps=3))  # requantized to fp8 the same way
    assert not torch.equal(output, sample(kept, steps=3))
    assert memory.is_compressed(comfy.utils.get_attr(lora_kept.model, key))  # baked in, then compressed again


def test_offloaded_fp8_layers_get_loras_in_their_own_format(tiny_fp8_mixed_model):
    # A LoRA on a layer that doesn't fit in VRAM is applied each step; the result goes to the fp8 kernel uncompressed.
    compressed = compress_in_comfyui("diffusion_models", tiny_fp8_mixed_model)
    original = nodes.UNETLoader().load_unet(tiny_fp8_mixed_model, "default")[0]
    kept = LoadDiffusionModelLossless.execute(compressed, True).args[0]
    name = "input_blocks.1.1.transformer_blocks.0.attn1.to_q"
    kept_layer = comfy.utils.get_attr(kept.model.diffusion_model, name)
    original_layer = comfy.utils.get_attr(original.model.diffusion_model, name)
    patched = original_layer.weight.dequantize() + 0.05

    ours = kept_layer.set_weight(patched.clone(), seed=7, return_weight=True)
    theirs = original_layer.set_weight(patched.clone(), seed=7, return_weight=True)
    assert ours._layout_cls == theirs._layout_cls == "TensorCoreFP8E4M3Layout"
    assert torch.equal(ours._qdata.view(torch.uint8), theirs._qdata.view(torch.uint8))
    assert memory.is_compressed(kept_layer.weight)  # the layer itself is untouched


def test_kept_compressed_int8_layers_save_their_convrot_settings(tiny_int8_convrot_model):
    compressed = compress_in_comfyui("diffusion_models", tiny_int8_convrot_model)
    kept = LoadDiffusionModelLossless.execute(compressed, True).args[0]
    layer = quantized_layer(kept, "TensorWiseINT8Layout")
    state = layer.state_dict()
    marker = json.loads(bytes(state["comfy_quant"].tolist()))
    assert marker["convrot"] is True and marker["convrot_groupsize"] in (16, 64, 256)
    assert state["weight"].dtype == torch.int8 and state["weight"].shape == layer.weight.shape
    assert state["weight"].nbytes < layer.weight.numel()  # measured by its compressed size


def test_layers_the_loader_does_not_expect_get_decoded_weights(tiny_unet_weights, tmp_path, monkeypatch, caplog):
    # Some models have a layer of a kind no lossless op replaces (such as a plain torch Embedding). It must get its
    # decoded weight instead of making the whole model load decoded.
    import comfy.ldm.modules.diffusionmodules.openaimodel as openaimodel
    init = openaimodel.UNetModel.__init__

    def with_embedding(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.extra_embedding = torch.nn.Embedding(3, 5120)
    monkeypatch.setattr(openaimodel.UNetModel, "__init__", with_embedding)
    state_dict = {**tiny_unet_weights, "extra_embedding.weight": torch.randn(3, 5120) * 0.02}
    comfy.utils.save_torch_file(state_dict, str(tmp_path / "with_embedding.safetensors"))
    folder_paths.add_model_folder_path("diffusion_models", str(tmp_path))
    compressed = compress_in_comfyui("diffusion_models", "with_embedding.safetensors")

    with caplog.at_level("WARNING"):
        kept = LoadDiffusionModelLossless.execute(compressed, True).args[0]
    assert "failed" not in caplog.text
    assert compressed_weights(kept) > 0
    assert torch.equal(kept.model.diffusion_model.extra_embedding.weight, state_dict["extra_embedding.weight"])


def test_loading_tensors_survive_comfyui_state_dict_conversions():
    torch.manual_seed(0)
    q, k = weights((64, 128), torch.bfloat16), weights((64, 128), torch.bfloat16)
    loading = [memory.compressed_tensor(*codec.encode(t)) for t in (q, k)]

    # How ComfyUI joins weights of diffusers-format files: an empty tensor like the first piece, grown as needed,
    # with each piece copied into its slice.
    joined = None
    for weight, start in zip(loading, (0, 64)):
        old = joined if joined is not None else torch.empty_like(weight)
        if old.shape[0] < start + 64:
            grown = torch.empty([start + 64, 128], device=weight.device, dtype=weight.dtype)
            grown[:old.shape[0]] = old
            old = grown
        old.narrow(0, start, 64)[:] = weight
        joined = old
    assert not isinstance(joined, QuantizedTensor) and same_bits(joined, torch.cat([q, k]))

    # ComfyUI's lazy layer loading (clone, then a Parameter) and plain layers copying the weight into their own.
    assert memory.is_compressed(torch.nn.Parameter(loading[0].clone(), requires_grad=False))
    plain = torch.nn.Linear(128, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        plain.weight.copy_(loading[0])
    assert same_bits(plain.weight.data, q)


def test_keep_model_compressed_on_a_regular_model_file(tiny_diffusion_model):
    original = nodes.UNETLoader().load_unet(tiny_diffusion_model, "default")[0]
    kept = KeepModelCompressed.execute(original).args[0]

    assert compressed_weights(kept) > 0 and compressed_weights(original) == 0  # the input is left alone
    assert kept.model_size() < original.model_size() * 0.9
    assert kept.model.lossless_decode_memory > 0
    assert torch.equal(sample(kept), sample(original))
    assert compressed_weights(kept.clone(force_deepcopy=True)) > 0  # reloads (torch.compile, multi-GPU) are compressed too
    assert KeepModelCompressed.execute(kept).args[0] is kept


def test_keep_model_compressed_keeps_loras(tiny_diffusion_model):
    original = nodes.UNETLoader().load_unet(tiny_diffusion_model, "default")[0]
    patched = with_lora(original)
    kept = KeepModelCompressed.execute(patched).args[0]
    assert kept.patches.keys() == patched.patches.keys()
    assert torch.equal(sample(kept, steps=3), sample(patched, steps=3))


def test_keep_model_compressed_after_compress_model_int8(tiny_diffusion_model):
    int8 = CompressModel.execute(nodes.UNETLoader().load_unet(tiny_diffusion_model, "default")[0], "int8_convrot").args[0]
    kept = KeepModelCompressed.execute(int8).args[0]
    assert inner_layouts(kept) == {"TensorWiseINT8Layout"}
    assert kept.model_size() < int8.model_size()
    assert torch.equal(sample(kept, steps=2), sample(int8, steps=2))
    assert CompressModel.execute(kept, "fp8_e4m3fn").args[0] is kept  # already quantized: passed through


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", list(codec.FORMATS), ids=str)
def test_encodes_and_decodes_exactly_on_each_device(dtype, device):
    torch.manual_seed(0)
    patterns = every_bit_pattern(dtype)
    tensor = torch.cat([weights(10 * patterns.numel() + 5, dtype), patterns]).view(-1, 1)
    blob, info = codec.encode(tensor.to(device))
    assert blob.device.type == device
    assert same_bits(codec.decode(blob, info).cpu(), tensor)
    assert same_bits(codec.decode(blob.cpu(), info), tensor)  # blobs decode the same anywhere


@pytest.mark.parametrize("device", DEVICES)
def test_kept_compressed_layer_moves_and_runs_exactly(device):
    from comfy.quant_ops import QuantizedTensor
    from lossless import memory
    torch.manual_seed(0)
    weight = (torch.randn(512, 1024) * 0.02).to(torch.bfloat16)
    layer = memory.LosslessOps.Linear(1024, 512, dtype=torch.bfloat16)
    layer.weight = torch.nn.Parameter(memory.compressed_tensor(*codec.encode(weight)), requires_grad=False)
    layer.bias = torch.nn.Parameter(torch.zeros(512, dtype=torch.bfloat16), requires_grad=False)
    layer.to(device)

    assert isinstance(layer.weight, QuantizedTensor) and layer.weight.device.type == device
    x = torch.randn(4, 1024, dtype=torch.bfloat16, device=device)
    assert torch.equal(layer(x), torch.nn.functional.linear(x, weight.to(device), layer.bias))
