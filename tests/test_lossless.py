import gc
import json
import os
import subprocess
import sys
import weakref

import pytest
import safetensors
import safetensors.torch
import torch

import comfy.model_management
import comfy.ops
import comfy.sd
import comfy.utils
import folder_paths
import nodes
from conftest import REPO_ROOT, sample
from lossless import codec, fileformat, kernels, memory
from lossless.nodes import (CompressModelFile, KeepModelCompressed, LoadCheckpointLossless, LoadCLIPLossless,
                            LoadDiffusionModelLossless, LoadLoraLossless, LoadVAELossless)
from optimizer_nodes import CompressModel, LoadCheckpointFP8


def same_bits(a, b):
    return (a.dtype == b.dtype and a.shape == b.shape
            and torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)))


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
    weights = (torch.randn(10 * patterns.numel() + 5) * 0.02).to(dtype)
    tensor = torch.cat([weights, patterns])[torch.randperm(weights.numel() + patterns.numel())].view(-1, 1)

    blob, info = codec.encode(tensor)
    assert same_bits(codec.decode(blob, info), tensor)


@pytest.mark.parametrize("dtype, saving", [(torch.bfloat16, 0.25), (torch.float32, 0.12), (torch.float16, 0.08),
                                           (torch.float8_e4m3fn, 0.15), (torch.float8_e5m2, 0.15)], ids=str)
def test_weights_get_smaller(dtype, saving):
    torch.manual_seed(0)
    tensor = (torch.randn(512, 1024) * 0.02).to(dtype)
    blob, _ = codec.encode(tensor)
    assert blob.numel() < tensor.nbytes * (1 - saving)


def test_tensors_that_would_not_shrink_are_left_alone():
    assert codec.encode(torch.tensor([0.5], dtype=torch.bfloat16)) is None
    assert codec.encode(torch.arange(10000)) is None


@pytest.fixture
def model_file(tmp_path):
    torch.manual_seed(0)
    tensors = {
        "blocks.0.weight": (torch.randn(256, 300) * 0.02).to(torch.bfloat16),
        "blocks.0.fp8": (torch.randn(128, 64) * 0.05).to(torch.float8_e4m3fn),
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
    sizes = {"F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "I64": 8, "U8": 1, "BOOL": 1}
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


def compressed_weights(model):
    return sum(memory.is_compressed(p) for p in model.model.diffusion_model.parameters())


def with_lora(model, key="diffusion_model.out.2.weight"):
    patched = model.clone()
    weight = patched.model.state_dict()[key]
    assert patched.add_patches({key: (torch.full(weight.shape, 0.05),)})
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


@pytest.mark.parametrize("dtype", list(codec.FORMATS), ids=str)
def test_keep_model_compressed_works_for_every_weight_dtype(tiny_diffusion_model, dtype):
    path = folder_paths.get_full_path("diffusion_models", tiny_diffusion_model)
    original = comfy.sd.load_diffusion_model(path, model_options={"dtype": dtype})
    assert original.model_dtype() == dtype
    kept = KeepModelCompressed.execute(original).args[0]

    assert compressed_weights(kept) > 0 and compressed_weights(original) == 0  # the input model is left alone
    assert torch.equal(sample(kept), sample(original))
    assert kept.loaded_size() < original.loaded_size()


def test_keep_model_compressed_on_a_prequantized_fp8_model(tiny_fp8_quantized_model, monkeypatch):
    from comfy.quant_ops import QuantizedTensor
    original = nodes.UNETLoader().load_unet(tiny_fp8_quantized_model, "default")[0]
    assert original.model.model_config.quant_config is not None
    kept = KeepModelCompressed.execute(original).args[0]

    wrapped = [p for p in kept.model.diffusion_model.parameters() if memory.is_compressed(p) and p._params.inner]
    assert wrapped and all(p._params.inner == "TensorCoreFP8E4M3Layout" for p in wrapped)
    assert torch.equal(sample(kept), sample(original))
    assert kept.loaded_size() < original.loaded_size()

    # On GPUs the fp8 layers run an fp8 kernel, which must get ComfyUI's own fp8 weight back. The CPU makes these
    # layers compute in full precision, so switch them to the fp8 path to check it.
    for model in (original, kept):
        for module in model.model.diffusion_model.modules():
            if hasattr(module, "_full_precision_mm"):
                module._full_precision_mm = False
    weights_seen, linear = [], torch.nn.functional.linear

    def spy(input, weight, bias=None):
        if isinstance(input, QuantizedTensor):
            weights_seen.append(weight._layout_cls)
        return linear(input, weight, bias)
    monkeypatch.setattr(torch.nn.functional, "linear", spy)
    fp8_path_output = sample(kept, steps=2)
    assert weights_seen and set(weights_seen) == {"TensorCoreFP8E4M3Layout"}
    assert torch.equal(fp8_path_output, sample(original, steps=2))
    monkeypatch.undo()

    quantized_layer = "diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"
    assert torch.equal(sample(with_lora(kept, quantized_layer), steps=3), sample(with_lora(original, quantized_layer), steps=3))

    # Offloaded layers apply LoRAs on the fly each step: they get ComfyUI's own fp8 weight, not a compressed one.
    layers = [comfy.utils.get_attr(model.model, quantized_layer.removesuffix(".weight")) for model in (kept, original)]
    patched = torch.randn(layers[0].weight.shape)
    on_the_fly = [layer.set_weight(patched.clone(), seed=7, return_weight=True) for layer in layers]  # (scaled in place)
    assert not memory.is_compressed(on_the_fly[0]) and on_the_fly[0]._layout_cls == on_the_fly[1]._layout_cls
    assert same_bits(on_the_fly[0]._qdata, on_the_fly[1]._qdata) and torch.equal(on_the_fly[0]._params.scale, on_the_fly[1]._params.scale)


def test_fp8_compression_goes_before_keep_model_compressed(tiny_diffusion_model):
    model = nodes.UNETLoader().load_unet(tiny_diffusion_model, "default")[0]
    fp8_kept = KeepModelCompressed.execute(CompressModel.execute(model, "fp8_e4m3fn").args[0]).args[0]
    assert fp8_kept.model_dtype() == torch.float8_e4m3fn and compressed_weights(fp8_kept) > 0
    assert torch.isfinite(sample(fp8_kept, steps=2)).all()

    with pytest.raises(ValueError, match="before that node"):
        CompressModel.execute(KeepModelCompressed.execute(model).args[0], "fp8_e4m3fn")


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", list(codec.FORMATS), ids=str)
def test_encodes_and_decodes_exactly_on_each_device(dtype, device):
    torch.manual_seed(0)
    patterns = every_bit_pattern(dtype)
    weights = (torch.randn(10 * patterns.numel() + 5) * 0.02).to(dtype)
    tensor = torch.cat([weights, patterns]).view(-1, 1)
    blob, info = codec.encode(tensor.to(device))
    assert blob.device.type == device
    assert same_bits(codec.decode(blob, info).cpu(), tensor)
    assert same_bits(codec.decode(blob.cpu(), info), tensor)  # blobs decode the same anywhere


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("ops", ["disable_weight_init", "manual_cast"])
def test_kept_compressed_layer_moves_and_runs_exactly(ops, device):
    torch.manual_seed(0)
    weight = (torch.randn(512, 1024) * 0.02).to(torch.bfloat16)
    bias = torch.zeros(512, dtype=torch.bfloat16)
    layer = getattr(comfy.ops, ops).Linear(1024, 512, dtype=torch.bfloat16)
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    layer.bias = torch.nn.Parameter(bias, requires_grad=False)
    assert memory.compress_module(layer) is not None
    layer.to(device)

    assert memory.is_compressed(layer.weight) and layer.weight.device.type == device
    x = torch.randn(4, 1024, dtype=torch.bfloat16, device=device)
    assert torch.equal(layer(x), torch.nn.functional.linear(x, weight.to(device), bias.to(device)))


def test_safetensors_reader_matches_the_safetensors_library(tmp_path):
    tensors = {
        "bf16": torch.randn(3, 5).to(torch.bfloat16), "fp8": torch.randn(7).to(torch.float8_e4m3fn),
        "f32": torch.randn(2, 2, 2), "i64": torch.arange(5), "mask": torch.tensor([True, False, True]),
        "scalar": torch.tensor(2.5), "empty": torch.zeros(0, 4),
    }
    path = str(tmp_path / "all.safetensors")
    safetensors.torch.save_file(tensors, path, metadata={"note": "x"})
    with fileformat.SafetensorsFile(path) as f:
        assert f.metadata == {"note": "x"}
        read = {k: f.get(k) for k in f.keys()}
    assert read.keys() == tensors.keys()
    assert all(same_bits(read[k], tensors[k]) for k in tensors)


def test_lossless_files_are_never_memory_mapped(tiny_diffusion_model, tiny_lora, tiny_vae, tmp_path, monkeypatch):
    # On Windows, memory-mapping a big file fails when the paging file can't cover it
    # ("The paging file is too small for this operation to complete. (os error 1455)").
    def refuse(*args, **kwargs):
        raise OSError("The paging file is too small for this operation to complete. (os error 1455)")
    monkeypatch.setattr(safetensors, "safe_open", refuse)
    model = compress_in_comfyui("diffusion_models", tiny_diffusion_model)
    lora = compress_in_comfyui("loras", tiny_lora)
    vae = compress_in_comfyui("vae", tiny_vae)

    for keep_compressed in (False, True):
        loaded = LoadDiffusionModelLossless.execute(model, keep_compressed).args[0]
        assert LoadLoraLossless.execute(loaded, lora, 1.0, 1.0).args[0] is not None
    assert LoadVAELossless.execute(vae).args[0] is not None
    path = folder_paths.get_full_path("diffusion_models", model)
    assert fileformat.verify(folder_paths.get_full_path("diffusion_models", tiny_diffusion_model), path) == []
    fileformat.decompress_file(path, str(tmp_path / "restored.safetensors"))


def test_kept_compressed_layers_are_freed_without_a_garbage_collection(tiny_diffusion_model):
    # Layers that reference themselves stay in memory (VRAM too) until Python's next full garbage collection.
    kept = LoadDiffusionModelLossless.execute(compress_in_comfyui("diffusion_models", tiny_diffusion_model), True).args[0]
    layers = [weakref.ref(m) for m in kept.model.diffusion_model.modules() if memory.is_compressed(getattr(m, "weight", None))]
    assert layers
    gc.collect()
    gc.disable()
    try:
        del kept
        assert all(layer() is None for layer in layers)
    finally:
        gc.enable()


def test_sampling_reserves_the_memory_decoding_needs(tiny_diffusion_model, monkeypatch):
    original = nodes.UNETLoader().load_unet(tiny_diffusion_model, "default")[0]
    kept = LoadDiffusionModelLossless.execute(compress_in_comfyui("diffusion_models", tiny_diffusion_model), True).args[0]
    requested, load_models_gpu = [], comfy.model_management.load_models_gpu

    def spy(models, memory_required=0, *args, **kwargs):
        requested.append(memory_required)
        return load_models_gpu(models, memory_required, *args, **kwargs)
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", spy)
    sample(original, steps=1)
    sample(kept, steps=1)

    assert kept.model.lossless_decode_memory > 0
    assert requested[1] == pytest.approx(requested[0] + kept.model.lossless_decode_memory)
    assert "memory_required" not in vars(kept.model)  # only changed while loading


class PeakMemory(torch.utils._python_dispatch.TorchDispatchMode):
    """Largest total size of the tensors created inside it that were alive at the same time. Tensors that already
    exist (the inputs) are passed in, since views of them look like new tensors."""

    def __init__(self, *existing):
        super().__init__()
        self.live = self.peak = 0
        self.storages = {tensor.untyped_storage().data_ptr() for tensor in existing}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for tensor in torch.utils._pytree.tree_flatten(out)[0]:
            if isinstance(tensor, torch.Tensor) and tensor.untyped_storage().nbytes():
                storage = tensor.untyped_storage()
                if storage.data_ptr() not in self.storages:
                    self.storages.add(storage.data_ptr())
                    self.live += storage.nbytes()
                    self.peak = max(self.peak, self.live)
                    weakref.finalize(storage, self.free, storage.data_ptr(), storage.nbytes())
        return out

    def free(self, pointer, size):
        self.live -= size
        self.storages.discard(pointer)


@pytest.mark.parametrize("escapes", ["nonzero_static", "cumsum"])
@pytest.mark.parametrize("shape", [(256, 4096), (2048, 5000)], ids=["one chunk", "two chunks"])
@pytest.mark.parametrize("dtype", list(codec.FORMATS), ids=str)
def test_decoding_stays_within_the_memory_reserved_for_it(dtype, shape, escapes, monkeypatch):
    # Decoding takes memory next to the model; if ComfyUI reserves less than that, a full GPU runs over.
    monkeypatch.setitem(codec._NONZERO_STATIC, "cpu", escapes == "nonzero_static")
    torch.manual_seed(0)
    weight = (torch.randn(shape) * (50 if dtype == torch.float8_e4m3fn else 0.02)).to(dtype)
    blob, params = memory.pack(weight, torch.ones(()), dtype, weight.shape, None)
    with PeakMemory(blob) as peak:
        decoded = codec.decode(blob, params.info)
    assert same_bits(decoded, weight)
    assert peak.peak <= memory.decode_memory(params.info)


@pytest.mark.skipif(not kernels.installed(), reason="needs Triton")
def test_triton_decoder_is_exact():
    env = dict(os.environ)
    if not torch.cuda.is_available():
        env["TRITON_INTERPRET"] = "1"  # Triton's CPU interpreter runs the same kernel
    result = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "tests", "triton_check.py")], env=env,
                            capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not kernels.installed(), reason="needs Triton")
@pytest.mark.parametrize("failure", ["wrong results", "error"])
def test_a_failing_triton_decoder_falls_back_to_exact_decoding(failure, monkeypatch, caplog):
    torch.manual_seed(0)
    weight = (torch.randn(512, 1024) * 0.02).to(torch.bfloat16)
    layer = comfy.ops.manual_cast.Linear(1024, 512, bias=False, dtype=torch.bfloat16)
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    memory.compress_module(layer)
    assert "tables" in layer.weight._params.info

    class FailingKernel:
        def __getitem__(self, grid):
            def launch(blob, luts, starts, out, *args, **kwargs):
                if failure == "error":
                    raise RuntimeError("no compiler")
                out.zero_()
            return launch
    # pytest also imports this package under its folder name, and either copy may be the one decoding
    copies = [module for name, module in list(sys.modules.items()) if name.endswith("lossless.kernels")]
    for copy in copies:
        monkeypatch.setattr(copy, "_decode_kernel", FailingKernel())
        monkeypatch.setattr(copy, "_verified", set())
        monkeypatch.setitem(copy._state, "broken", False)
    monkeypatch.setenv("TRITON_INTERPRET", "1")  # lets kernels.usable() pick the kernel on the CPU

    x = torch.randn(4, 1024, dtype=torch.bfloat16)
    assert torch.equal(layer(x), torch.nn.functional.linear(x, weight))
    assert any(copy._state["broken"] for copy in copies) and "slower PyTorch decoder" in caplog.text
