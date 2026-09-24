"""Nodes that lower VRAM use and sampling time through ComfyUI's own hooks.

Nothing here replaces ComfyUI's memory manager: the model is only ever patched
with wrappers, attention overrides and load options that ComfyUI itself owns,
so offloading, LoRAs and other model patches keep working.
"""
import gc
import inspect
import logging
import uuid

import torch

import comfy.model_management
import comfy.patcher_extension
import comfy.sd
import folder_paths
from comfy.ldm.modules import attention as comfy_attention
from comfy_api.latest import io
from comfy_api.torch_helpers import set_torch_compile_wrapper
from comfy_api.torch_helpers.torch_compile import COMPILE_KEY
from comfy_extras.nodes_easycache import EasyCacheNode
from comfy_extras.nodes_torch_compile import skip_torch_compile_dict

CATEGORY = "optimization"

# "auto" takes the first of these that is installed. Both fall back to PyTorch
# attention on their own for any call they can't handle.
AUTO_ATTENTION = ["sage", "flash"]
ATTENTION_OPTIONS = ["auto", "keep", "sage", "sage3", "flash", "xformers", "comfy_kitchen_int8", "pytorch"]

FP16_ACCUMULATION_KEY = "optimizer_fp16_accumulation"
COMPACT_VRAM_KEY = "optimizer_compact_vram"

WEIGHT_DTYPES = {
    "fp8_e4m3fn": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "default": None,
}
FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def pick_attention(choice):
    """Registered attention backend name for a node choice, or None to leave attention alone."""
    if choice == "keep":
        return None
    if choice == "auto":
        backend = next((name for name in AUTO_ATTENTION if comfy_attention.get_attention_function(name, None) is not None), None)
        if backend is None:
            logging.info("Speed Up Model: neither SageAttention nor FlashAttention is installed, keeping the current attention.")
        return backend
    if comfy_attention.get_attention_function(choice, None) is None:
        logging.warning(f"Speed Up Model: attention backend '{choice}' is not installed, keeping the current attention.")
        return None
    return choice


def fp16_accumulation_wrapper(executor, *args, **kwargs):
    """Turns on fp16 matmul accumulation only while this model runs, so other models are unaffected."""
    matmul = torch.backends.cuda.matmul
    previous = matmul.allow_fp16_accumulation
    matmul.allow_fp16_accumulation = True
    try:
        return executor(*args, **kwargs)
    finally:
        matmul.allow_fp16_accumulation = previous


def compact_vram_wrapper(unload_other_models):
    """Runs just before the sampler loads the model, so the load starts from as much free VRAM as possible."""
    def wrapper(executor, model, *args, **kwargs):
        device = model.load_device
        free_before = comfy.model_management.get_free_memory(device)
        if unload_other_models:
            # Clones share the weights of the model about to load; unloading them would unload it too.
            keep = [loaded for loaded in comfy.model_management.current_loaded_models
                    if loaded.model is not None and loaded.model.clone_base_uuid == model.clone_base_uuid]
            comfy.model_management.free_memory(1e30, device, keep_loaded=keep)
        gc.collect()
        comfy.model_management.soft_empty_cache()
        free_after = comfy.model_management.get_free_memory(device)
        logging.info(f"Compact VRAM: {gigabytes(free_before):.2f} GB -> {gigabytes(free_after):.2f} GB free on {device} before loading the model")
        return executor(model, *args, **kwargs)
    return wrapper


def compress_model(model, dtype):
    """A copy of the model whose weights are stored in `dtype`, keeping its LoRAs, patches and options.

    The weights are rebuilt through the loader that made the model, the same way ComfyUI's
    own weight_dtype options load them, and the input model is left untouched."""
    if model.model.model_config.quant_config is not None:
        logging.info("Compress Model: the model is already quantized (a ComfyUI mixed-precision file), leaving it as is.")
        return model
    if model.model_dtype() in FP8_DTYPES:
        logging.info(f"Compress Model: the model is already stored in {model.model_dtype()}, leaving it as is.")
        return model
    if model.get_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, COMPILE_KEY):
        raise ValueError("Compress Model must come before torch.compile (Speed Up Model's torch_compile or TorchCompileModel), "
                         "otherwise the compiled model keeps using the uncompressed weights.")
    if model.cached_patcher_init is None:
        raise ValueError("Compress Model can't rebuild this model because its loader doesn't record how it was loaded. "
                         "It works with models from Load Diffusion Model, Load Checkpoint and this pack's loaders. "
                         "GGUF models are already compressed and don't need it.")

    loader, loader_args, *output_index = model.cached_patcher_init
    bound = inspect.signature(loader).bind(*loader_args)
    if bound.arguments.get("model_options", {}).get("custom_operations") is not None:
        raise ValueError("Compress Model can't convert this model: it was loaded with its own weight storage "
                         "(for example kept compressed by a Lossless loader). Use one or the other.")
    bound.arguments["model_options"] = {**bound.arguments.get("model_options", {}), "dtype": dtype}
    loaded = loader(*bound.args, **bound.kwargs, disable_dynamic=not model.is_dynamic())
    if output_index:
        loaded = loaded[output_index[0]]

    # Same approach as ModelPatcher.deepclone_multigpu: this model's patches on fresh weights.
    compressed = model.clone(model_override=(loaded.model, ({}, {}, {}, set())))
    compressed.hook_backup = {}
    compressed.size = loaded.model_size()
    compressed.cached_patcher_init = loaded.cached_patcher_init
    # The weights differ from the input model's, so ComfyUI must not treat the two as sharing them.
    compressed.parent = None
    compressed.clone_base_uuid = uuid.uuid4()
    logging.info(f"Compress Model: {model.model_dtype()} -> {dtype} weights")
    return compressed


def compute_dtype(model):
    return model.get_model_object("manual_cast_dtype") or model.model_dtype()


def gigabytes(size):
    return size / (1024 ** 3)


class SpeedUpModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OptimizerSpeedUpModel",
            display_name="Speed Up Model",
            category=CATEGORY,
            search_aliases=["sage attention", "fp16 accumulation", "easycache", "torch compile", "faster sampling"],
            description="Makes image and video sampling faster. Connect it between the model loader (and LoRAs) and the sampler.",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("attention", options=ATTENTION_OPTIONS, default="auto",
                               tooltip="Attention implementation. auto: SageAttention if installed, else FlashAttention, else unchanged. "
                                       "keep: leave ComfyUI's choice alone. Backends that aren't installed are skipped with a warning."),
                io.Boolean.Input("fp16_accumulation", default=False,
                                 tooltip="Faster matrix multiplies on NVIDIA RTX GPUs with PyTorch 2.7+, for models that compute in fp16. "
                                         "Only affects this model while it samples. Can slightly change results."),
                io.Float.Input("step_cache", default=0.0, min=0.0, max=3.0, step=0.01,
                               tooltip="Reuses the model output on steps where it barely changes (ComfyUI's EasyCache). "
                                       "0 disables it. 0.1 to 0.25 usually skips a third or more of the steps with little quality loss; "
                                       "higher is faster but blurrier."),
                io.Boolean.Input("torch_compile", default=False,
                                 tooltip="Compiles the model with torch.compile (needs Triton). The first run takes minutes and "
                                         "recompiles when the resolution or frame count changes; later runs are faster."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, attention, fp16_accumulation, step_cache, torch_compile) -> io.NodeOutput:
        # torch.compile needs a model that isn't streamed in and out of VRAM layer by layer
        m = model.clone(disable_dynamic=torch_compile)
        applied = []

        backend = pick_attention(attention)
        if backend is not None:
            m.set_model_optimized_attention(comfy_attention.get_attention_function(backend))
            applied.append(f"{backend} attention")

        if fp16_accumulation:
            if not hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                logging.warning("Speed Up Model: fp16 accumulation needs PyTorch 2.7 or newer, skipping it.")
            else:
                dtype = compute_dtype(m)
                if m.load_device.type != "cuda":
                    logging.warning(f"Speed Up Model: fp16 accumulation only speeds up NVIDIA and AMD GPUs, this model runs on {m.load_device}.")
                elif dtype != torch.float16:
                    logging.warning(f"Speed Up Model: this model computes in {dtype}, so fp16 accumulation won't speed it up. "
                                    "Load fp16 weights or set the compute dtype to fp16 (ModelComputeDtype node).")
                m.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, FP16_ACCUMULATION_KEY)
                m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, FP16_ACCUMULATION_KEY, fp16_accumulation_wrapper)
                applied.append("fp16 accumulation")

        if step_cache > 0:
            m = EasyCacheNode.execute(m, reuse_threshold=step_cache, start_percent=0.15, end_percent=0.95, verbose=False).args[0]
            applied.append(f"step cache {step_cache}")

        if torch_compile:
            set_torch_compile_wrapper(model=m, backend="inductor", options={"guard_filter_fn": skip_torch_compile_dict})
            applied.append("torch.compile")

        logging.info(f"Speed Up Model: {', '.join(applied) or 'no changes'}")
        return io.NodeOutput(m)


class LoadCheckpointFP8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OptimizerLoadCheckpointFP8",
            display_name="Load Checkpoint (FP8, Low VRAM)",
            category=CATEGORY,
            search_aliases=["fp8 checkpoint", "low vram checkpoint", "load checkpoint"],
            description="Loads a checkpoint with the diffusion model's weights stored in fp8, "
                        "which halves the VRAM they take compared to fp16 or bf16.",
            inputs=[
                io.Combo.Input("ckpt_name", options=folder_paths.get_filename_list("checkpoints")),
                io.Combo.Input("weight_dtype", options=list(WEIGHT_DTYPES), default="fp8_e4m3fn",
                               tooltip="fp8_e4m3fn keeps more precision and suits most models. fp8_e5m2 keeps more range. "
                                       "default loads the weights like the regular checkpoint loader."),
            ],
            outputs=[io.Model.Output(), io.Clip.Output(), io.Vae.Output()],
        )

    @classmethod
    def execute(cls, ckpt_name, weight_dtype) -> io.NodeOutput:
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        model_options = {}
        if WEIGHT_DTYPES[weight_dtype] is not None:
            model_options["dtype"] = WEIGHT_DTYPES[weight_dtype]
        model, clip, vae = comfy.sd.load_checkpoint_guess_config(
            ckpt_path, output_vae=True, output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            model_options=model_options,
        )[:3]
        return io.NodeOutput(model, clip, vae)


class CompressModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OptimizerCompressModel",
            display_name="Compress Model (FP8)",
            category=CATEGORY,
            search_aliases=["fp8", "quantize model", "compress vram", "low vram"],
            description="Stores the model's weights in fp8, which halves the VRAM they need compared to fp16 or bf16. "
                        "Works on a model from any core loader, and keeps LoRAs and other patches already applied to it. "
                        "Put it right after the loader or LoRAs.",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("weight_dtype", options=[k for k, v in WEIGHT_DTYPES.items() if v is not None], default="fp8_e4m3fn",
                               tooltip="fp8_e4m3fn keeps more precision and suits most models. fp8_e5m2 keeps more range."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, weight_dtype) -> io.NodeOutput:
        return io.NodeOutput(compress_model(model, WEIGHT_DTYPES[weight_dtype]))


class CompactVRAMBeforeLoad(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OptimizerCompactVRAM",
            display_name="Compact VRAM Before Load",
            category=CATEGORY,
            search_aliases=["defragment vram", "clear vram before load", "out of memory", "oom"],
            description="Right before the sampler loads this model onto the GPU, unloads the other models and hands "
                        "PyTorch's cached memory back to the driver, so the model loads into the largest possible free "
                        "block of VRAM. Use it when a model runs out of memory or gets partly offloaded because other "
                        "models or leftover memory were in the way.",
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("unload_other_models", default=True,
                                 tooltip="Unload every model except this one (text encoders, VAEs, other diffusion models). "
                                         "They reload when next needed. Off: only clear the cached memory."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, unload_other_models) -> io.NodeOutput:
        m = model.clone()
        m.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING, COMPACT_VRAM_KEY)
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING, COMPACT_VRAM_KEY,
                               compact_vram_wrapper(unload_other_models))
        return io.NodeOutput(m)


class FreeVRAM(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OptimizerFreeVRAM",
            display_name="Free VRAM",
            category=CATEGORY,
            search_aliases=["unload models", "clear vram", "empty cache"],
            description="Passes its input through unchanged after unloading models and clearing the GPU cache. "
                        "Put it between stages, e.g. between the sampler and VAE Decode, so the next stage starts with free VRAM.",
            inputs=[
                io.AnyType.Input("value", tooltip="Anything; it is passed through unchanged."),
                io.Boolean.Input("unload_models", default=True,
                                 tooltip="Unload every model from VRAM. They reload when next used, which costs time on the next run."),
            ],
            outputs=[io.AnyType.Output(display_name="value")],
        )

    @classmethod
    def execute(cls, value, unload_models) -> io.NodeOutput:
        device = comfy.model_management.get_torch_device()
        free_before = comfy.model_management.get_free_memory(device)
        if unload_models:
            comfy.model_management.unload_all_models()
        gc.collect()
        comfy.model_management.soft_empty_cache()
        free_after = comfy.model_management.get_free_memory(device)
        logging.info(f"Free VRAM: {gigabytes(free_before):.2f} GB -> {gigabytes(free_after):.2f} GB free on {device}")
        return io.NodeOutput(value)


NODES = [SpeedUpModel, LoadCheckpointFP8, CompressModel, CompactVRAMBeforeLoad, FreeVRAM]
