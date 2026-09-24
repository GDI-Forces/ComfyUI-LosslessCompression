"""ComfyUI nodes for losslessly compressed model files.

The loaders decode the file back into the exact original weights and hand them to
ComfyUI's normal loading code, so a compressed model behaves bit for bit like the
original. Each loader records how to reload its model, so torch.compile, multi-GPU
and Compress Model (FP8) work with it like with the built-in loaders.
"""
import logging
import os
import uuid

import comfy.model_management
import comfy.patcher_extension
import comfy.sd
import comfy.utils
import folder_paths
import nodes
from comfy_api.latest import io, ui

from . import fileformat, memory

CATEGORY = "optimization/lossless"
COMPRESSIBLE_FOLDERS = ["diffusion_models", "checkpoints", "text_encoders", "loras", "vae"]
MARKER = ".lossless."


def compressed_files(folder):
    return [name for name in folder_paths.get_filename_list(folder) if MARKER in name]


def decode_device():
    """The GPU when there is one: encoding and decoding there is much faster than on the CPU."""
    device = comfy.model_management.get_torch_device()
    return None if device.type == "cpu" else device


def load_state_dict(path):
    return fileformat.load(path, decode_device())


KEEP_COMPRESSED = "lossless_keep_compressed"  # model option: keep the weights compressed in RAM and VRAM


def keep_compressed_options(keep_compressed):
    return {KEEP_COMPRESSED: True} if keep_compressed else {}


def keep_weights_compressed(model, name):
    """Compresses a freshly loaded model's weights in memory and reports the saving."""
    before, after, layers = memory.keep_compressed(model, decode_device())
    model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING, KEEP_COMPRESSED, memory.reserve_decode_memory)
    device = comfy.model_management.get_torch_device()
    if layers and device.type == "cuda" and not memory.kernels.usable(device):
        logging.info("Lossless: install Triton (triton-windows on Windows) to decode compressed weights much faster.")
    if layers == 0:
        logging.warning(f"{name}: no weights could be kept compressed; int8 and 4-bit layers don't compress.")
    else:
        logging.info(f"{name}: {layers} layer weights kept compressed, {before / 1024 ** 3:.2f} GB -> "
                     f"{after / 1024 ** 3:.2f} GB ({100 * (1 - after / before):.1f}% less memory)")


def split_options(model_options, disable_dynamic):
    """(ComfyUI's model options, disable_dynamic, keep compressed?). Compressed weights are only used with
    ComfyUI's classic memory manager, which is what they are tested with, not with DynamicVRAM."""
    keep = model_options.get(KEEP_COMPRESSED, False)
    return {k: v for k, v in model_options.items() if k != KEEP_COMPRESSED}, disable_dynamic or keep, keep


def load_diffusion_model(path, model_options={}, disable_dynamic=False):
    options, disable_dynamic, keep = split_options(model_options, disable_dynamic)
    state_dict, metadata = load_state_dict(path)
    model = comfy.sd.load_diffusion_model_state_dict(state_dict, model_options=options, metadata=metadata, disable_dynamic=disable_dynamic)
    if model is None:
        raise RuntimeError(f"Could not detect the model type of {path}")
    if keep:
        keep_weights_compressed(model, os.path.basename(path))
    model.cached_patcher_init = (load_diffusion_model, (path, model_options))
    return model


def load_checkpoint(path, output_vae=True, output_clip=True, embedding_directory=None, model_options={}, disable_dynamic=False):
    options, disable_dynamic, keep = split_options(model_options, disable_dynamic)
    state_dict, metadata = load_state_dict(path)
    out = comfy.sd.load_state_dict_guess_config(state_dict, output_vae=output_vae, output_clip=output_clip,
                                                embedding_directory=embedding_directory, model_options=options,
                                                metadata=metadata, disable_dynamic=disable_dynamic)
    if out is None:
        raise RuntimeError(f"Could not detect the model type of {path}")
    if keep:
        keep_weights_compressed(out[0], os.path.basename(path))
    out[0].cached_patcher_init = (load_checkpoint, (path, False, False, embedding_directory, model_options), 0)
    return out


def reload_keeping_compressed(init, disable_dynamic=False):
    """Reload recipe for models from Keep Model Compressed: the original loader, then compression."""
    loader, args, *index = init
    model = loader(*args, disable_dynamic=True)
    if index:
        model = model[index[0]]
    keep_weights_compressed(model, "Keep Model Compressed")
    return model


def load_text_encoder(paths, embedding_directory=None, clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION, model_options={}, disable_dynamic=False):
    state_dicts = []
    for path in paths:
        state_dict, metadata = load_state_dict(path)
        state_dicts.append(comfy.utils.convert_old_quants(state_dict, model_prefix="", metadata=metadata)[0])
    clip = comfy.sd.load_text_encoder_state_dicts(state_dicts, embedding_directory=embedding_directory, clip_type=clip_type,
                                                  model_options=model_options, disable_dynamic=disable_dynamic)
    clip.patcher.cached_patcher_init = (load_text_encoder_patcher, (paths, embedding_directory, clip_type, model_options))
    return clip


def load_text_encoder_patcher(paths, embedding_directory=None, clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION, model_options={}, disable_dynamic=False):
    return load_text_encoder(paths, embedding_directory, clip_type, model_options, disable_dynamic).patcher


def load_vae(path, device=None):
    state_dict, metadata = load_state_dict(path)
    vae = comfy.sd.VAE(sd=state_dict, metadata=metadata, device=device)
    vae.throw_exception_if_invalid()
    vae.patcher.cached_patcher_init = (load_vae_patcher, (path, device))
    return vae


def load_vae_patcher(path, device=None, disable_dynamic=False):
    return load_vae(path, device).patcher


class LoadDiffusionModelLossless(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LosslessLoadDiffusionModel",
            display_name="Load Diffusion Model (Lossless)",
            category=CATEGORY,
            description="Loads a diffusion model file made by Compress Model File (Lossless). The weights are decoded "
                        "back to exactly the original bits, so results are identical to the original file.",
            inputs=[
                io.Combo.Input("unet_name", options=compressed_files("diffusion_models")),
                io.Boolean.Input("keep_compressed", default=False,
                                 tooltip="Experimental. Keep the weights compressed in RAM and VRAM and decode each layer as it runs: "
                                         "the model takes as much memory as the file, but every step is slower (much less so with "
                                         "Triton installed). Off: decode once "
                                         "when loading, which only saves disk space."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, unet_name, keep_compressed=False) -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        return io.NodeOutput(load_diffusion_model(path, keep_compressed_options(keep_compressed)))


class LoadCheckpointLossless(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LosslessLoadCheckpoint",
            display_name="Load Checkpoint (Lossless)",
            category=CATEGORY,
            description="Loads a checkpoint made by Compress Model File (Lossless), with exactly the original weights.",
            inputs=[
                io.Combo.Input("ckpt_name", options=compressed_files("checkpoints")),
                io.Boolean.Input("keep_compressed", default=False,
                                 tooltip="Experimental. Keep the weights compressed in RAM and VRAM and decode each layer as it runs: "
                                         "the model takes as much memory as the file, but every step is slower (much less so with "
                                         "Triton installed). Off: decode once "
                                         "when loading, which only saves disk space."),
            ],
            outputs=[io.Model.Output(), io.Clip.Output(), io.Vae.Output()],
        )

    @classmethod
    def execute(cls, ckpt_name, keep_compressed=False) -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        model, clip, vae, _ = load_checkpoint(path, embedding_directory=folder_paths.get_folder_paths("embeddings"),
                                              model_options=keep_compressed_options(keep_compressed))
        return io.NodeOutput(model, clip, vae)


class LoadCLIPLossless(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LosslessLoadCLIP",
            display_name="Load CLIP (Lossless)",
            category=CATEGORY,
            description="Loads a text encoder made by Compress Model File (Lossless), with exactly the original weights. "
                        "The type works like in the regular Load CLIP node.",
            inputs=[
                io.Combo.Input("clip_name", options=compressed_files("text_encoders")),
                io.Combo.Input("type", options=nodes.CLIPLoader.INPUT_TYPES()["required"]["type"][0]),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, clip_name, type) -> io.NodeOutput:
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        return io.NodeOutput(load_text_encoder([path], folder_paths.get_folder_paths("embeddings"), clip_type))


class KeepModelCompressed(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LosslessKeepModelCompressed",
            display_name="Keep Model Compressed (Lossless VRAM)",
            category=CATEGORY,
            search_aliases=["lossless vram", "compress vram", "low vram", "keep compressed"],
            description="Keeps the model's weights losslessly compressed in RAM and VRAM and decodes each layer when it "
                        "runs. Results are exactly the same; the weights need less memory (about 31% less for bf16, 22-26% for fp8, "
                        "16% for fp32, 13% for fp16 and for ComfyUI's pre-quantized fp8 files) but every step is slower; "
                        "install Triton to decode with a single GPU kernel. Works after any core loader; int8 and 4-bit layers "
                        "are left as they are.",
            inputs=[io.Model.Input("model")],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        if any(memory.is_compressed(p) for p in model.model.diffusion_model.parameters()):
            return io.NodeOutput(model)
        if model.cached_patcher_init is None:
            raise ValueError("Keep Model Compressed needs its own copy of the model, but this model's loader doesn't "
                             "record how it was loaded. It works with Load Diffusion Model, Load Checkpoint and this pack's loaders.")
        # A private copy on the classic memory manager, with this model's LoRAs and patches.
        kept = model.clone(disable_dynamic=True, force_deepcopy=True)
        kept.hook_backup = {}
        kept.parent = None
        kept.clone_base_uuid = uuid.uuid4()  # its weights are no longer the input model's
        keep_weights_compressed(kept, "Keep Model Compressed")
        kept.cached_patcher_init = (reload_keeping_compressed, (model.cached_patcher_init,))
        return io.NodeOutput(kept)


class LoadLoraLossless(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LosslessLoadLora",
            display_name="Load LoRA (Lossless)",
            category=CATEGORY,
            description="Applies a LoRA made by Compress Model File (Lossless), with exactly the original weights. "
                        "Works like the regular Load LoRA node; leave clip unconnected to change only the model.",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip", optional=True),
                io.Combo.Input("lora_name", options=compressed_files("loras")),
                io.Float.Input("strength_model", default=1.0, min=-100.0, max=100.0, step=0.01),
                io.Float.Input("strength_clip", default=1.0, min=-100.0, max=100.0, step=0.01),
            ],
            outputs=[io.Model.Output(), io.Clip.Output()],
        )

    @classmethod
    def execute(cls, model, lora_name, strength_model, strength_clip, clip=None) -> io.NodeOutput:
        if strength_model == 0 and (clip is None or strength_clip == 0):
            return io.NodeOutput(model, clip)
        lora, metadata = load_state_dict(folder_paths.get_full_path_or_raise("loras", lora_name))
        return io.NodeOutput(*comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip, lora_metadata=metadata))


class LoadVAELossless(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LosslessLoadVAE",
            display_name="Load VAE (Lossless)",
            category=CATEGORY,
            description="Loads a VAE made by Compress Model File (Lossless), with exactly the original weights.",
            inputs=[io.Combo.Input("vae_name", options=compressed_files("vae"))],
            outputs=[io.Vae.Output()],
        )

    @classmethod
    def execute(cls, vae_name) -> io.NodeOutput:
        return io.NodeOutput(load_vae(folder_paths.get_full_path_or_raise("vae", vae_name)))


class CompressModelFile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        options = [f"{folder}/{name}" for folder in COMPRESSIBLE_FOLDERS
                   for name in folder_paths.get_filename_list(folder) if MARKER not in name]
        return io.Schema(
            node_id="LosslessCompressModelFile",
            display_name="Compress Model File (Lossless)",
            category=CATEGORY,
            description="Writes a smaller copy of a model file next to it, named <name>.lossless.safetensors, and checks "
                        "that it decodes to exactly the original weights. Load it with the (Lossless) loaders; press R to "
                        "refresh their lists. The original file is left alone; delete it yourself once you're happy.",
            inputs=[io.Combo.Input("model_file", options=options)],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, model_file):
        folder, name = model_file.split("/", 1)
        path = folder_paths.get_full_path(folder, name)
        return (os.path.getmtime(path), os.path.getsize(path)) if path else None

    @classmethod
    def execute(cls, model_file) -> io.NodeOutput:
        folder, name = model_file.split("/", 1)
        src = folder_paths.get_full_path_or_raise(folder, name)
        dst = os.path.splitext(src)[0] + ".lossless.safetensors"
        progress = comfy.utils.ProgressBar(1)
        original, written = fileformat.compress_file(src, dst, on_tensor=lambda done, total: progress.update_absolute(done, total),
                                                     device=decode_device())
        problems = fileformat.verify(src, dst, decode_device())
        if problems:
            os.remove(dst)
            raise RuntimeError(f"The compressed file didn't decode exactly ({len(problems)} tensors, e.g. {problems[:3]}); "
                               "it was deleted and the original is untouched.")
        text = (f"{os.path.basename(dst)}\n{original / 1024 ** 3:.2f} GB -> {written / 1024 ** 3:.2f} GB "
                f"({100 * (1 - written / original):.1f}% smaller), verified bit-exact.")
        return io.NodeOutput(ui=ui.PreviewText(text))


NODES = [CompressModelFile, LoadDiffusionModelLossless, LoadCheckpointLossless, LoadCLIPLossless, LoadLoraLossless,
         LoadVAELossless, KeepModelCompressed]
