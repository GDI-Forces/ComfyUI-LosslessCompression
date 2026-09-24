"""Runs the nodes against a real ComfyUI checkout on the CPU.

Point COMFYUI_PATH at a ComfyUI folder and run pytest from a Python environment
that has ComfyUI's requirements installed.
"""
import os
import sys

import pytest

COMFYUI_PATH = os.environ.get("COMFYUI_PATH")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if not COMFYUI_PATH:
    pytest.exit("Set COMFYUI_PATH to a ComfyUI checkout to run these tests.", returncode=4)

sys.path.insert(0, os.path.abspath(COMFYUI_PATH))
sys.path.insert(0, REPO_ROOT)

from comfy.cli_args import args  # noqa: E402

args.cpu = True

import torch  # noqa: E402

import comfy.sample  # noqa: E402
import comfy.supported_models  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402

# A two-level SD1.5 UNet with random weights: small enough for the CPU, but
# ComfyUI still detects it as SD1.5 and loads it through the normal code path.
TINY_SD15_UNET = {
    "use_checkpoint": False, "image_size": 32, "in_channels": 4, "out_channels": 4,
    "use_spatial_transformer": True, "legacy": False, "adm_in_channels": None,
    "model_channels": 320, "channel_mult": [1, 1], "num_res_blocks": [1, 1],
    "transformer_depth": [1, 0], "transformer_depth_output": [1, 1, 0, 0], "transformer_depth_middle": 1,
    "use_linear_in_transformer": False, "context_dim": 768,
    "use_temporal_attention": False, "use_temporal_resblock": False,
}


@pytest.fixture(scope="session")
def tiny_unet_weights():
    torch.manual_seed(0)
    model_config = comfy.supported_models.SD15(TINY_SD15_UNET)
    model_config.set_inference_dtype(torch.float32, None)
    unet = model_config.get_model({}).diffusion_model
    for p in unet.parameters():
        torch.nn.init.normal_(p, std=0.02)
    return unet.state_dict()


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory, tiny_unet_weights):
    """Checkpoint name for the regular Load Checkpoint path."""
    folder = tmp_path_factory.mktemp("checkpoints")
    state_dict = {f"model.diffusion_model.{k}": v for k, v in tiny_unet_weights.items()}
    comfy.utils.save_torch_file(state_dict, str(folder / "tiny_sd15.safetensors"))
    folder_paths.add_model_folder_path("checkpoints", str(folder))
    return "tiny_sd15.safetensors"


@pytest.fixture(scope="session")
def tiny_diffusion_model(tmp_path_factory, tiny_unet_weights):
    """Diffusion model file name for the Load Diffusion Model path, which most video models use."""
    folder = tmp_path_factory.mktemp("diffusion_models")
    comfy.utils.save_torch_file(dict(tiny_unet_weights), str(folder / "tiny_sd15_unet.safetensors"))
    folder_paths.add_model_folder_path("diffusion_models", str(folder))
    return "tiny_sd15_unet.safetensors"


@pytest.fixture(scope="session")
def tiny_text_encoder(tmp_path_factory):
    """A bf16 CLIP-L text encoder with random weights, in the text_encoders folder."""
    import comfy.sd1_clip
    torch.manual_seed(0)
    transformer = comfy.sd1_clip.SD1ClipModel(dtype=torch.float32).clip_l.transformer
    for p in transformer.parameters():  # ComfyUI leaves weights uninitialized
        torch.nn.init.normal_(p, std=0.02)
    folder = tmp_path_factory.mktemp("text_encoders")
    comfy.utils.save_torch_file({k: v.to(torch.bfloat16) for k, v in transformer.state_dict().items()},
                                str(folder / "tiny_clip_l.safetensors"))
    folder_paths.add_model_folder_path("text_encoders", str(folder))
    return "tiny_clip_l.safetensors"


@pytest.fixture(scope="session")
def tiny_lora(tmp_path_factory):
    """A bf16 rank-16 LoRA (kohya key names) for two attention layers of the tiny UNet and one of CLIP-L."""
    torch.manual_seed(0)
    layers = {  # LoRA name: (out features, in features)
        "lora_unet_input_blocks_1_1_transformer_blocks_0_attn1_to_q": (320, 320),
        "lora_unet_input_blocks_1_1_transformer_blocks_0_attn2_to_k": (320, 768),
        "lora_te_text_model_encoder_layers_0_self_attn_q_proj": (768, 768),
    }
    lora = {}
    for name, (out_features, in_features) in layers.items():
        lora[f"{name}.lora_up.weight"] = (torch.randn(out_features, 16) * 0.05).to(torch.bfloat16)
        lora[f"{name}.lora_down.weight"] = (torch.randn(16, in_features) * 0.05).to(torch.bfloat16)
        lora[f"{name}.alpha"] = torch.tensor(16.0)
    folder = tmp_path_factory.mktemp("loras")
    comfy.utils.save_torch_file(lora, str(folder / "tiny_lora.safetensors"), metadata={"ss_network_dim": "16"})
    folder_paths.add_model_folder_path("loras", str(folder))
    return "tiny_lora.safetensors"


@pytest.fixture(scope="session")
def tiny_vae(tmp_path_factory):
    """A bf16 VAE in the SD1.x format, with random weights."""
    from comfy.ldm.models.autoencoder import AutoencoderKL
    torch.manual_seed(0)
    ddconfig = {"double_z": True, "z_channels": 4, "resolution": 256, "in_channels": 3, "out_ch": 3, "ch": 128,
                "ch_mult": [1, 2, 4, 4], "num_res_blocks": 2, "attn_resolutions": [], "dropout": 0.0}
    vae = AutoencoderKL(ddconfig=ddconfig, embed_dim=4)
    for p in vae.parameters():  # ComfyUI leaves weights uninitialized
        torch.nn.init.normal_(p, std=0.02)
    folder = tmp_path_factory.mktemp("vae")
    comfy.utils.save_torch_file({k: v.to(torch.bfloat16) for k, v in vae.state_dict().items()}, str(folder / "tiny_vae.safetensors"))
    folder_paths.add_model_folder_path("vae", str(folder))
    return "tiny_vae.safetensors"


def sample(model, steps=8, seed=0):
    latent = torch.zeros(1, 4, 8, 8)
    noise = comfy.sample.prepare_noise(latent, seed)
    context = torch.randn(1, 77, 768, generator=torch.Generator().manual_seed(1))
    positive = [[context, {}]]
    negative = [[torch.zeros_like(context), {}]]
    return comfy.sample.sample(model, noise, steps, 5.0, "euler", "normal", positive, negative, latent,
                               seed=seed, disable_pbar=True)


@pytest.fixture
def watch_forward(monkeypatch):
    """Counts full passes through a model's diffusion network, after any step cache has had its say.
    Each entry records whether fp16 accumulation was on during that pass."""
    def watch(model):
        unet = model.model.diffusion_model
        original = unet._forward
        calls = []

        def counted(*args, **kwargs):
            calls.append(getattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", None))
            return original(*args, **kwargs)

        monkeypatch.setattr(unet, "_forward", counted)
        return calls
    return watch
