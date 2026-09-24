# ComfyUI VRAM & Speed Optimizer

Custom nodes that make image and video generation faster and use less VRAM.
They work by switching on optimizations ComfyUI already supports internally but
doesn't expose in one place. The model is only patched through ComfyUI's own
hooks, so its memory manager, LoRAs and other model patches keep working.

| Node | What it does |
| --- | --- |
| **Speed Up Model** | Faster attention (SageAttention / FlashAttention), fp16 accumulation, step caching and `torch.compile`, from one node. |
| **Load Checkpoint (FP8, Low VRAM)** | Loads a checkpoint with the diffusion model stored in fp8, halving the VRAM its weights need. |
| **Compress Model (FP8)** | Converts any loaded model's weights to fp8, halving their VRAM, and keeps LoRAs and patches. |
| **Compact VRAM Before Load** | Right before a model loads onto the GPU, unloads the other models and clears cached memory so it gets the most free VRAM. |
| **Free VRAM** | Passthrough node that unloads models and clears the GPU cache between stages of a workflow. |
| **Compress Model File (Lossless)** | Writes a smaller copy of a model file whose weights decode to exactly the original bits. |
| **Load Diffusion Model / Checkpoint / CLIP (Lossless)** | Load those files; the diffusion model and checkpoint loaders can keep the weights compressed in RAM and VRAM. |
| **Load LoRA / VAE (Lossless)** | Apply a compressed LoRA or load a compressed VAE, exactly like the regular nodes. |

They are under the **optimization** category in the node menu (the lossless ones under **optimization/lossless**).

## Installation

1. Clone this repository into `ComfyUI/custom_nodes/`:
   ```
   cd ComfyUI/custom_nodes
   git clone https://github.com/GDI-Forces/ComfyUINodeTest
   ```
2. Restart ComfyUI.

Requires a recent ComfyUI (one with the `comfy_api.latest` node API and the EasyCache node).
There are no extra Python dependencies. For the biggest speed-up, also install
[SageAttention](https://github.com/thu-ml/SageAttention) into the Python environment ComfyUI runs in.
Pick the build that matches your GPU, PyTorch and OS as described on its page.

## Usage

```
Any loader ──MODEL──> [LoRAs] ──> Compress Model ──> Compact VRAM ──> Speed Up Model ──> KSampler ──LATENT──> Free VRAM ──> VAE Decode
                                   (smaller)        (clear space)       (faster)
```

Use only the nodes you need. Compress Model must come before Speed Up Model when `torch_compile` is on.

**Speed Up Model** takes any MODEL, so it works after `Load Diffusion Model` and with video
models such as Wan, Hunyuan and LTX. Workflows with two models (for example Wan 2.2's
high-noise and low-noise models) need one Speed Up Model node per model.

### Speed Up Model

| Option | Effect | Cost |
| --- | --- | --- |
| `attention` | `auto` uses SageAttention if installed, then FlashAttention, otherwise leaves attention alone. Attention is most of the compute in video models, so this is usually the biggest single gain. | SageAttention quantizes attention internally. The difference is rarely visible. |
| `fp16_accumulation` | Faster matrix multiplies on NVIDIA RTX and AMD GPUs with PyTorch 2.7+. Only applies while this model samples. | Only helps models that compute in fp16. The log says when it can't help. Slight change in results. |
| `step_cache` | ComfyUI's EasyCache: reuses the model's output on steps where it barely changes, skipping full model passes. `0` is off, `0.1`–`0.25` is a good range. | Higher values skip more steps and lose detail. Changes results. |
| `torch_compile` | Compiles the model with `torch.compile` for faster steps. | Needs Triton (`triton-windows` on Windows). The first run takes minutes and recompiles when the resolution or frame count changes. |

The node logs what it applied, e.g. `Speed Up Model: sage attention, step cache 0.2`.
Its step cache always uses EasyCache's default 15%–95% window. For finer control, use the
built-in **EasyCache** node instead and leave `step_cache` at 0.

### Load Checkpoint (FP8, Low VRAM)

This is a drop-in replacement for `Load Checkpoint`. It stores the diffusion model's weights in fp8 and
converts each layer back up as it runs. The weights take half the VRAM of fp16/bf16, so a
model that used to spill into system RAM can fit on the GPU.
`fp8_e4m3fn` suits most models. `default` behaves like the regular loader, which is handy for A/B comparisons.
For standalone diffusion model files, the built-in `Load Diffusion Model` node already has a
`weight_dtype` option that does the same.

### Compress Model (FP8)

Takes a model from any core loader (`Load Diffusion Model`, `Load Checkpoint`, including after LoRAs) and outputs
a copy whose weights are stored in fp8. Each layer is converted back up as it runs. The weights need half the VRAM
of fp16/bf16: roughly 14 GB instead of 28 GB for Wan 14B, 12 GB instead of 24 GB for Flux, 2.5 GB instead of 5 GB
for an SDXL UNet. The console confirms it, e.g. `Compress Model: torch.bfloat16 -> torch.float8_e4m3fn weights`.

- It rebuilds the weights from the model file through the model's own loader, exactly like the loaders'
  `weight_dtype` fp8 option, then moves the LoRAs, patches and options already on the model onto them.
  The input model is left unchanged.
- Models already stored in fp8 or loaded from a pre-quantized file pass through unchanged. Models from
  custom loaders that don't record how they were loaded (such as GGUF loaders, whose models are already
  compressed) stop the workflow with an error explaining this.
- fp8 is lossy. Results change a little, usually not visibly. `fp8_e4m3fn` suits most models.
- If your loader already has a `weight_dtype` option, setting fp8 there saves the same VRAM and a bit of loading time.

### Compact VRAM Before Load

Put it between the model loader (or LoRAs) and the sampler. When the sampler is about to load the model
onto the GPU, the node unloads every other model (text encoders, VAEs, other diffusion models), runs Python's
garbage collector and hands PyTorch's cached memory back to the driver. The model then loads into the largest
free block of VRAM, instead of being partly offloaded or running out of memory because leftovers were in the way.
The console shows the effect, e.g. `Compact VRAM: 3.10 GB -> 9.80 GB free on cuda:0 before loading the model`.

It runs every time the sampler loads the model, so models it unloaded (such as the text encoder) reload when next
used. Leave it out when everything already fits. Turn off `unload_other_models` to only clear the cached memory.
In workflows with two diffusion models (e.g. Wan 2.2), a Compact node on each one makes them take turns on the GPU.

### Free VRAM

Put it on any connection (it accepts and returns any type) where the next stage needs a lot
of memory. A common spot is between the sampler and **VAE Decode** in video workflows.
It unloads all models and empties the GPU cache, and logs free VRAM before and after. Unloaded models reload
when next used, so leave it out of workflows where everything already fits.

## Lossless model compression

The fp8 nodes save more memory but change the weights slightly. The lossless tools keep every weight
**bit for bit identical**, in whatever dtype it already has (fp32, fp16, bf16, fp8), so results are exactly
the same as with the original file.

How much it saves, measured on 25M real trained parameters (every tensor verified bit-exact):

| Stored as | Saving | Theoretical limit |
| --- | --- | --- |
| bf16 | **31%** | 32% |
| fp8 e5m2 / e4m3fn | 25% / 22% | 27% / 24% |
| fp32 | 16% | 16% |
| fp16 | 13% | 13% |

For example, a 28 GB bf16 model becomes about 19 GB. Lossless compression can't go much further: in trained
weights the sign and mantissa bits are close to random, and only the exponent bits are predictable. The tools
store sign and mantissa unchanged and give each tensor's exponents short codes, which gets within 1–2 points
of the theoretical limit.

### In ComfyUI

1. Add **Compress Model File (Lossless)**, pick a file from `diffusion_models`, `checkpoints`,
   `text_encoders`, `loras` or `vae`, and queue it. It writes `<name>.lossless.safetensors` next to the original, checks that
   every tensor decodes to exactly the original bits, and shows the saving. The original is never touched;
   delete it yourself once you're happy.
2. Press **R** to refresh, then use the new file with the matching lossless node in place of the regular one:

   | File in | Regular node | Lossless node |
   | --- | --- | --- |
   | `diffusion_models` | Load Diffusion Model | **Load Diffusion Model (Lossless)** |
   | `checkpoints` | Load Checkpoint | **Load Checkpoint (Lossless)** |
   | `text_encoders` | Load CLIP | **Load CLIP (Lossless)** |
   | `loras` | Load LoRA / LoraLoaderModelOnly | **Load LoRA (Lossless)**; leave `clip` unconnected to change only the model |
   | `vae` | Load VAE | **Load VAE (Lossless)** |

   The results are identical to the regular nodes with the original file.

The diffusion model and checkpoint loaders have a `keep_compressed` option:

- **Off (default):** the weights are decoded once while loading. Generation is exactly as fast as with the
  original file; only disk space (and download size) is saved. Decoding runs on the GPU when there is one.
- **On (experimental):** the weights stay compressed in RAM and VRAM, and each layer is decoded when it runs,
  so the model takes about as much memory as the file (for bf16, ~31% less VRAM and RAM). Every step gets
  slower because of the decoding; the cost is smaller for big video models, where each step does a lot of
  work per weight, than for small image models. LoRAs work and stay compressed. Models loaded this way use
  ComfyUI's classic memory manager rather than DynamicVRAM, can't also go through Compress Model (FP8), and
  haven't been tried with `torch.compile`. Files that were already quantized (fp8 "scaled" or other ComfyUI
  quantized formats) are loaded decoded.

### Command line

Run it with the Python that runs ComfyUI. It works for `.safetensors`, `.ckpt`, `.pt` and `.pth` files,
including VAEs and LoRAs, and uses the GPU when there is one:

```
python custom_nodes/ComfyUINodeTest/lossless_compress.py compress models/diffusion_models/model.safetensors
python custom_nodes/ComfyUINodeTest/lossless_compress.py info model.lossless.safetensors
python custom_nodes/ComfyUINodeTest/lossless_compress.py verify model.safetensors model.lossless.safetensors
python custom_nodes/ComfyUINodeTest/lossless_compress.py decompress model.lossless.safetensors -o model.safetensors
```

`compress` verifies its output unless you pass `--no-verify`. `decompress` gives back a regular `.safetensors`
file with the original tensors and metadata, so nothing is ever locked into this format.

### Limits

- LoRAs, VAEs and text encoders are always decoded when loaded: they only save disk space. They are usually
  small next to the diffusion model, and LoRA weights are folded into the model's weights anyway.
- With `keep_compressed` off, a decoded model sits in RAM like a model loaded from a `.ckpt` file. ComfyUI can't
  page it back to disk the way it does with memory-mapped `.safetensors` files, so on a machine with little RAM
  the original file may load more comfortably.
- Speeds were only measured on a 4-thread CPU here (about 65 MB/s to compress and 180 MB/s to decode bf16).
  GPU speed and the per-step cost of `keep_compressed` haven't been measured yet.

## Measuring the speed-up on your GPU

`benchmark.py` runs one of your own workflows with each optimization on and off and reports time and peak VRAM.
With ComfyUI running, export your workflow with **Workflow → Export (API)**, then run:

```
python custom_nodes/ComfyUINodeTest/benchmark.py workflow_api.json
```

Use `--url` if ComfyUI isn't on `http://127.0.0.1:8188` (the desktop app uses port 8000), and `--configs` to choose
what to compare (`compile` and `fp16_accumulation` are opt-in). Results are saved to `benchmark_results.md`,
and the images or videos land in `output/benchmark/` for side-by-side comparison.

## What this pack deliberately doesn't do

- **No "block swap".** ComfyUI already streams weights between RAM and VRAM when a model doesn't
  fit. Nodes that move blocks by hand fight that system. ComfyUI itself turns the most common
  one (`wanBlockSwap`) into a no-op for that reason.
- **No miracles on image size.** For long or high-resolution videos, also use **VAE Decode (Tiled)**,
  and consider fp8 or GGUF versions of the model files.

## Development

The tests run the nodes against a real ComfyUI checkout. They use a tiny SD1.5-shaped model with random
weights, so no downloads are needed. Tests marked for CUDA are skipped on machines without an NVIDIA GPU:

```
pip install pytest   # in an environment with ComfyUI's requirements installed
COMFYUI_PATH=/path/to/ComfyUI python -m pytest tests
```

On that test model, fp8 loading cut the diffusion model from 139.8 MiB (fp32) to 35.0 MiB, and
`step_cache` 0.1–0.5 skipped 5–6 of 12 model passes. Real models and GPUs will differ.

To publish the pack to the ComfyUI Registry, generate a `pyproject.toml` with `comfy node init`
(from `comfy-cli`) and follow the ComfyUI Registry's publishing guide.
