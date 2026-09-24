"""Measures how much the optimizer nodes speed up one of your own workflows on your GPU.

1. Start ComfyUI as usual and open the workflow you want to test.
2. Export it in API format (Workflow menu > Export (API)), e.g. as workflow_api.json.
3. Run:  python benchmark.py workflow_api.json

For every configuration it frees VRAM, does one warm-up run (loads the models, and compiles
with --configs compile), then times --runs more runs with new seeds. The same seeds are used
for every configuration, so the images or videos saved under output/benchmark/ can be
compared side by side. Results are printed and saved to benchmark_results.md.
"""
import argparse
import copy
import json
import sys
import time
import urllib.error
import urllib.request

SPEED_UP_DEFAULTS = {"attention": "keep", "fp16_accumulation": False, "step_cache": 0.0, "torch_compile": False}
STEP_CACHE = 0.2

CONFIGS = {
    "baseline": ({}, False),
    "attention": ({"attention": "auto"}, False),
    "step_cache": ({"step_cache": STEP_CACHE}, False),
    "fp16_accumulation": ({"fp16_accumulation": True}, False),
    "fp8": (None, True),
    "compile": ({"torch_compile": True}, False),
    "combined": ({"attention": "auto", "step_cache": STEP_CACHE}, True),
}
DEFAULT_CONFIGS = ["baseline", "attention", "step_cache", "fp8", "combined"]

# The local server must not go through any HTTP proxy configured in the environment.
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def api(url, path, data=None):
    body = None if data is None else json.dumps(data).encode()
    request = urllib.request.Request(url + path, data=body, headers={"Content-Type": "application/json"})
    with opener.open(request) as response:
        raw = response.read()
    return json.loads(raw) if raw else None


def is_link(value):
    return isinstance(value, list) and len(value) == 2 and isinstance(value[1], int)


def add_speed_up(prompt, options):
    """Routes every sampler's (or guider's) model through one Speed Up Model node per model source."""
    next_id = max((int(k) for k in prompt if k.isdigit()), default=0) + 1000
    inserted = {}
    for node in list(prompt.values()):
        name = node["class_type"].lower()
        model = node.get("inputs", {}).get("model")
        if not ("sampler" in name or "guider" in name) or not is_link(model):
            continue
        source = tuple(model)
        if source not in inserted:
            inserted[source] = str(next_id)
            prompt[str(next_id)] = {"class_type": "OptimizerSpeedUpModel",
                                    "inputs": {"model": list(source), **SPEED_UP_DEFAULTS, **options}}
            next_id += 1
        node["inputs"]["model"] = [inserted[source], 0]
    if not inserted:
        raise ValueError("no sampler or guider node with a model input found")


def use_fp8_weights(prompt):
    changed = False
    for node in prompt.values():
        if node["class_type"] == "CheckpointLoaderSimple":
            node["class_type"] = "OptimizerLoadCheckpointFP8"
            node["inputs"] = {"ckpt_name": node["inputs"]["ckpt_name"], "weight_dtype": "fp8_e4m3fn"}
            changed = True
        elif node["class_type"] == "UNETLoader" and node["inputs"].get("weight_dtype") == "default":
            node["inputs"]["weight_dtype"] = "fp8_e4m3fn"
            changed = True
    if not changed:
        raise ValueError("no Load Checkpoint or Load Diffusion Model node with default weights to switch to fp8")


def build(prompt, config):
    options, fp8 = CONFIGS[config]
    prompt = copy.deepcopy(prompt)
    if options:
        add_speed_up(prompt, options)
    if fp8:
        use_fp8_weights(prompt)
    for node in prompt.values():
        if isinstance(node.get("inputs", {}).get("filename_prefix"), str):
            node["inputs"]["filename_prefix"] = f"benchmark/{config}"
    return prompt


def with_seed_offset(prompt, offset):
    prompt = copy.deepcopy(prompt)
    for node in prompt.values():
        for key in ("seed", "noise_seed"):
            if isinstance(node.get("inputs", {}).get(key), int):
                node["inputs"][key] = (node["inputs"][key] + offset) % 2 ** 64
    return prompt


def has_seed(prompt):
    return any(isinstance(node.get("inputs", {}).get(key), int)
               for node in prompt.values() for key in ("seed", "noise_seed"))


def memory_free(url):
    device = api(url, "/system_stats")["devices"][0]
    return device["vram_free"], device["vram_total"]


def run(url, prompt):
    """Queues a prompt and waits for it. Returns (seconds, lowest free memory seen, error or None)."""
    try:
        prompt_id = api(url, "/prompt", {"prompt": prompt})["prompt_id"]
    except urllib.error.HTTPError as e:
        return None, None, f"rejected by ComfyUI: {e.read().decode()[:500]}"
    lowest_free = None
    while True:
        free, _ = memory_free(url)
        lowest_free = free if lowest_free is None else min(lowest_free, free)
        history = api(url, f"/history/{prompt_id}").get(prompt_id)
        if history and history["status"].get("completed") is not None:
            break
        time.sleep(0.1)
    times, error = {}, None
    for kind, data in history["status"]["messages"]:
        times[kind] = data.get("timestamp")
        if kind == "execution_error":
            error = f"{data.get('node_type')}: {data.get('exception_message', '').strip()}"
    if error or history["status"]["status_str"] != "success":
        return None, lowest_free, error or history["status"]["status_str"]
    end = times.get("execution_success")
    return (end - times["execution_start"]) / 1000, lowest_free, None


def benchmark(url, prompt, config, runs):
    try:
        prompt = build(prompt, config)
    except ValueError as e:
        return {"config": config, "status": f"skipped: {e}"}
    api(url, "/free", {"unload_models": True, "free_memory": True})
    time.sleep(2)
    print(f"[{config}] warm-up run...", flush=True)
    first, _, error = run(url, with_seed_offset(prompt, 0))
    if error:
        return {"config": config, "status": f"failed: {error}"}
    timings, lowest_free = [], None
    for i in range(1, runs + 1):
        print(f"[{config}] timed run {i}/{runs}...", flush=True)
        seconds, free, error = run(url, with_seed_offset(prompt, i))
        if error:
            return {"config": config, "status": f"failed: {error}"}
        timings.append(seconds)
        lowest_free = free if lowest_free is None else min(lowest_free, free)
    _, total = memory_free(url)
    return {"config": config, "status": "ok", "first_run": first, "average": sum(timings) / len(timings),
            "peak_memory_gb": (total - lowest_free) / 1024 ** 3}


def report(system, results):
    info = system["system"]
    device = system["devices"][0]
    lines = [
        "# Optimizer benchmark",
        "",
        f"- Device: {device['name']} ({device['vram_total'] / 1024 ** 3:.1f} GB)",
        f"- ComfyUI {info.get('comfyui_version', '?')}, PyTorch {info.get('pytorch_version', '?')}, Python {info.get('python_version', '?').split()[0]}",
        "",
        "| Config | Status | First run (s) | Avg run (s) | Speed-up | Peak memory (GB) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    baseline = next((r["average"] for r in results if r["config"] == "baseline" and r["status"] == "ok"), None)
    for r in results:
        if r["status"] != "ok":
            lines.append(f"| {r['config']} | {r['status']} | | | | |")
            continue
        speed_up = f"{baseline / r['average']:.2f}x" if baseline else ""
        lines.append(f"| {r['config']} | ok | {r['first_run']:.1f} | {r['average']:.1f} | {speed_up} | {r['peak_memory_gb']:.2f} |")
    lines += [
        "",
        "Peak memory is sampled every 0.1 s from ComfyUI's free-memory report, so short spikes can be missed.",
        "The ComfyUI console shows what each Speed Up Model node applied (lines starting with `Speed Up Model:`).",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workflow", help="workflow exported with Workflow > Export (API)")
    parser.add_argument("--url", default="http://127.0.0.1:8188", help="address of the running ComfyUI (default: %(default)s)")
    parser.add_argument("--runs", type=int, default=2, help="timed runs per configuration (default: %(default)s)")
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS),
                        help=f"comma-separated, from: {', '.join(CONFIGS)} (default: %(default)s)")
    parser.add_argument("--output", default="benchmark_results.md", help="where to save the results (default: %(default)s)")
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        sys.exit(f"unknown config(s): {', '.join(unknown)}")
    with open(args.workflow, encoding="utf-8") as f:
        prompt = json.load(f)
    if "nodes" in prompt and "links" in prompt:
        sys.exit("This is a regular workflow file. Export it with Workflow > Export (API) instead.")
    if not has_seed(prompt):
        sys.exit("No seed or noise_seed value found in the workflow, so repeated runs would just be cached.")
    try:
        system = api(args.url, "/system_stats")
    except urllib.error.URLError as e:
        sys.exit(f"Can't reach ComfyUI at {args.url} ({e.reason}). Start it first, or pass --url.")

    results = [benchmark(args.url, prompt, config, args.runs) for config in configs]
    text = report(system, results)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + text + f"\n\nSaved to {args.output}")


if __name__ == "__main__":
    main()
