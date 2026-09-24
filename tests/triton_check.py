"""Checks the Triton decoder bit for bit against the original tensors, for every format.

Runs on the GPU when there is one; otherwise run it with TRITON_INTERPRET=1 to use
Triton's CPU interpreter. Exits non-zero on any difference. Used by test_lossless.py.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lossless import codec, kernels  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
assert kernels.usable(torch.device(device)), "Triton can't run here (on the CPU, set TRITON_INTERPRET=1)"


def bit_patterns(dtype):
    view = codec.FORMATS[dtype][0]
    if view == torch.uint8:
        return torch.arange(256, dtype=torch.int32).to(torch.uint8).view(dtype)
    if view == torch.int16:
        return torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(dtype)
    specials = torch.tensor([0, -2**31, 2**31 - 1, 0x7F800000, -8388608, 0x7FC00001, 1], dtype=torch.int32)
    return torch.cat([torch.randint(-2**31, 2**31 - 1, (20000,), dtype=torch.int32), specials]).view(dtype)


def weights(shape, dtype):
    values = torch.randn(shape)
    if dtype == torch.int8:
        return (values * 30).round().clamp(-127, 127).to(torch.int8)
    return (values * (50 if dtype == torch.float8_e4m3fn else 0.02)).to(dtype)


def cases(dtype):
    patterns = bit_patterns(dtype)
    mixed = torch.cat([weights(10 * patterns.numel(), dtype), patterns])
    yield "every bit pattern", mixed[torch.randperm(mixed.numel())]
    yield "weights", weights((700, 333), dtype)
    yield "shorter than a block", weights(4100, dtype)
    # seven common exponents and some rare ones, with one 3-bit code level whose escapes go straight to raw
    exponents = torch.randint(0, 7, (20000,)).float()
    exponents[::500] = -3
    one_level = exponents.exp2() * (1 + 0.9 * torch.rand(20000)) * torch.randn(20000).sign()
    yield "one level", one_level.round().clamp(-127, 127).to(dtype) if dtype == torch.int8 else one_level.to(dtype)


def plan_one_level(counts, e):
    return [3], 0  # (int8 tries several splits; each gets this plan and the first is kept)


torch.manual_seed(0)
failures = 0
for dtype in codec.FORMATS:
    view = codec.FORMATS[dtype][0]
    for name, tensor in cases(dtype):
        plan_levels = codec.plan_levels
        if name == "one level":
            codec.plan_levels = plan_one_level
        try:
            encoded = codec.encode(tensor.to(device), chunk=1 << 12)
        finally:
            codec.plan_levels = plan_levels
        if encoded is None:
            print(f"{str(dtype):22} {name:20} not compressible, skipped")
            continue
        blob, info = kernels.attach(*encoded)
        decoded = kernels.decode(blob, info)
        exact = decoded is not None and torch.equal(decoded.cpu().view(view), tensor.view(view))
        failures += not exact
        levels = [level["bits"] for level in info["levels"]]
        print(f"{str(dtype):22} {name:20} levels {str(levels):10} {'exact' if exact else 'WRONG'}")
sys.exit(1 if failures else 0)
