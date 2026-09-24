"""Decodes a compressed weight in one pass on the GPU with a Triton kernel.

The plain PyTorch decoder in codec.py runs a few dozen small operations per chunk,
each a trip through GPU memory. This kernel reads the compressed bytes once and
writes each value once. To find where the escapes of each block of values continue
in the next level's stream without a pass over the whole tensor, it uses a small
table of per-block starting positions, computed when the weight is compressed and
stored after it ("tables").

The first run of each kernel variant on a device is checked bit for bit against
the PyTorch decoder, and any failure turns the kernel off for the session, so a
problem with Triton costs speed, never correctness.
"""
import logging
import os

import torch

from . import codec

try:
    import triton
    import triton.language as tl
except ImportError:  # the PyTorch decoder is used instead
    triton = None

BLOCK = 4096      # values per kernel program and per table entry
LUT_SIZE = 128    # code tables are padded to this; codes have at most 7 bits
ALIGN = 16

_state = {"broken": False, "logged": False}
_verified = set()


def installed():
    return triton is not None


def usable(device):
    if triton is None or _state["broken"]:
        return False
    return device.type == "cuda" or (device.type == "cpu" and os.environ.get("TRITON_INTERPRET") == "1")


def tables(blob, info):
    """Code tables and per-block escape positions for a codec blob, as bytes to store after it:
    three LUT_SIZE-byte code tables, then int32 [levels, blocks] where row s says where each block's escapes
    start in the stream after level s + 1."""
    levels, counts = info["levels"], info["counts"]
    n, device = counts[0], blob.device
    blocks = -(-n // BLOCK)
    sections = [blob[offset:offset + size] for offset, size in info["sections"]]

    luts = torch.zeros(3, LUT_SIZE, dtype=torch.uint8)
    for i, level in enumerate(levels):
        luts[i, :len(level["lut"])] = torch.tensor(level["lut"], dtype=torch.uint8)

    # Level-1 escapes per block. The level-1 codes are one bit stream; read it a few blocks at a time.
    bits = levels[0]["bits"]
    per_block = torch.empty(blocks, dtype=torch.int64, device=device)
    piece = BLOCK * 1024
    for begin in range(0, n, piece):
        size = min(piece, n - begin)
        codes = codec.unpack_bits(sections[0][begin // 8 * bits:(begin + size + 7) // 8 * bits], bits, size)
        escaped = torch.zeros(-(-size // BLOCK) * BLOCK, dtype=torch.uint8, device=device)
        escaped[:size] = codes == (1 << bits) - 1
        per_block[begin // BLOCK:(begin + size + BLOCK - 1) // BLOCK] = escaped.view(-1, BLOCK).sum(1)
    starts = [per_block.cumsum(0) - per_block]

    # Deeper levels: count escapes along each stream and look the counts up at the previous level's starts.
    for s in range(1, len(levels)):
        codes = codec.unpack_bits(sections[s], levels[s]["bits"], counts[s])
        before = torch.zeros(counts[s] + 1, dtype=torch.int64, device=device)
        torch.cumsum(codes == (1 << levels[s]["bits"]) - 1, 0, out=before[1:])
        starts.append(before[starts[-1]])
    return torch.cat([luts.view(-1).to(device), torch.stack(starts).to(torch.int32).view(-1).view(torch.uint8)])


def attach(blob, info):
    """(blob with its tables after it, info that points at them)."""
    table = tables(blob, info)
    padding = -blob.numel() % ALIGN
    parts = [blob, torch.zeros(padding, dtype=torch.uint8, device=blob.device), table]
    return torch.cat(parts), dict(info, tables=blob.numel() + padding)


def decode(blob, info):
    """The decoded tensor, or None when the kernel can't be used (then use codec.decode)."""
    launch = info.get("launch") or info.setdefault("launch", _launch_settings(info))
    table, luts, rows = info["tables"], 3 * LUT_SIZE, len(info["levels"])
    out = torch.empty(launch["n"], dtype=launch["view"], device=blob.device)
    try:
        _decode_kernel[(launch["blocks"],)](
            blob, blob[table:table + luts], blob[table + luts:table + luts + 4 * rows * launch["blocks"]].view(torch.int32),
            out, *launch["args"], **launch["constants"], num_warps=8)
    except Exception as error:  # a missing compiler, an unsupported GPU, ...
        return _give_up(f"the Triton decoder failed ({type(error).__name__}: {error})")

    variant = (info["dtype"], launch["constants"]["BITS1"], rows, blob.device)
    if variant not in _verified:  # first use of this kernel variant here: compare with the PyTorch decoder
        if not torch.equal(out, codec.decode(blob, info).view(launch["view"]).view(-1)):
            return _give_up(f"the Triton decoder gave wrong results for {info['dtype']} weights")
        _verified.add(variant)
        if not _state["logged"]:
            _state["logged"] = True
            logging.info("Lossless: decoding compressed weights with Triton.")
    return out.view(launch["dtype"]).view(info["shape"])


def _launch_settings(info):
    dtype = codec.DTYPE_NAMES[info["dtype"]]
    view, e, m = codec.FORMATS[dtype]
    levels, n = info["levels"], info["counts"][0]
    blocks = -(-n // BLOCK)
    sections = [offset for offset, _ in info["sections"]]
    planes, extra = divmod(1 + m, 8)
    rest = sections[len(levels) + 1:]
    bits = [level["bits"] for level in levels] + [1, 1]
    args = (n, blocks, sections[0], sections[1] if len(levels) >= 2 else 0, sections[2] if len(levels) >= 3 else 0,
            sections[len(levels)], *(rest[j] if j < planes else 0 for j in range(3)), rest[planes] if extra else 0,
            bits[1], bits[2])
    constants = dict(BITS1=bits[0], CROSS1=8 % bits[0] != 0, LEVELS=len(levels), E=e, CROSS_RAW=8 % e != 0, M=m,
                     PLANES=planes, EXTRA=extra, CROSS_EXTRA=extra > 0 and 8 % extra != 0, LUT=LUT_SIZE, BLOCK=BLOCK)
    return {"n": n, "blocks": blocks, "dtype": dtype, "view": view, "args": args, "constants": constants}


def _give_up(reason):
    _state["broken"] = True
    logging.warning(f"Lossless: {reason}; using the slower PyTorch decoder from now on.")
    return None


if triton is not None:
    @triton.jit
    def _field(base, index, width, mask, CROSS: tl.constexpr):
        """Value `index` of a stream of `width`-bit values packed lowest bits first."""
        bit = index * width
        value = tl.load(base + (bit >> 3), mask=mask, other=0).to(tl.int32)
        if CROSS:  # the value can continue into the next byte
            value = value | (tl.load(base + (bit >> 3) + 1, mask=mask, other=0).to(tl.int32) << 8)
        return (value >> (bit & 7).to(tl.int32)) & ((1 << width) - 1)

    @triton.jit(do_not_specialize=["n", "blocks", "level1", "level2", "level3", "raw", "plane0", "plane1", "plane2",
                                   "extra", "bits2", "bits3"])
    def _decode_kernel(blob, luts, starts, out, n, blocks, level1, level2, level3, raw, plane0, plane1, plane2, extra,
                       bits2, bits3, BITS1: tl.constexpr, CROSS1: tl.constexpr, LEVELS: tl.constexpr, E: tl.constexpr,
                       CROSS_RAW: tl.constexpr, M: tl.constexpr, PLANES: tl.constexpr, EXTRA: tl.constexpr,
                       CROSS_EXTRA: tl.constexpr, LUT: tl.constexpr, BLOCK: tl.constexpr):
        block = tl.program_id(0)
        index = block.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = index < n

        # Exponent: level-1 code through its table; escapes continue in the next stream, in order.
        code = _field(blob + level1, index, BITS1, valid, CROSS1)
        exponent = tl.load(luts + code, mask=valid, other=0).to(tl.int32)
        escaped = valid & (code == (1 << BITS1) - 1)
        position = tl.load(starts + block).to(tl.int64) + tl.cumsum(escaped.to(tl.int32), 0) - 1
        if LEVELS >= 2:
            code = _field(blob + level2, position, bits2, escaped, True)
            exponent = tl.where(escaped, tl.load(luts + LUT + code, mask=escaped, other=0).to(tl.int32), exponent)
            escaped = escaped & (code == (1 << bits2) - 1)
            position = tl.load(starts + blocks + block).to(tl.int64) + tl.cumsum(escaped.to(tl.int32), 0) - 1
        if LEVELS >= 3:
            code = _field(blob + level3, position, bits3, escaped, True)
            exponent = tl.where(escaped, tl.load(luts + 2 * LUT + code, mask=escaped, other=0).to(tl.int32), exponent)
            escaped = escaped & (code == (1 << bits3) - 1)
            position = tl.load(starts + 2 * blocks + block).to(tl.int64) + tl.cumsum(escaped.to(tl.int32), 0) - 1
        exponent = tl.where(escaped, _field(blob + raw, position, E, escaped, CROSS_RAW), exponent)

        # Sign and mantissa: whole byte planes, then the leftover bits.
        rest = tl.zeros([BLOCK], dtype=tl.int32)
        if PLANES >= 1:
            rest = rest | tl.load(blob + plane0 + index, mask=valid, other=0).to(tl.int32)
        if PLANES >= 2:
            rest = rest | (tl.load(blob + plane1 + index, mask=valid, other=0).to(tl.int32) << 8)
        if PLANES >= 3:
            rest = rest | (tl.load(blob + plane2 + index, mask=valid, other=0).to(tl.int32) << 16)
        if EXTRA > 0:
            rest = rest | (_field(blob + extra, index, EXTRA, valid, CROSS_EXTRA) << (8 * PLANES))

        value = (((rest >> M) & 1) << (E + M)) | (exponent << M) | (rest & ((1 << M) - 1))
        tl.store(out + index, value.to(out.dtype.element_ty), mask=valid)
