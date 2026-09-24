"""Bit-exact compression of floating point and int8 tensors.

A float is sign, exponent and mantissa bits. In trained weights the sign and
mantissa bits are close to random, but only a handful of exponent values occur,
so the exponent is where the savings are:

- The sign and mantissa bits ("rest") are stored unchanged, packed tightly.
- The exponent goes through a cascade of short fixed-width codes. Level 1 gives
  the most common exponents a b1-bit code, and its all-ones code means "look in
  the next level", which codes the rarer exponents the same way, down to a last
  level that stores them raw. Each level is a dense stream in element order, so
  decoding needs no positions: the escapes are filled back in order.

int8 weights (ComfyUI's int8 and int8 convrot models) go through the same code:
their low 7 bits are first XORed with the sign bit, which leaves the sign on top
and the magnitude below it, so the top magnitude bits play the exponent's part.
Which bits count as "exponent" is picked per tensor.

Everything is plain PyTorch, so decoding runs on the CPU or the GPU. Level-1
codes are processed in fixed-size chunks to bound temporary memory, and the
escape count of every chunk is recorded so decoding never waits on the GPU.
Decoding works on bytes: the output is assembled byte by byte in place, and
besides the escape positions and table lookups (done in small pieces), each
temporary takes one byte per value.
"""
import logging
import sys

import torch

assert sys.byteorder == "little", "the decoder assembles values from their bytes"

# dtype: (same-width integer view, exponent bits, mantissa bits)
FORMATS = {
    torch.float32: (torch.int32, 8, 23),
    torch.bfloat16: (torch.int16, 8, 7),
    torch.float16: (torch.int16, 5, 10),
    torch.float8_e4m3fn: (torch.uint8, 4, 3),
    torch.float8_e5m2: (torch.uint8, 5, 2),
    torch.int8: (torch.uint8, 4, 3),  # "exponent" = top magnitude bits; see SPLITS
}
# Formats whose split into exponent and mantissa is picked per tensor, best first when tied.
SPLITS = {torch.int8: [(4, 3), (5, 2), (6, 1), (3, 4)]}
FOLDED = {torch.int8}  # stored with the low 7 bits XORed with the sign bit


def layout(info):
    """(integer view, exponent bits, mantissa bits) of an encoded tensor."""
    view, e, m = FORMATS[DTYPE_NAMES[info["dtype"]]]
    return (view, *info["split"]) if "split" in info else (view, e, m)
DTYPE_NAMES = {str(dtype).removeprefix("torch."): dtype for dtype in FORMATS}

CHUNK = 1 << 24  # level-1 elements decoded at a time; a multiple of 8
MAX_LEVELS = 3   # coded levels before the raw one
MIN_SAVING = 0.01


def pack_bits(values, width):
    """Packs uint8 values below 2**width, eight to `width` bytes, lowest bits first."""
    if width == 8:
        return values
    values = torch.nn.functional.pad(values, (0, -values.numel() % 8))
    if 8 % width == 0:  # whole values per byte: stay in uint8
        shifts = torch.arange(0, 8, width, device=values.device, dtype=torch.uint8)
        return (values.view(-1, 8 // width) << shifts).sum(1, dtype=torch.uint8)
    wide = torch.int32 if width <= 3 else torch.int64
    shifts = torch.arange(8, device=values.device, dtype=wide) * width
    acc = (values.view(-1, 8).to(wide) << shifts).sum(1, dtype=wide)
    return torch.stack([(acc >> (8 * j)) & 0xFF for j in range(width)], 1).to(torch.uint8).flatten()


_CACHE = {}


def cached(key, make):
    """Small constant tensors (lookup tables, shift amounts), made once per device: creating them from Python
    lists every call would copy them to the GPU and wait for it each time."""
    value = _CACHE.get(key)
    if value is None:
        value = _CACHE[key] = make()
    return value


def unpack_bits(packed, width, count):
    """Inverse of pack_bits, as uint8. Only ever makes one-byte-per-value temporaries."""
    if width == 8:
        return packed[:count]
    device, mask = packed.device, (1 << width) - 1
    if 8 % width == 0:
        per_byte = 8 // width
        shifts = cached(("shifts", width, device), lambda: torch.arange(0, 8, width, device=device, dtype=torch.uint8))
        values = packed[:-(-count // per_byte)].unsqueeze(1) >> shifts
        values &= mask
        return values.view(-1)[:count]
    # Eight values share `width` bytes; value j starts at bit j * width and may straddle two bytes.
    groups = -(-count // 8)
    grouped = packed[:groups * width].view(groups, width)
    values = torch.empty(groups, 8, dtype=torch.uint8, device=packed.device)
    spill = torch.empty(groups, dtype=torch.uint8, device=packed.device)
    for j in range(8):
        byte, shift = divmod(j * width, 8)
        column = values[:, j]
        torch.bitwise_right_shift(grouped[:, byte], shift, out=column)
        if shift + width > 8:
            torch.bitwise_left_shift(grouped[:, byte + 1], 8 - shift, out=spill)
            column |= spill
        column &= mask
    return values.view(-1)[:count]


def split_bits(flat, dtype, e=None, m=None):
    """(exponent, rest) of every element of a flat chunk as int32; rest is the sign above the mantissa."""
    view, e0, m0 = FORMATS[dtype]
    e, m = (e0, m0) if e is None else (e, m)
    bits = flat.view(view).to(torch.int32)
    if view != torch.int32:
        bits &= (1 << (1 + e + m)) - 1  # undo sign extension of int16
    if dtype in FOLDED:
        bits ^= (bits >> 7) * 0x7F
    exponent = (bits >> m) & ((1 << e) - 1)
    rest = (((bits >> (e + m)) & 1) << m) | (bits & ((1 << m) - 1))
    return exponent, rest


def plan_levels(counts, e):
    """Cheapest cascade of code widths for exponent counts sorted most common first.
    Returns (widths of the coded levels, total bits); no coded levels means raw."""
    n = sum(counts)
    best = ([], n * e)

    def search(widths, start, reaching, bits):
        nonlocal best
        total = bits + reaching * e  # the rest stored raw
        if total < best[1]:
            best = (widths, total)
        if len(widths) == MAX_LEVELS or start >= len(counts):
            return
        for b in range(1, e):
            size = (1 << b) - 1
            search(widths + [b], start + size, reaching - sum(counts[start:start + size]), bits + reaching * b)

    search([], 0, n, 0)
    return best


OUT_OF_MEMORY = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)
HEADROOM = 256 << 20  # VRAM left free when deciding whether work fits on the GPU
ENCODE_MEMORY_PER_VALUE = 32  # temporaries of encode(), per value of a chunk


def has_room(device, needed):
    """Whether `device` has `needed` bytes free, counting memory PyTorch holds cached. Only checked for CUDA."""
    device = torch.device(device)
    if device.type != "cuda":
        return True
    free, _ = torch.cuda.mem_get_info(device)
    free += torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    return needed + HEADROOM <= free


def run_on(device, needed, work, fallback="cpu"):
    """work(device) when `device` is set and has room for `needed` bytes, else (or if it runs out of memory
    anyway) work(fallback). Loading a model while another one fills the GPU must not fail just because the
    GPU is the faster place to encode or decode."""
    if device is not None and has_room(device, needed):
        try:
            return work(device)
        except OUT_OF_MEMORY:
            torch.cuda.empty_cache()
            logging.warning(f"Lossless: not enough memory on {device}, using the {fallback} for this tensor.")
    return work(fallback)


def encode_memory(tensor, chunk=None):
    """About how much device memory encode() takes for a tensor, including the tensor itself."""
    return 2 * tensor.nbytes + ENCODE_MEMORY_PER_VALUE * min(tensor.numel(), chunk or CHUNK)


def decode_memory(blob, info):
    """How much device memory decode() takes for a blob, including the blob and the result."""
    return blob.nbytes + info["counts"][0] * DTYPE_NAMES[info["dtype"]].itemsize + temporary_memory(info)


def encode(tensor, chunk=None):
    """(blob, info) for a tensor, or None when compressing it wouldn't save at least 1%.
    Runs on the tensor's device and returns the blob there. `chunk` (a multiple of 8, CHUNK by default) bounds
    the decoder's temporary memory, which takes up to about eight bytes per value of a chunk."""
    chunk_size = chunk or CHUNK
    if tensor.dtype not in FORMATS or tensor.numel() == 0:
        return None
    device = tensor.device
    chunks = tensor.detach().contiguous().view(-1).split(chunk_size)
    n = tensor.numel()

    # Plan the exponent codes for each candidate split and keep the smallest.
    best = None
    for e, m in SPLITS.get(tensor.dtype, [FORMATS[tensor.dtype][1:]]):
        counts = sum(torch.bincount(split_bits(c, tensor.dtype, e, m)[0], minlength=1 << e) for c in chunks)
        order = counts.argsort(descending=True, stable=True)
        widths, exponent_bits = plan_levels(counts[order][:int((counts > 0).sum())].tolist(), e)
        if best is None or n * (1 + m) + exponent_bits < best[0]:
            best = (n * (1 + m) + exponent_bits, e, m, order, widths)
    total_bits, e, m, order, widths = best
    if not widths or total_bits > n * (1 + e + m) * (1 - MIN_SAVING):
        return None  # not worth encoding; checked again below on the real size, which includes padding
    rank = torch.empty(1 << e, dtype=torch.int32, device=device)
    rank[order] = torch.arange(1 << e, dtype=torch.int32, device=device)
    rest_planes, rest_extra = divmod(1 + m, 8)

    # Level 1 and the rest bits, a chunk at a time; escapes are few, so collect them whole.
    escape = (1 << widths[0]) - 1
    level1, planes, extra, chunk_escapes, escaped_ranks, escaped_values = [], [[] for _ in range(rest_planes)], [], [], [], []
    for chunk in chunks:
        exponent, rest = split_bits(chunk, tensor.dtype, e, m)
        ranks = rank[exponent]
        codes = ranks.clamp(max=escape)
        escaped = codes == escape
        level1.append(pack_bits(codes.to(torch.uint8), widths[0]))
        chunk_escapes.append(int(escaped.sum()))
        escaped_ranks.append(ranks[escaped])
        escaped_values.append(exponent[escaped])
        for j in range(rest_planes):
            planes[j].append(((rest >> (8 * j)) & 0xFF).to(torch.uint8))
        if rest_extra:
            extra.append(pack_bits((rest >> (8 * rest_planes)).to(torch.uint8), rest_extra))

    sections = [torch.cat(level1)]
    levels = [{"bits": widths[0], "lut": order[:escape].tolist()}]
    level_counts = [n]
    symbols, values, start = torch.cat(escaped_ranks), torch.cat(escaped_values), escape
    for b in widths[1:]:
        level_counts.append(values.numel())
        escape = (1 << b) - 1
        codes = (symbols - start).clamp(max=escape)
        escaped = codes == escape
        sections.append(pack_bits(codes.to(torch.uint8), b))
        levels.append({"bits": b, "lut": order[start:start + escape].tolist()})
        symbols, values, start = symbols[escaped], values[escaped], start + escape
    level_counts.append(values.numel())
    sections.append(pack_bits(values.to(torch.uint8), e))  # the raw level
    sections += [torch.cat(plane) for plane in planes]
    if rest_extra:
        sections.append(torch.cat(extra))

    offsets, offset = [], 0
    for section in sections:
        offsets.append([offset, section.numel()])
        offset += section.numel()
    info = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "levels": levels,
        "counts": level_counts,
        "chunk": chunk_size,
        "chunk_escapes": chunk_escapes,
        "sections": offsets,
    }
    if tensor.dtype in SPLITS:
        info["split"] = [e, m]
    blob = torch.cat(sections)
    if blob.numel() > tensor.numel() * tensor.element_size() * (1 - MIN_SAVING):
        return None
    return blob, info


def decode(blob, info):
    """The exact original tensor, on the blob's device."""
    dtype = DTYPE_NAMES[info["dtype"]]
    view, e, m = layout(info)
    levels, counts, chunk = info["levels"], info["counts"], info["chunk"]
    sections = [blob[offset:offset + size] for offset, size in info["sections"]]
    n = counts[0]
    device = blob.device

    # Resolve the escape levels from the raw level up, into the level-1 escapes in order. These are small.
    escaped_values = unpack_bits(sections[len(levels)], e, counts[len(levels)])
    for i in range(len(levels) - 1, 0, -1):
        escaped_values = _decode_level(unpack_bits(sections[i], levels[i]["bits"], counts[i]), levels[i], escaped_values)

    rest_planes, rest_extra = divmod(1 + m, 8)
    rest_sections = sections[len(levels) + 1:]
    width = torch.empty((), dtype=view).element_size()
    out = torch.empty(n, dtype=view, device=device)
    out_bytes = out.view(torch.uint8).view(n, width)
    b = levels[0]["bits"]
    level1_bytes = escape_offset = 0
    for c, begin in enumerate(range(0, n, chunk)):
        size = min(chunk, n - begin)
        packed = sections[0][level1_bytes:level1_bytes + (size + 7) // 8 * b]
        level1_bytes += size // 8 * b  # every chunk but the last is a whole number of groups
        escapes = info["chunk_escapes"][c]
        exponent = _decode_level(unpack_bits(packed, b, size), levels[0], escaped_values[escape_offset:escape_offset + escapes])
        escape_offset += escapes

        # The rest bits of this chunk, one uint8 per element for each byte of them.
        rest = [rest_sections[j][begin:begin + size] for j in range(rest_planes)]
        if rest_extra:
            packed = rest_sections[rest_planes][begin // 8 * rest_extra:(begin + size + 7) // 8 * rest_extra]
            rest.append(unpack_bits(packed, rest_extra, size))
        _assemble(out_bytes[begin:begin + size], exponent, rest, e, m)
        if dtype in FOLDED:  # the fold is its own inverse
            values = out_bytes[begin:begin + size].view(-1)
            values ^= torch.bitwise_right_shift(values, 7, out=exponent).mul_(0x7F)
    return out.view(dtype).view(info["shape"])


def _assemble(out_bytes, exponent, rest, e, m):
    """Writes sign, exponent and mantissa into each byte of the output, using uint8 operations only.
    rest[j] holds bits 8j..8j+7 of (sign << m | mantissa)."""
    scratch = torch.empty_like(exponent)
    width = out_bytes.shape[1]
    for k in range(width):
        byte = out_bytes[:, k]
        written = False

        def put(value):
            nonlocal written
            if written:
                byte.bitwise_or_(value)
            else:
                byte.copy_(value)
                written = True

        mantissa_bits = min(max(m - 8 * k, 0), 8)
        if mantissa_bits == 8:
            put(rest[k])
        elif mantissa_bits:
            put(torch.bitwise_and(rest[k], (1 << mantissa_bits) - 1, out=scratch))
        shift = m - 8 * k  # where the exponent starts, relative to this byte
        if 0 <= shift < 8:
            put(torch.bitwise_left_shift(exponent, shift, out=scratch))  # uint8: bits past this byte fall off
        elif -e < shift < 0:
            put(torch.bitwise_right_shift(exponent, -shift, out=scratch))
        if k == width - 1:  # the sign is bit m of rest, and the top bit of the value
            sign = torch.bitwise_right_shift(rest[m // 8], m % 8, out=scratch)
            put(sign.bitwise_left_shift_(7))
        if not written:
            byte.zero_()


LOOKUP_PIECE = 1 << 20  # codes widened to int32 at a time for the table lookup


def temporary_memory(info):
    """An upper bound on the temporary memory decode() takes for a blob, besides its output. The escape streams
    are decoded whole first (about 2-8 bytes per entry, plus 9 per entry of the stream after), then the values a
    chunk at a time (up to 8 bytes per value and 8 per escape, next to the level-1 escapes)."""
    counts, chunk = info["counts"], min(info["counts"][0], info["chunk"])
    streams = max((8 * counts[i] + 9 * counts[i + 1] + 4 * min(counts[i], LOOKUP_PIECE)
                   for i in range(1, len(info["levels"]))), default=0)
    chunks = counts[1] + 8 * chunk + 8 * max(info["chunk_escapes"]) + 4 * min(chunk, LOOKUP_PIECE)
    return max(streams, chunks, 9 * counts[len(info["levels"])])


def _decode_level(codes, level, escaped_values):
    """Maps a level's codes to exponents through its table and puts the escaped ones in place, as uint8."""
    device = codes.device
    lut = cached(("lut", tuple(level["lut"]), device),
                 lambda: torch.tensor(level["lut"] + [0], dtype=torch.uint8, device=device))
    escaped = codes == (1 << level["bits"]) - 1 if escaped_values.numel() else None
    positions = _positions(escaped, escaped_values.numel()) if escaped is not None else None
    if positions is not None:
        escaped = None  # not needed any more; free it before the lookup
    values = torch.empty_like(codes)
    for begin in range(0, codes.numel(), LOOKUP_PIECE):
        piece = codes[begin:begin + LOOKUP_PIECE]
        torch.index_select(lut, 0, piece.to(torch.int32), out=values[begin:begin + LOOKUP_PIECE])
    if positions is not None:
        values.index_put_((positions,), escaped_values)
    elif escaped is not None:
        # Without nonzero_static: the rank of each escape among the escapes, then a gather. Four bytes per value.
        rank = escaped.to(torch.int32)
        rank.cumsum_(0)
        rank -= 1
        rank.clamp_(min=0)
        torch.where(escaped, escaped_values.index_select(0, rank), values, out=values)
    return values


_NONZERO_STATIC = {}  # device type: whether nonzero_static works there


def _positions(mask, count):
    """Positions of the `count` set elements of mask, found without waiting on the GPU (the count is known), or
    None where nonzero_static isn't available. Eight bytes per escape; masked_scatter would take eight per value."""
    kind = mask.device.type
    if not _NONZERO_STATIC.get(kind, True):
        return None
    try:
        positions = torch.nonzero_static(mask, size=count).view(-1)
    except (AttributeError, NotImplementedError, RuntimeError):
        if kind in _NONZERO_STATIC:
            raise
        _NONZERO_STATIC[kind] = False
        return None
    _NONZERO_STATIC[kind] = True
    return positions
