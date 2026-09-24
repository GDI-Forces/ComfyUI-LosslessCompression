"""Bit-exact compression of floating point tensors.

A float is sign, exponent and mantissa bits. In trained weights the sign and
mantissa bits are close to random, but only a handful of exponent values occur,
so the exponent is where the savings are:

- The sign and mantissa bits ("rest") are stored unchanged, packed tightly.
- The exponent goes through a cascade of short fixed-width codes. Level 1 gives
  the most common exponents a b1-bit code, and its all-ones code means "look in
  the next level", which codes the rarer exponents the same way, down to a last
  level that stores them raw. Each level is a dense stream in element order, so
  decoding needs no positions: a masked_scatter fills the escapes back in.

Everything is plain PyTorch, so decoding runs on the CPU or the GPU. Level-1
codes are processed in fixed-size chunks to bound temporary memory, and the
escape count of every chunk is recorded so decoding never waits on the GPU.
"""
import torch

# dtype: (same-width integer view, exponent bits, mantissa bits)
FORMATS = {
    torch.float32: (torch.int32, 8, 23),
    torch.bfloat16: (torch.int16, 8, 7),
    torch.float16: (torch.int16, 5, 10),
    torch.float8_e4m3fn: (torch.uint8, 4, 3),
    torch.float8_e5m2: (torch.uint8, 5, 2),
}
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


def unpack_bits(packed, width, count):
    if width == 8:
        return packed[:count]
    if 8 % width == 0:
        shifts = torch.arange(0, 8, width, device=packed.device, dtype=torch.uint8)
        return ((packed.unsqueeze(1) >> shifts) & ((1 << width) - 1)).flatten()[:count]
    wide = torch.int32 if width <= 3 else torch.int64
    acc = (packed.view(-1, width).to(wide) << (torch.arange(width, device=packed.device, dtype=wide) * 8)).sum(1, dtype=wide)
    values = (acc.unsqueeze(1) >> (torch.arange(8, device=packed.device, dtype=wide) * width)) & ((1 << width) - 1)
    return values.to(torch.uint8).flatten()[:count]


def split_bits(flat, dtype):
    """(exponent, rest) of every element of a flat chunk as int32; rest is the sign above the mantissa."""
    view, e, m = FORMATS[dtype]
    bits = flat.view(view).to(torch.int32)
    if view != torch.int32:
        bits &= (1 << (1 + e + m)) - 1  # undo sign extension of int16
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


def encode(tensor):
    """(blob, info) for a tensor, or None when compressing it wouldn't save at least 1%.
    Runs on the tensor's device and returns the blob there."""
    if tensor.dtype not in FORMATS or tensor.numel() == 0:
        return None
    _, e, m = FORMATS[tensor.dtype]
    device = tensor.device
    chunks = tensor.detach().contiguous().view(-1).split(CHUNK)
    n = tensor.numel()

    counts = sum(torch.bincount(split_bits(c, tensor.dtype)[0], minlength=1 << e) for c in chunks)
    order = counts.argsort(descending=True, stable=True)
    widths, exponent_bits = plan_levels(counts[order][:int((counts > 0).sum())].tolist(), e)
    if not widths or n * (1 + m) + exponent_bits > n * (1 + e + m) * (1 - MIN_SAVING):
        return None  # not worth encoding; checked again below on the real size, which includes padding
    rank = torch.empty(1 << e, dtype=torch.int32, device=device)
    rank[order] = torch.arange(1 << e, dtype=torch.int32, device=device)
    rest_planes, rest_extra = divmod(1 + m, 8)

    # Level 1 and the rest bits, a chunk at a time; escapes are few, so collect them whole.
    escape = (1 << widths[0]) - 1
    level1, planes, extra, chunk_escapes, escaped_ranks, escaped_values = [], [[] for _ in range(rest_planes)], [], [], [], []
    for chunk in chunks:
        exponent, rest = split_bits(chunk, tensor.dtype)
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
        "chunk": CHUNK,
        "chunk_escapes": chunk_escapes,
        "sections": offsets,
    }
    blob = torch.cat(sections)
    if blob.numel() > tensor.numel() * tensor.element_size() * (1 - MIN_SAVING):
        return None
    return blob, info


def decode(blob, info):
    """The exact original tensor, on the blob's device."""
    dtype = DTYPE_NAMES[info["dtype"]]
    view, e, m = FORMATS[dtype]
    levels, counts, chunk = info["levels"], info["counts"], info["chunk"]
    sections = [blob[offset:offset + size] for offset, size in info["sections"]]
    n = counts[0]
    device = blob.device

    # Resolve the escape levels from the raw level up, into the level-1 escapes in order.
    escaped_values = unpack_bits(sections[len(levels)], e, counts[len(levels)])
    for i in range(len(levels) - 1, 0, -1):
        escaped_values = _decode_level(sections[i], levels[i], counts[i], escaped_values, device)

    rest_planes, rest_extra = divmod(1 + m, 8)
    rest_sections = sections[len(levels) + 1:]
    out = torch.empty(n, dtype=view, device=device)
    level1_bytes = 0
    escape_offset = 0
    for c, begin in enumerate(range(0, n, chunk)):
        size = min(chunk, n - begin)
        b = levels[0]["bits"]
        packed = sections[0][level1_bytes:level1_bytes + (size + 7) // 8 * b]
        level1_bytes += size // 8 * b  # every chunk but the last is a whole number of groups
        escapes = info["chunk_escapes"][c]
        exponent = _decode_level(packed, levels[0], size, escaped_values[escape_offset:escape_offset + escapes], device)
        escape_offset += escapes

        # Reassemble in the dtype's own integer width; shifts into the sign bit wrap as intended.
        rest = torch.zeros(size, dtype=view, device=device)
        for j in range(rest_planes):
            rest |= rest_sections[j][begin:begin + size].to(view) << (8 * j)
        if rest_extra:
            packed = rest_sections[rest_planes][begin // 8 * rest_extra:(begin + size + 7) // 8 * rest_extra]
            rest |= unpack_bits(packed, rest_extra, size).to(view) << (8 * rest_planes)
        sign = (rest >> m) & 1
        out[begin:begin + size] = (sign << (e + m)) | (exponent.to(view) << m) | (rest & ((1 << m) - 1))
    return out.view(dtype).view(info["shape"])


def _decode_level(packed, level, count, escaped_values, device):
    codes = unpack_bits(packed, level["bits"], count)
    lut = torch.tensor(level["lut"] + [0], dtype=torch.uint8, device=device)
    values = lut.index_select(0, codes.to(torch.int32))
    return values.masked_scatter_(codes == (1 << level["bits"]) - 1, escaped_values)
