"""The Triton kernel behind kernels.py, in its own module so Triton is only imported when it's used."""
import triton
import triton.language as tl


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
def decode_kernel(blob, luts, starts, out, n, blocks, level1, level2, level3, raw, plane0, plane1, plane2, extra,
                  bits2, bits3, BITS1: tl.constexpr, CROSS1: tl.constexpr, LEVELS: tl.constexpr, E: tl.constexpr,
                  CROSS_RAW: tl.constexpr, M: tl.constexpr, PLANES: tl.constexpr, EXTRA: tl.constexpr,
                  CROSS_EXTRA: tl.constexpr, FOLD: tl.constexpr, LUT: tl.constexpr, BLOCK: tl.constexpr):
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
    if FOLD:  # int8: undo the XOR of the low 7 bits with the sign
        value = value ^ (((value >> 7) & 1) * 0x7F)
    tl.store(out + index, value.to(out.dtype.element_ty), mask=valid)
