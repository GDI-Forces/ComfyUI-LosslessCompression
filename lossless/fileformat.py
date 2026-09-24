"""Losslessly compressed model files.

A compressed file is an ordinary .safetensors file. Every tensor that compresses
is stored as one uint8 blob named "<name>::lossless", with what decoding needs in
the header metadata; everything else is stored unchanged. The original file's
own metadata is kept, so decompressing gives back the exact same tensors and
metadata.
"""
import json
import os
import shutil
import tempfile

import safetensors
import torch

from . import codec

FORMAT_KEY = "lossless_format"
TENSORS_KEY = "lossless_tensors"
METADATA_KEY = "lossless_metadata"
BLOB_SUFFIX = "::lossless"
MIN_ELEMENTS = 4096  # smaller tensors aren't worth their header entry

SAFETENSORS_DTYPES = {
    torch.float64: "F64", torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
    torch.float8_e4m3fn: "F8_E4M3", torch.float8_e5m2: "F8_E5M2",
    torch.int64: "I64", torch.int32: "I32", torch.int16: "I16", torch.int8: "I8",
    torch.uint64: "U64", torch.uint32: "U32", torch.uint16: "U16", torch.uint8: "U8", torch.bool: "BOOL",
}


def is_compressed(metadata):
    return bool(metadata) and FORMAT_KEY in metadata


def compress_file(src, dst, on_tensor=None, device=None):
    """Writes a compressed copy of a .safetensors, .ckpt, .pt or .pth model. Returns (original bytes, compressed bytes).
    Tensors are encoded on `device` (the GPU is much faster); on_tensor(done, total) is called after each one."""
    metadata, total, tensors = _read(src)
    infos, original, written = {}, 0, 0

    def compressed():
        nonlocal original, written
        for done, (name, tensor) in enumerate(tensors, 1):
            original += tensor.nbytes
            encoded = codec.encode(tensor.to(device) if device else tensor) if tensor.numel() >= MIN_ELEMENTS else None
            if encoded is None:
                written += tensor.nbytes
                yield name, tensor
            else:
                blob, infos[name] = encoded
                written += blob.numel()
                yield name + BLOB_SUFFIX, blob
            if on_tensor:
                on_tensor(done, total)

    # The header, which carries the decoding info, is only complete once every tensor is encoded.
    _write(dst, compressed(), lambda: {
        FORMAT_KEY: "1",
        TENSORS_KEY: json.dumps(infos, separators=(",", ":")),
        METADATA_KEY: json.dumps(metadata or {}),
    })
    return original, written


def load(path, device=None):
    """(state_dict, metadata) of any .safetensors file, decoding a compressed one.
    Blobs are decoded on `device` (the GPU is much faster) and returned on the CPU."""
    with safetensors.safe_open(path, framework="pt") as f:
        metadata = f.metadata()
        if not is_compressed(metadata):
            return {k: f.get_tensor(k) for k in f.keys()}, metadata
        infos = json.loads(metadata[TENSORS_KEY])
        state_dict = {}
        for key in f.keys():
            tensor = f.get_tensor(key)
            if key.endswith(BLOB_SUFFIX):
                name = key[:-len(BLOB_SUFFIX)]
                state_dict[name] = codec.decode(tensor.to(device) if device else tensor, infos[name]).cpu()
            else:
                state_dict[key] = tensor
    return state_dict, json.loads(metadata[METADATA_KEY]) or None


def verify(original, compressed, device=None):
    """Names of tensors that don't decode to exactly the original bits, or are missing or extra.
    An empty list means the compressed file restores the original exactly, metadata included."""
    metadata, _, tensors = _read(original)
    problems = []
    with safetensors.safe_open(compressed, framework="pt") as f:
        stored = f.metadata()
        if not is_compressed(stored):
            return ["(not a compressed file)"]
        infos = json.loads(stored[TENSORS_KEY])
        remaining = set(f.keys())
        for name, tensor in tensors:
            if name in infos:
                remaining.discard(name + BLOB_SUFFIX)
                blob = f.get_tensor(name + BLOB_SUFFIX)
                restored = codec.decode(blob.to(device) if device else blob, infos[name]).cpu()
            elif name in remaining:
                remaining.discard(name)
                restored = f.get_tensor(name)
            else:
                problems.append(name)
                continue
            if not _same_bits(tensor, restored):
                problems.append(name)
        problems += sorted(remaining)
        if json.loads(stored[METADATA_KEY]) != (metadata or {}):
            problems.append("(metadata)")
    return problems


def _same_bits(a, b):
    return (a.dtype == b.dtype and a.shape == b.shape
            and torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)))


def decompress_file(src, dst, device=None):
    """Writes the original, uncompressed .safetensors file back out."""
    state_dict, metadata = load(src, device)
    _write(dst, state_dict.items(), lambda: metadata or {})


def _read(path):
    """(metadata, tensor count, lazy (name, tensor) pairs) of a model file."""
    if path.endswith(".safetensors"):
        with safetensors.safe_open(path, framework="pt") as f:
            metadata, keys = f.metadata(), list(f.keys())

        def tensors():
            with safetensors.safe_open(path, framework="pt") as f:
                for key in keys:
                    yield key, f.get_tensor(key)
        return metadata, len(keys), tensors()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint = checkpoint.get("state_dict", checkpoint)
    tensors = [(k, v) for k, v in checkpoint.items() if isinstance(v, torch.Tensor)]
    return None, len(tensors), iter(tensors)


def _write(path, tensors, metadata):
    """Streams (name, tensor) pairs into a .safetensors file without holding them all in memory.
    `metadata` is called once every tensor has been written. Like the safetensors library, it stores
    tensors with larger elements first, so every tensor is aligned without gaps between them."""
    directory = os.path.dirname(os.path.abspath(path))
    groups = {size: tempfile.TemporaryFile(dir=directory) for size in (8, 4, 2, 1)}
    entries = []
    try:
        for name, tensor in tensors:
            raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            group = groups[tensor.element_size()]
            entries.append((name, tensor.dtype, list(tensor.shape), tensor.element_size(), group.tell(), raw.size))
            group.write(raw)

        base, offset = {}, 0
        for size, group in groups.items():
            base[size] = offset
            offset += group.tell()
        header = {name: {"dtype": SAFETENSORS_DTYPES[dtype], "shape": shape,
                         "data_offsets": [base[size] + start, base[size] + start + length]}
                  for name, dtype, shape, size, start, length in entries}
        extra = metadata()
        if extra:
            header["__metadata__"] = {k: str(v) for k, v in extra.items()}
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        with open(path, "wb") as out:
            out.write(len(encoded).to_bytes(8, "little"))
            out.write(encoded)
            for group in groups.values():
                group.seek(0)
                shutil.copyfileobj(group, out, 64 << 20)
    finally:
        for group in groups.values():
            group.close()
