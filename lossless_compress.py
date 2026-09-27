"""Losslessly compress model files for ComfyUI.

The weights keep their exact bits and dtype (fp32, fp16, bf16, fp8, ...); only the
file gets smaller. Typical savings: bf16 ~30%, fp8 ~20-25%, fp32 ~15%, fp16 ~12%.
ComfyUI loads the result with this pack's "(Lossless)" loader nodes.

  python lossless_compress.py compress model.safetensors      -> model.lossless.safetensors (verified)
  python lossless_compress.py decompress model.lossless.safetensors -o model.safetensors
  python lossless_compress.py verify model.safetensors model.lossless.safetensors
  python lossless_compress.py info model.lossless.safetensors

Run it with the Python that runs ComfyUI; it needs torch and safetensors.
"""
import argparse
import json
import math
import os
import sys
import time

import safetensors
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lossless import codec, fileformat  # noqa: E402


def gigabytes(size):
    return f"{size / 1024 ** 3:.2f} GB"


def default_device():
    return "cuda" if torch.cuda.is_available() else None


def compress(args):
    dst = args.output or os.path.splitext(args.file)[0] + ".lossless.safetensors"
    start = time.time()

    def progress(done, total):
        if sys.stdout.isatty():
            print(f"\r{done}/{total} tensors", end="", flush=True)

    original, written = fileformat.compress_file(args.file, dst, on_tensor=progress, device=default_device())
    print(f"\n{gigabytes(original)} -> {gigabytes(written)} ({100 * (1 - written / original):.1f}% smaller) "
          f"in {time.time() - start:.0f} s: {dst}")
    if not args.no_verify:
        problems = fileformat.verify(args.file, dst, default_device())
        if problems:
            sys.exit(f"VERIFY FAILED for {len(problems)} tensors, e.g. {problems[:5]}. Keep the original file.")
        print("Verified: every tensor decodes to exactly the original bits.")


def decompress(args):
    dst = args.output or args.file.replace(".lossless.safetensors", ".safetensors")
    if os.path.abspath(dst) == os.path.abspath(args.file):
        sys.exit("Pass -o with the output file name.")
    fileformat.decompress_file(args.file, dst, default_device())
    print(f"Wrote {dst}")


def verify(args):
    problems = fileformat.verify(args.original, args.compressed, default_device())
    if problems:
        sys.exit(f"MISMATCH in {len(problems)} tensors, e.g. {problems[:5]}")
    print("OK: every tensor decodes to exactly the original bits, and the metadata matches.")


def info(args):
    dtypes = {name: dtype for dtype, name in fileformat.SAFETENSORS_DTYPES.items()}
    by_dtype = {}
    with safetensors.safe_open(args.file, framework="pt") as f:
        metadata = f.metadata()
        if not fileformat.is_compressed(metadata):
            sys.exit("Not a losslessly compressed file.")
        infos = json.loads(metadata[fileformat.TENSORS_KEY])
        for key in f.keys():
            piece = f.get_slice(key)
            stored = math.prod(piece.get_shape()) * dtypes[piece.get_dtype()].itemsize
            name = key.removesuffix(fileformat.BLOB_SUFFIX)
            if name != key:
                dtype = codec.DTYPE_NAMES[infos[name]["dtype"]]
                original = math.prod(infos[name]["shape"]) * dtype.itemsize
            else:
                dtype, original = dtypes[piece.get_dtype()], stored
            before, after = by_dtype.get(dtype, (0, 0))
            by_dtype[dtype] = (before + original, after + stored)
    original = sum(before for before, _ in by_dtype.values())
    stored = sum(after for _, after in by_dtype.values())
    print(f"{args.file}: {gigabytes(original)} of weights stored in {gigabytes(stored)} ({100 * (1 - stored / original):.1f}% smaller)")
    for dtype, (before, after) in sorted(by_dtype.items(), key=lambda item: -item[1][0]):
        print(f"  {str(dtype).removeprefix('torch.'):14} {gigabytes(before):>10} -> {gigabytes(after):>10} ({100 * (1 - after / before):.1f}% smaller)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("compress", help="compress a .safetensors/.ckpt/.pt/.pth model")
    p.add_argument("file")
    p.add_argument("-o", "--output", help="default: <name>.lossless.safetensors next to the input")
    p.add_argument("--no-verify", action="store_true", help="skip checking that the result decodes exactly")
    p.set_defaults(run=compress)
    p = commands.add_parser("decompress", help="restore the original .safetensors")
    p.add_argument("file")
    p.add_argument("-o", "--output")
    p.set_defaults(run=decompress)
    p = commands.add_parser("verify", help="check a compressed file against the original")
    p.add_argument("original")
    p.add_argument("compressed")
    p.set_defaults(run=verify)
    p = commands.add_parser("info", help="show how much a compressed file saves")
    p.add_argument("file")
    p.set_defaults(run=info)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
