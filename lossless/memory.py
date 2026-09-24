"""Keeps losslessly compressed weights compressed in RAM and VRAM.

Each compressed layer weight becomes a comfy_kitchen QuantizedTensor with a
"LosslessLayout" whose dequantize is the exact decoder. ComfyUI's layers already
handle QuantizedTensor weights when moving and casting them (as they do for
fp8 and fp4 checkpoints), and any operation without a special path decodes the
weight first, so the model computes with exactly the original weights while
only the compressed bytes stay resident. The price is decoding each layer's
weight every time it runs.
"""
import dataclasses
import json

import torch

import comfy.ops
from comfy.quant_ops import QuantizedLayout, QuantizedTensor, register_layout_class
from comfy_kitchen.tensor.base import BaseLayoutParams

from . import codec, fileformat

LAYOUT = "LosslessLayout"


@dataclasses.dataclass(frozen=True)
class LosslessParams(BaseLayoutParams):
    info: dict = None


class LosslessLayout(QuantizedLayout):
    Params = LosslessParams

    @classmethod
    def quantize(cls, tensor, **kwargs):
        # Used when a LoRA is baked into a weight; a weight that no longer compresses is kept raw.
        encoded = codec.encode(tensor)
        blob, info = encoded if encoded is not None else (tensor.contiguous().reshape(-1).view(torch.uint8), None)
        return blob, LosslessParams(
            scale=torch.ones((), device=tensor.device), orig_dtype=tensor.dtype, orig_shape=tuple(tensor.shape), info=info)

    @classmethod
    def dequantize(cls, qdata, params):
        if params.info is None:
            return qdata.view(params.orig_dtype).view(params.orig_shape)
        return codec.decode(qdata, params.info).to(params.orig_dtype)

    @classmethod
    def get_plain_tensors(cls, qtensor):
        return (qtensor._qdata,)

    @classmethod
    def state_dict_tensors(cls, qdata, params):
        return {"": qdata}


register_layout_class(LAYOUT, LosslessLayout)


def compressed_tensor(blob, info):
    dtype = codec.DTYPE_NAMES[info["dtype"]]
    params = LosslessParams(scale=torch.ones(()), orig_dtype=dtype, orig_shape=tuple(info["shape"]), info=info)
    return QuantizedTensor(blob, LAYOUT, params)


class CompressedWeight:
    """Mixin for ComfyUI layers: adopts a compressed weight from the state dict as is, and keeps it
    compressed when the layer moves between devices."""

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = prefix + "weight"
        weight = state_dict.get(key)
        if isinstance(weight, QuantizedTensor):
            del state_dict[key]
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if isinstance(weight, QuantizedTensor) and key in missing_keys:
            missing_keys.remove(key)

    # ComfyUI hands LoRA patching the decoded weight (convert) and gives the result back (set).
    def convert_weight(self, weight, inplace=False, **kwargs):
        return weight.dequantize() if isinstance(weight, QuantizedTensor) else weight

    def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
        if return_weight:  # patched on the fly for one forward pass: no point re-encoding it
            return weight.to(self.weight.dtype)
        if isinstance(self.weight, QuantizedTensor):
            weight = QuantizedTensor.from_float(weight.to(self.weight.dtype), LAYOUT)
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def _apply(self, fn, recurse=True):
        # Same as ComfyUI's quantized layers: re-wrap parameters so .to() goes through the tensor subclass.
        if recurse:
            for child in self.children():
                child._apply(fn)
        for key, param in self._parameters.items():
            if param is not None:
                self.register_parameter(key, torch.nn.Parameter(fn(param), requires_grad=False))
        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)
        return self


class LosslessOps(comfy.ops.manual_cast):
    class Linear(CompressedWeight, comfy.ops.manual_cast.Linear):
        pass

    class Conv1d(CompressedWeight, comfy.ops.manual_cast.Conv1d):
        pass

    class Conv2d(CompressedWeight, comfy.ops.manual_cast.Conv2d):
        pass

    class Conv3d(CompressedWeight, comfy.ops.manual_cast.Conv3d):
        pass


def load_keeping_compressed(path, prefix=""):
    """(state_dict, metadata) where the layer weights under `prefix` stay compressed, or None when the file can't
    be used that way: not compressed, or already quantized with ComfyUI's own formats, which need its quantized layers."""
    with fileformat.SafetensorsFile(path) as f:
        metadata = f.metadata
        keys = f.keys()
        if not fileformat.is_compressed(metadata) or any(k.endswith(("comfy_quant", "scale_weight", "scaled_fp8")) for k in keys):
            return None
        infos = json.loads(metadata[fileformat.TENSORS_KEY])
        state_dict = {}
        for key in keys:
            tensor = f.get(key)
            name = key.removesuffix(fileformat.BLOB_SUFFIX)
            if name == key:
                state_dict[key] = tensor
            elif name.startswith(prefix) and name.endswith(".weight") and len(infos[name]["shape"]) >= 2:
                state_dict[name] = compressed_tensor(tensor, infos[name])
            else:
                state_dict[name] = codec.decode(tensor, infos[name])
    return state_dict, json.loads(metadata[fileformat.METADATA_KEY]) or None
