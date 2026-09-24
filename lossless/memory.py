"""Keeps a loaded model's weights losslessly compressed in RAM and VRAM.

Each layer weight is replaced by a comfy_kitchen QuantizedTensor with a
"LosslessLayout" whose dequantize is the exact decoder. ComfyUI's layers already
handle QuantizedTensor weights when moving and casting them (as they do for its
fp8 and fp4 files), and any operation without a special path decodes the weight
first, so the model computes with exactly its original weights while only the
compressed bytes stay resident. The price is decoding each layer's weight every
time it runs.

Plain weights in any float format (fp32, fp16, bf16, fp8) are compressed as they
are. Layers of ComfyUI's pre-quantized fp8 and int8 files (int8 convrot included)
keep their fp8 or int8 data compressed and get their own quantized tensor back,
with all its layout settings, right before their fp8 or int8 kernel runs. 4-bit
formats are left alone: their packed data doesn't compress.

On the GPU the weights are decoded by a Triton kernel when Triton is installed
(kernels.py), otherwise by the PyTorch decoder. Either way decoding needs some
memory next to the model. That is added to ComfyUI's estimate of what sampling
needs, so ComfyUI keeps it free instead of the GPU running out mid-step.
"""
import dataclasses
import types

import torch

import comfy.ops
from comfy.quant_ops import QuantizedLayout, QuantizedTensor, get_layout_class, register_layout_class
from comfy_kitchen.tensor.base import BaseLayoutParams

from . import codec, kernels

LAYOUT = "LosslessLayout"
MIN_ELEMENTS = 4096
BASE_FIELDS = {"scale", "orig_dtype", "orig_shape"}
CHUNK = 1 << 23  # values the PyTorch decoder handles at a time; bounds its temporary memory


@dataclasses.dataclass(frozen=True)
class LosslessParams(BaseLayoutParams):
    info: dict = None          # how to decode the data
    inner: str = None          # the ComfyUI quantized layout the data belongs to, or None for a plain weight
    inner_fields: dict = None  # that layout's other settings, e.g. int8's convrot and convrot_groupsize

    def __getattr__(self, name):
        # ComfyUI reads layout settings off a weight's params (convrot when saving, transposed before int8
        # matmuls); answer with the inner layout's.
        fields = self.__dict__.get("inner_fields")
        if fields and name in fields:
            return fields[name]
        raise AttributeError(name)


class LosslessLayout(QuantizedLayout):
    Params = LosslessParams

    @classmethod
    def quantize(cls, tensor, inner=None, **kwargs):
        # Used when ComfyUI bakes a LoRA into a weight.
        if inner is None:
            return pack(tensor, torch.ones((), device=tensor.device), tensor.dtype, tuple(tensor.shape), None)
        quantized = QuantizedTensor.from_float(tensor, inner, **kwargs)
        params = quantized._params
        return pack(quantized._qdata, params.scale, params.orig_dtype, params.orig_shape, inner, inner_fields=extra_fields(params))

    @classmethod
    def dequantize(cls, qdata, params):
        data = decode(qdata, params.info)
        if params.inner is not None:
            return inner_tensor(data, params).dequantize()
        return data.to(params.orig_dtype)

    @classmethod
    def requantize_kwargs(cls, qtensor):
        # What requantizing the inner layout takes (int8 convrot: per-channel, convrot and its group size).
        params = qtensor._params
        if params.inner is None:
            return {"inner": None}
        return {"inner": params.inner, **inner_requantize_kwargs(params)}

    @classmethod
    def get_plain_tensors(cls, qtensor):
        return qtensor._qdata, qtensor._params.scale

    @classmethod
    def state_dict_tensors(cls, qdata, params):
        # What ComfyUI's quantized layers put in state_dict(): a weight with the right shape and dtype (the fp8 data
        # for fp8 layers, next to its scale) that still only takes the compressed bytes.
        if params.inner is None:
            return {"": QuantizedTensor(qdata, LAYOUT, params)}
        data_dtype = codec.DTYPE_NAMES[params.info.get("raw", params.info.get("dtype"))]
        data = dataclasses.replace(params, scale=torch.ones((), device=qdata.device), orig_dtype=data_dtype, inner=None, inner_fields=None)
        return {"": QuantizedTensor(qdata, LAYOUT, data), "_scale": params.scale}


register_layout_class(LAYOUT, LosslessLayout)


def pack(data, scale, orig_dtype, orig_shape, inner, compress_on=None, inner_fields=None):
    """(blob, params) for `data`; data that doesn't compress is kept as raw bytes. Encodes on `compress_on`
    (by default where the data is) when it has room, else on the CPU, and returns the blob on data's device."""
    def compress(device):
        encoded = codec.encode(data.to(device), chunk=CHUNK)
        if encoded is None:
            return None
        blob, info = encoded
        if kernels.installed():
            blob, info = kernels.attach(blob, info)
        return blob.to(data.device), info

    encoded = codec.run_on(compress_on or data.device, codec.encode_memory(data, CHUNK), compress)
    if encoded is None:
        blob, info = data.contiguous().reshape(-1).view(torch.uint8), {"raw": str(data.dtype).removeprefix("torch."), "shape": list(data.shape)}
    else:
        blob, info = encoded
    return blob, LosslessParams(scale=scale, orig_dtype=orig_dtype, orig_shape=tuple(orig_shape), info=info, inner=inner,
                                inner_fields=inner_fields or None)


@torch.compiler.disable  # plain eager code; torch.compile would recompile it for every layer
def decode(qdata, info):
    if "raw" in info:
        return qdata.view(codec.DTYPE_NAMES[info["raw"]]).view(info["shape"])
    if "tables" in info and kernels.usable(qdata.device):
        decoded = kernels.decode(qdata, info)
        if decoded is not None:
            return decoded
    return codec.decode(qdata, info)


def inner_params(params):
    """The inner layout's own params for a compressed weight."""
    layout = get_layout_class(params.inner)
    return layout.Params(scale=params.scale, orig_dtype=params.orig_dtype, orig_shape=params.orig_shape, **(params.inner_fields or {}))


def inner_tensor(data, params):
    return QuantizedTensor(data, params.inner, inner_params(params))


def inner_requantize_kwargs(params):
    """The inner layout's requantize_kwargs, which only look at params."""
    return get_layout_class(params.inner).requantize_kwargs(types.SimpleNamespace(_params=inner_params(params)))


def extra_fields(params):
    """A quantized layout's settings besides scale, dtype and shape."""
    return {f.name: getattr(params, f.name) for f in dataclasses.fields(params) if f.name not in BASE_FIELDS}


def can_wrap(weight):
    """Whether a ComfyUI quantized weight's data can be compressed: fp8 or int8 data, and no tensor settings
    besides its scale (4-bit layouts have more, and their packed data doesn't compress)."""
    params = weight._params
    tensor_fields = params._tensor_fields() if hasattr(params, "_tensor_fields") else None
    return weight._qdata.dtype in codec.FORMATS and tensor_fields == ["scale"] and BASE_FIELDS <= {f.name for f in dataclasses.fields(params)}


def is_compressed(weight):
    return isinstance(weight, QuantizedTensor) and weight._layout_cls == LAYOUT


# A compressed layer becomes an instance of a subclass of its class with one of these mixed in. (Methods stored on
# the layer itself would reference the layer from itself, and such a model is only freed by a full garbage collect.)

class PlainLayer:
    """For layers with a compressed plain weight: the hooks ComfyUI's quantized layers have. LoRA patching gets
    the decoded weight (convert) and hands the result back (set), and .to() goes through the tensor subclass."""

    def convert_weight(self, weight, inplace=False, **kwargs):
        return weight.dequantize() if isinstance(weight, QuantizedTensor) else weight

    def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
        if return_weight:  # patched on the fly for one forward pass: no point re-encoding it
            return weight.to(self.weight.dtype)
        self.weight = torch.nn.Parameter(QuantizedTensor.from_float(weight.to(self.weight.dtype), LAYOUT), requires_grad=False)

    def _apply(self, fn, recurse=True):
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


class QuantizedLayer:
    """For ComfyUI fp8 layers whose fp8 data is compressed."""

    def _forward(self, input, weight, bias):
        # The fp8 kernel gets the layer's own quantized weight back.
        if is_compressed(weight) and weight._params.inner is not None:
            weight = inner_tensor(decode(weight._qdata, weight._params.info), weight._params)
        return super()._forward(input, weight, bias)

    def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
        current = self.weight
        if return_weight and is_compressed(current) and current._params.inner is not None:
            # A LoRA applied on the fly to an offloaded layer, for one forward pass: quantize the result the way
            # ComfyUI does (requantize_from_float of the fp8 or int8 weight), but don't compress it only to decode it again.
            options = {**inner_requantize_kwargs(current._params), "scale": "recalculate", "stochastic_rounding": seed, "inplace_ops": True}
            return QuantizedTensor.from_float(weight, current._params.inner, **options).to(current.dtype)
        return super().set_weight(weight, inplace_update=inplace_update, seed=seed, return_weight=return_weight, **kwargs)


def compressed_class(cls, hooks):
    """cls with the hooks mixed in, made once per class and kept on it."""
    attribute = f"_lossless_{hooks.__name__}"
    subclass = cls.__dict__.get(attribute)
    if subclass is None:
        subclass = type(cls.__name__, (hooks, cls), {"__module__": cls.__module__, "__qualname__": cls.__qualname__})
        setattr(cls, attribute, subclass)
    return subclass


def compress_module(module, compress_on=None):
    """Replaces a ComfyUI layer's weight with a compressed copy, in place. Returns (bytes before, bytes after),
    or None when the layer is left alone (not a ComfyUI layer, too small, 4-bit, or doesn't compress)."""
    weight = getattr(module, "weight", None)
    if not isinstance(module, comfy.ops.CastWeightBiasOp) or not isinstance(weight, torch.Tensor) or weight.numel() < MIN_ELEMENTS:
        return None
    if isinstance(weight, QuantizedTensor):
        params = weight._params
        if not can_wrap(weight) or not hasattr(module, "_forward"):
            return None
        data = weight._qdata
        blob, new_params = pack(data, params.scale, params.orig_dtype, params.orig_shape, weight._layout_cls, compress_on,
                                inner_fields=extra_fields(params))
    elif weight.dtype in codec.FORMATS and weight.is_floating_point() and weight.ndim >= 2:
        data = weight
        blob, new_params = pack(data, torch.ones((), device=weight.device), weight.dtype, weight.shape, None, compress_on)
    else:
        return None
    if "raw" in new_params.info:
        return None

    module.__class__ = compressed_class(type(module), PlainLayer if new_params.inner is None else QuantizedLayer)
    module.weight = torch.nn.Parameter(QuantizedTensor(blob, LAYOUT, new_params), requires_grad=False)
    return data.nbytes, blob.nbytes


def decode_memory(info, triton=False):
    """Memory that decoding a weight takes besides its compressed bytes: the decoded weight, plus the PyTorch
    decoder's temporaries unless the Triton kernel does the work."""
    decoded = info["counts"][0] * codec.DTYPE_NAMES[info["dtype"]].itemsize
    return decoded if triton else decoded + codec.temporary_memory(info)


def keep_compressed(model, compress_on=None):
    """Compresses the diffusion model's layer weights in place and returns (bytes before, bytes after, layers).
    `compress_on` is the device to encode on (the GPU is much faster). Also records how much memory decoding
    needs, for reserve_decode_memory."""
    before = after = layers = workspace = 0
    triton = compress_on is not None and kernels.usable(torch.device(compress_on))
    for module in model.model.diffusion_model.modules():
        sizes = compress_module(module, compress_on)
        if sizes is not None:
            before, after, layers = before + sizes[0], after + sizes[1], layers + 1
            workspace = max(workspace, decode_memory(module.weight._params.info, triton))
    model.size = 0  # ComfyUI recomputes it from the compressed weights
    model.model.lossless_decode_memory = workspace
    return before, after, layers


def reserve_decode_memory(executor, model, noise_shape, conds, *args, **kwargs):
    """PREPARE_SAMPLING wrapper: adds the memory decoding needs to ComfyUI's estimate of what sampling needs, so
    ComfyUI leaves that much VRAM free when it decides how much of the model to load. Without it, decoding can push
    a full GPU over its limit; on Windows the driver then moves memory to shared system RAM, and every following
    step and generation gets slower."""
    base = model.model
    extra = getattr(base, "lossless_decode_memory", 0)
    if not extra:
        return executor(model, noise_shape, conds, *args, **kwargs)
    estimate = base.memory_required
    base.memory_required = lambda *a, **k: estimate(*a, **k) + extra
    try:
        return executor(model, noise_shape, conds, *args, **kwargs)
    finally:
        del base.memory_required
