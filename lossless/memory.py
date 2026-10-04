"""Keeps a model's weights losslessly compressed in RAM and VRAM.

Each compressed layer weight is a comfy_kitchen QuantizedTensor with a
"LosslessLayout" whose dequantize is the exact decoder. ComfyUI's layers already
handle QuantizedTensor weights when moving and casting them (as they do for its
fp8 and fp4 files), and any operation without a special path decodes the weight
first, so the model computes with exactly its original weights while only the
compressed bytes stay resident. The price is decoding each layer's weight every
time it runs.

Weights get there two ways:
- The lossless loaders hand ComfyUI the compressed weights straight from the file
  (load_keeping_compressed), and LosslessOps layers adopt them as they are.
- compress_in_memory() compresses a loaded model's layers in place. It also covers
  what the loaders can't: ComfyUI's pre-quantized fp8 and int8 layers (int8 convrot
  included), whose fp8 or int8 data is compressed and handed back to the layer as
  its own quantized weight, with all its layout settings, right before its fp8 or
  int8 kernel runs; and layers that got decoded weights while loading. 4-bit
  formats are left alone: their packed data doesn't compress.

On the GPU the weights are decoded by a Triton kernel when Triton is installed
(kernels.py), otherwise by the PyTorch decoder. Either way decoding needs some
memory next to the model. That is added to ComfyUI's estimate of what sampling
needs (reserve_decode_memory), so ComfyUI keeps it free instead of the GPU
running out mid-step.
"""
import dataclasses
import json
import types

import safetensors
import torch

import comfy.float
import comfy.ops
from comfy.quant_ops import QuantizedLayout, QuantizedTensor, get_layout_class, register_layout_class
from comfy_kitchen.tensor.base import BaseLayoutParams

from . import codec, fileformat, kernels

LAYOUT = "LosslessLayout"
MIN_ELEMENTS = 4096  # smaller weights aren't worth it
BASE_FIELDS = {"scale", "orig_dtype", "orig_shape"}
CHUNK = 1 << 23  # values the PyTorch decoder handles at a time for weights compressed here; bounds its temporary memory


@dataclasses.dataclass(frozen=True)
class LosslessParams(BaseLayoutParams):
    info: dict = None          # how to decode the data; {"raw": dtype, "shape": ...} for data kept as it is
    inner: str = None          # the ComfyUI quantized layout the data belongs to (fp8 and int8 layers), or None
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
        # Used when ComfyUI bakes a LoRA into a weight. fp8 and int8 layers are requantized in their own format first.
        if inner is None:
            return pack(tensor, torch.ones((), device=tensor.device), tensor.dtype, tuple(tensor.shape))
        quantized = QuantizedTensor.from_float(tensor, inner, **kwargs)
        params = quantized._params
        return pack(quantized._qdata, params.scale, params.orig_dtype, params.orig_shape, inner, extra_fields(params))

    @classmethod
    def dequantize(cls, qdata, params):
        data = decode(qdata, params.info)
        if params.inner is not None:
            return inner_tensor(data, params).dequantize()
        return data.to(params.orig_dtype)

    @classmethod
    def requantize_kwargs(cls, qtensor):
        # What requantizing takes: for fp8 and int8 layers, their layout and its settings (int8 convrot: per channel,
        # convrot and its group size).
        params = qtensor._params
        if params.inner is None:
            return {"inner": None}
        return {"inner": params.inner, **inner_requantize_kwargs(params)}

    @classmethod
    def get_plain_tensors(cls, qtensor):
        return qtensor._qdata, qtensor._params.scale

    @classmethod
    def state_dict_tensors(cls, qdata, params):
        # What a layer's state_dict() holds, which is also how ComfyUI measures it: a weight with the right shape and
        # dtype (the fp8 or int8 data for those layers, next to its scale) that still only takes the compressed bytes.
        if params.inner is None:
            return {"": QuantizedTensor(qdata, LAYOUT, params)}
        data = dataclasses.replace(params, scale=torch.ones((), device=qdata.device), orig_dtype=data_dtype(params.info),
                                   inner=None, inner_fields=None)
        return {"": QuantizedTensor(qdata, LAYOUT, data), "_scale": params.scale}


register_layout_class(LAYOUT, LosslessLayout)


class LoadingTensor(QuantizedTensor):
    """A compressed weight in a state dict being loaded. LosslessOps layers adopt it as it is. Everything else gets
    plain values: a layer copying it into its own weight gets it decoded instead of an error, and ComfyUI's state dict
    conversions get a plain tensor to assemble joined weights in (empty_like, then copies into slices of it), where an
    empty compressed tensor would silently stay empty. Other operations decode it, as for any QuantizedTensor."""

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.ops.aten.copy_.default and not isinstance(args[0], QuantizedTensor) and isinstance(args[1], QuantizedTensor):
            return func(args[0], args[1].dequantize(), *args[2:], **kwargs)
        if func is torch.ops.aten.empty_like.default:
            source = args[0]
            return torch.empty(source.shape, dtype=kwargs.get("dtype") or source.dtype, device=kwargs.get("device") or source.device)
        if func is torch.ops.aten.detach.default:  # torch.nn.Parameter() wants the same type back
            return cls(args[0]._qdata.detach(), args[0]._layout_cls, args[0]._params)
        if func is torch.ops.aten.clone.default:
            return cls(args[0]._qdata.clone(), args[0]._layout_cls, args[0]._params)
        return super().__torch_dispatch__(func, types, args, kwargs)

    def adopted(self):
        """The weight as an ordinary compressed QuantizedTensor, for the layer that keeps it."""
        return QuantizedTensor(self._qdata, self._layout_cls, self._params)


@torch.compiler.disable  # plain eager code; torch.compile would recompile it for every layer
def decode(qdata, info):
    """A compressed weight's data, decoded: by the Triton kernel when the blob carries its tables and it can run on
    the blob's device, otherwise by the PyTorch decoder."""
    if "raw" in info:
        return qdata.view(getattr(torch, info["raw"])).view(info["shape"])
    if "tables" in info and kernels.usable(qdata.device):
        decoded = kernels.decode(qdata, info)
        if decoded is not None:
            return decoded
    return codec.decode(qdata, info)


def data_dtype(info):
    return getattr(torch, info["raw"]) if "raw" in info else codec.DTYPE_NAMES[info["dtype"]]


def pack(data, scale, orig_dtype, orig_shape, inner=None, inner_fields=None, encode_on=None):
    """(blob, params) for `data`, encoded on `encode_on` (by default where the data is) when it has room, else on the
    CPU, and returned on data's device. Data that doesn't compress is kept as its raw bytes."""
    def compress(device):
        encoded = codec.encode(data.to(device), chunk=CHUNK)
        if encoded is None:
            return None
        blob, info = encoded
        if kernels.installed() and torch.cuda.is_available():  # tables for the Triton decoder; about 0.2% more
            blob, info = kernels.attach(blob, info)
        return blob.to(data.device), info

    encoded = codec.run_on(encode_on or data.device, codec.encode_memory(data, CHUNK), compress)
    if encoded is None:
        blob = data.detach().contiguous().reshape(-1).view(torch.uint8)
        info = {"raw": str(data.dtype).removeprefix("torch."), "shape": list(data.shape)}
    else:
        blob, info = encoded
    return blob, LosslessParams(scale=scale, orig_dtype=orig_dtype, orig_shape=tuple(orig_shape), info=info,
                                inner=inner, inner_fields=inner_fields or None)


def inner_params(params):
    """The inner layout's own params for a compressed fp8 or int8 weight."""
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
    """Whether a ComfyUI quantized weight's data can be compressed: fp8 or int8 data, and no tensor settings besides
    its scale (4-bit layouts have more, and their packed data doesn't compress)."""
    params = weight._params
    tensor_fields = params._tensor_fields() if hasattr(params, "_tensor_fields") else None
    return weight._qdata.dtype in codec.FORMATS and tensor_fields == ["scale"] and BASE_FIELDS <= {f.name for f in dataclasses.fields(params)}


def is_compressed(weight):
    return isinstance(weight, QuantizedTensor) and weight._layout_cls == LAYOUT


def compressed_tensor(blob, info, tables_on=None):
    """A compressed weight for a state dict, from a blob and its info. With `tables_on` (a CUDA device) and Triton
    installed, the blob gets the tables the Triton decoder needs, computed there; it stays on the device it came from."""
    if tables_on is not None and kernels.usable(tables_on):
        try:
            attached, info = kernels.attach(blob.to(tables_on), info)
            blob = attached.to(blob.device)
        except codec.OUT_OF_MEMORY:  # this weight is decoded by PyTorch instead
            torch.cuda.empty_cache()
    dtype = codec.DTYPE_NAMES[info["dtype"]]
    params = LosslessParams(scale=torch.ones(()), orig_dtype=dtype, orig_shape=tuple(info["shape"]), info=info)
    return LoadingTensor(blob, LAYOUT, params)


def decoded_size(weight):
    """Bytes a compressed weight's data takes decoded."""
    info = weight._params.info
    if "raw" in info:
        return weight._qdata.nbytes
    return info["counts"][0] * codec.DTYPE_NAMES[info["dtype"]].itemsize


def decode_workspace(weights, triton):
    """Memory that decoding the largest of these compressed weights takes besides its compressed bytes: the
    decoded data, plus the PyTorch decoder's temporaries unless the Triton kernel does the work."""
    sizes = []
    for weight in weights:
        info = weight._params.info
        if "raw" not in info:
            decoded = decoded_size(weight)
            sizes.append(decoded if triton and "tables" in info else decoded + codec.temporary_memory(info))
    return max(sizes, default=0)


# A layer with a compressed weight becomes an instance of a subclass of its class with one of these mixed in. (Methods
# stored on the layer itself would reference the layer from itself, and such a model is only freed by a full garbage
# collection.)

class PlainLayer:
    """For layers with a compressed plain weight: the hooks ComfyUI's quantized layers have. LoRA patching gets the
    decoded weight (convert) and hands the result back (set), and .to() goes through the tensor subclass."""

    def convert_weight(self, weight, inplace=False, **kwargs):
        return weight.dequantize() if isinstance(weight, QuantizedTensor) else weight

    def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
        # Rounded back to the weight's dtype the way ComfyUI does it for layers without these hooks (stochastically for
        # fp8), so a LoRA gives exactly what it gives on the uncompressed model.
        weight = comfy.float.stochastic_rounding(weight, self.weight.dtype, seed=0 if seed is None else seed)
        if return_weight:  # patched on the fly for one forward pass: no point compressing it
            return weight
        if is_compressed(self.weight):
            weight = QuantizedTensor.from_float(weight, LAYOUT)
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

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
    """For ComfyUI's fp8 and int8 layers whose quantized data is compressed."""

    def _forward(self, input, weight, bias):
        # The fp8 or int8 kernel gets the layer's own quantized weight back.
        if is_compressed(weight) and weight._params.inner is not None:
            weight = inner_tensor(decode(weight._qdata, weight._params.info), weight._params)
        return super()._forward(input, weight, bias)

    def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
        current = self.weight
        if return_weight and is_compressed(current) and current._params.inner is not None:
            # A LoRA applied on the fly to an offloaded layer, for one forward pass: quantize the result the way
            # ComfyUI does (requantize_from_float of the fp8 or int8 weight), but don't compress it only to decode it.
            options = {**inner_requantize_kwargs(current._params), "scale": "recalculate", "stochastic_rounding": seed, "inplace_ops": True}
            return QuantizedTensor.from_float(weight, current._params.inner, **options).to(current.dtype)
        return super().set_weight(weight, inplace_update=inplace_update, seed=seed, return_weight=return_weight, **kwargs)


def compressed_class(cls, hooks):
    """cls with the hooks mixed in, made once per class and kept on it."""
    if issubclass(cls, hooks):
        return cls
    attribute = f"_lossless_{hooks.__name__}"
    subclass = cls.__dict__.get(attribute)
    if subclass is None:
        subclass = type(cls.__name__, (hooks, cls), {"__module__": cls.__module__, "__qualname__": cls.__qualname__})
        setattr(cls, attribute, subclass)
    return subclass


def compress_module(module, encode_on=None):
    """Compresses a ComfyUI layer's weight in place, or gives a layer whose weight was compressed while loading the
    hooks it needs. Returns whether the layer now holds a compressed weight. Leaves alone what isn't a ComfyUI linear,
    convolution or embedding layer, small weights, 4-bit layers, and weights that don't compress."""
    weight = getattr(module, "weight", None)
    if not isinstance(module, comfy.ops.CastWeightBiasOp) or not isinstance(weight, torch.Tensor):
        return False
    if weight.ndim != 2 and not isinstance(module, torch.nn.modules.conv._ConvNd):
        return False  # a bank of weights (such as MoE experts) is used piece by piece; decoding all of it each time would crawl
    if is_compressed(weight):
        hooks = PlainLayer if weight._params.inner is None else QuantizedLayer
        if isinstance(weight, LoadingTensor):  # a layer of ComfyUI's took the loading tensor itself
            module.weight = torch.nn.Parameter(weight.adopted(), requires_grad=False)
    elif weight.numel() < MIN_ELEMENTS or weight.is_meta:
        return False
    elif isinstance(weight, QuantizedTensor):
        if not can_wrap(weight) or not hasattr(module, "_forward"):
            return False
        p = weight._params
        blob, params = pack(weight._qdata, p.scale, p.orig_dtype, p.orig_shape, weight._layout_cls, extra_fields(p), encode_on)
        hooks = QuantizedLayer
    elif weight.dtype in codec.FORMATS and weight.is_floating_point():
        blob, params = pack(weight, torch.ones((), device=weight.device), weight.dtype, weight.shape, encode_on=encode_on)
        hooks = PlainLayer
    else:
        return False
    if not is_compressed(weight):
        if "raw" in params.info:
            return False  # doesn't compress: leave the layer as it is
        module.weight = torch.nn.Parameter(QuantizedTensor(blob, LAYOUT, params), requires_grad=False)
    module.__class__ = compressed_class(type(module), hooks)
    return True


def compressed_weights(model):
    return [m.weight for m in model.model.diffusion_model.modules() if is_compressed(getattr(m, "weight", None))]


def compress_in_memory(model, encode_on=None):
    """Compresses the diffusion model's layer weights in place: plain weights, and the data of ComfyUI's fp8 and int8
    layers. `encode_on` is the device to encode on (the GPU is much faster). Records the memory decoding needs, for
    reserve_decode_memory. Returns (bytes of the compressed weights decoded, their compressed bytes, how many)."""
    for module in model.model.diffusion_model.modules():
        compress_module(module, encode_on)
    weights = compressed_weights(model)
    device = model.load_device
    triton = device.type == "cuda" and kernels.usable(device)
    model.model.lossless_decode_memory = decode_workspace(weights, triton)
    model.size = 0  # ComfyUI measures the model again, from the compressed weights
    return sum(decoded_size(w) for w in weights), sum(w._qdata.nbytes for w in weights), len(weights)


RESERVE_KEY = "lossless_decode_memory"


def reserve_decode_memory(executor, model, noise_shape, conds, *args, **kwargs):
    """PREPARE_SAMPLING wrapper: adds the memory decoding needs to ComfyUI's estimate of what sampling needs, so
    ComfyUI leaves that much VRAM free when it decides how much of the model to load. Without it, decoding can push
    a full GPU over its limit; on Windows the driver then moves memory to shared system RAM, and every following
    step and generation gets slower."""
    base = model.model
    extra = getattr(base, RESERVE_KEY, 0)
    if not extra:
        return executor(model, noise_shape, conds, *args, **kwargs)
    estimate = base.memory_required
    base.memory_required = lambda *a, **k: estimate(*a, **k) + extra
    try:
        return executor(model, noise_shape, conds, *args, **kwargs)
    finally:
        del base.memory_required


class CompressedWeight(PlainLayer):
    """For LosslessOps layers: adopts a compressed weight from the state dict as it is."""

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = prefix + "weight"
        weight = state_dict.get(key)
        if isinstance(weight, QuantizedTensor):
            del state_dict[key]
            self.weight = torch.nn.Parameter(weight.adopted() if isinstance(weight, LoadingTensor) else weight, requires_grad=False)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if isinstance(weight, QuantizedTensor) and key in missing_keys:
            missing_keys.remove(key)


class LosslessOps(comfy.ops.manual_cast):
    class Linear(CompressedWeight, comfy.ops.manual_cast.Linear):
        pass

    class Conv1d(CompressedWeight, comfy.ops.manual_cast.Conv1d):
        pass

    class Conv2d(CompressedWeight, comfy.ops.manual_cast.Conv2d):
        pass

    class Conv3d(CompressedWeight, comfy.ops.manual_cast.Conv3d):
        pass


QUANTIZED_MARKERS = ("comfy_quant", "scale_weight", "scaled_fp8")


def is_quantized_file(keys):
    """Whether a file holds ComfyUI pre-quantized layers (fp8 "scaled", mixed precision, int8 convrot, ...), which
    only ComfyUI's quantized layers can load."""
    return any(k.removesuffix(fileformat.BLOB_SUFFIX).endswith(QUANTIZED_MARKERS) for k in keys)


def load_keeping_compressed(path, prefix="", device=None):
    """(state_dict, metadata) where the layer weights under `prefix` stay compressed, or None when the file can't be
    loaded that way: not compressed, or already quantized with ComfyUI's own formats, which its quantized layers have
    to load (compress_in_memory compresses those after loading). `device` is the GPU to prepare the Triton decoder's
    tables on, if there is one."""
    with safetensors.safe_open(path, framework="pt") as f:
        metadata = f.metadata()
        keys = list(f.keys())
        if not fileformat.is_compressed(metadata) or is_quantized_file(keys):
            return None
        infos = json.loads(metadata[fileformat.TENSORS_KEY])
        state_dict = {}
        for key in keys:
            tensor = f.get_tensor(key)
            name = key.removesuffix(fileformat.BLOB_SUFFIX)
            if name == key:
                state_dict[key] = tensor
            elif name.startswith(prefix) and name.endswith(".weight") and len(infos[name]["shape"]) >= 2:
                state_dict[name] = compressed_tensor(tensor, infos[name], device)
            else:
                state_dict[name] = codec.decode(tensor, infos[name])
    return state_dict, json.loads(metadata[fileformat.METADATA_KEY]) or None
