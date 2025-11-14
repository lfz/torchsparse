from typing import Dict, Tuple, Union

import torch
from torch.autograd import Function

import torchsparse.backend
from torchsparse import SparseTensor
from torchsparse.utils import make_ntuple

from .conv.conv_config import (
    get_default_conv_config,
    get_global_conv_config,
)
from .conv.conv_mode import get_conv_mode
from .conv.kmap.build_kmap import build_kernel_map

__all__ = ["global_avg_pool", "global_max_pool", "max_pool3d", "avg_pool3d"]


def global_avg_pool(inputs: SparseTensor) -> torch.Tensor:
    batch_size = torch.max(inputs.coords[:, 0]).item() + 1
    outputs = []
    for k in range(batch_size):
        input = inputs.feats[inputs.coords[:, 0] == k]
        output = torch.mean(input, dim=0)
        outputs.append(output)
    outputs = torch.stack(outputs, dim=0)
    return outputs


def global_max_pool(inputs: SparseTensor) -> torch.Tensor:
    batch_size = torch.max(inputs.coords[:, 0]).item() + 1
    outputs = []
    for k in range(batch_size):
        input = inputs.feats[inputs.coords[:, 0] == k]
        output = torch.max(input, dim=0)[0]
        outputs.append(output)
    outputs = torch.stack(outputs, dim=0)
    return outputs


class SparseMaxPoolFunction(Function):
    @staticmethod
    def forward(ctx, feats: torch.Tensor, out_in_map: torch.Tensor) -> torch.Tensor:
        if feats.device.type != "cuda":
            raise NotImplementedError("Sparse max pooling currently supports CUDA tensors only.")
        outputs, argmax = torchsparse.backend.sparse_maxpool_forward_cuda(
            feats.contiguous(), out_in_map.contiguous()
        )
        ctx.save_for_backward(argmax)
        ctx.input_size = feats.size(0)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (argmax,) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_feats = torchsparse.backend.sparse_maxpool_backward_cuda(
            grad_output, argmax, ctx.input_size
        )
        return grad_feats, None


class SparseAvgPoolFunction(Function):
    @staticmethod
    def forward(ctx, feats: torch.Tensor, out_in_map: torch.Tensor) -> torch.Tensor:
        if feats.device.type != "cuda":
            raise NotImplementedError("Sparse avg pooling currently supports CUDA tensors only.")
        outputs, counts = torchsparse.backend.sparse_avgpool_forward_cuda(
            feats.contiguous(), out_in_map.contiguous()
        )
        ctx.save_for_backward(out_in_map.contiguous(), counts)
        ctx.input_size = feats.size(0)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        out_in_map, counts = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_feats = torchsparse.backend.sparse_avgpool_backward_cuda(
            grad_output, out_in_map, counts, ctx.input_size
        )
        return grad_feats, None


def _get_pool_config(config: Dict, training: bool) -> Dict:
    conv_mode = get_conv_mode()
    if config is None:
        config = get_global_conv_config()
        if config is None:
            config = get_default_conv_config(conv_mode=conv_mode, training=training)
    return config


def _build_pool_kmap(
    input: SparseTensor,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    config: Dict,
    training: bool,
) -> Tuple[Dict, Dict]:
    config = _get_pool_config(config, training)

    kmap_key = (input.stride, kernel_size, stride, (1, 1, 1))
    kmap = input._caches.kmaps.get(kmap_key)
    if kmap is None:
        kmap_mode = config.kmap_mode
        if kmap_mode != "hashmap_on_the_fly":
            hashmap = input._caches.hashmaps.get(input.stride)
        else:
            hashmap = input._caches.hashmaps.get(
                tuple(input.stride[k] * stride[k] for k in range(3))
            )
        if hashmap is None:
            hashmap_keys, hashmap_vals = None, None
        else:
            hashmap_keys, hashmap_vals = hashmap

        kmap = build_kernel_map(
            input.coords,
            input.feats.shape[0],
            kernel_size,
            stride,
            padding,
            hashmap_keys,
            hashmap_vals,
            input.spatial_range,
            kmap_mode,
            config.dataflow,
            downsample_mode=config.downsample_mode,
            training=training,
            ifsort=config.ifsort,
            split_mask_num=config.split_mask_num,
            split_mask_num_bwd=config.split_mask_num_bwd,
        )

        hashmap = [kmap["hashmap_keys"], kmap["hashmap_vals"]]
        input._caches.kmaps[kmap_key] = kmap
        input._caches.hashmaps[input.stride] = hashmap

    return kmap, config


def _pool3d(
    input: SparseTensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: Union[int, Tuple[int, int, int]],
    padding: Union[int, Tuple[int, int, int]],
    pool_fn,
    config: Dict = None,
    training: bool = False,
) -> SparseTensor:
    if input.feats.device.type != "cuda":
        raise NotImplementedError("Sparse pooling currently supports CUDA tensors only.")
    kernel_size = make_ntuple(kernel_size, ndim=3)
    stride = make_ntuple(stride, ndim=3)
    padding = make_ntuple(padding, ndim=3)

    kmap, config = _build_pool_kmap(
        input,
        kernel_size,
        stride,
        padding,
        config,
        training,
    )

    coords = kmap["coords"]
    out_in_map = kmap["out_in_map"]
    num_outputs = 0 if coords is None else coords.shape[0]
    if coords is None or coords.numel() == 0 or num_outputs == 0:
        feats = input.feats.new_empty((0, input.feats.size(1)))
    else:
        out_in_map = out_in_map[:num_outputs].int().contiguous()
        feats = pool_fn(input.feats, out_in_map)

    output = SparseTensor(
        coords=coords,
        feats=feats,
        stride=tuple(input.stride[k] * stride[k] for k in range(3)),
        spatial_range=kmap.get("spatial_range"),
    )
    output._caches = input._caches
    return output


def max_pool3d(
    input: SparseTensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: Union[int, Tuple[int, int, int]] = None,
    padding: Union[int, Tuple[int, int, int]] = 0,
    config: Dict = None,
    training: bool = False,
) -> SparseTensor:
    if stride is None:
        stride = kernel_size
    return _pool3d(
        input,
        kernel_size,
        stride,
        padding,
        SparseMaxPoolFunction.apply,
        config=config,
        training=training,
    )


def avg_pool3d(
    input: SparseTensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: Union[int, Tuple[int, int, int]] = None,
    padding: Union[int, Tuple[int, int, int]] = 0,
    config: Dict = None,
    training: bool = False,
) -> SparseTensor:
    if stride is None:
        stride = kernel_size
    return _pool3d(
        input,
        kernel_size,
        stride,
        padding,
        SparseAvgPoolFunction.apply,
        config=config,
        training=training,
    )
