import torch
from torch import nn

from torchsparse import SparseTensor
from torchsparse.nn import functional as F
from torchsparse.utils import make_ntuple

__all__ = ["GlobalAvgPool", "GlobalMaxPool", "MaxPool3d", "AvgPool3d"]


class GlobalAvgPool(nn.Module):
    def forward(self, input: SparseTensor) -> torch.Tensor:
        return F.global_avg_pool(input)


class GlobalMaxPool(nn.Module):
    def forward(self, input: SparseTensor) -> torch.Tensor:
        return F.global_max_pool(input)


class MaxPool3d(nn.Module):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        config=None,
    ) -> None:
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = make_ntuple(kernel_size, ndim=3)
        self.stride = make_ntuple(stride, ndim=3)
        self.padding = make_ntuple(padding, ndim=3)
        self._config = config

    def forward(self, input: SparseTensor) -> SparseTensor:
        return F.max_pool3d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            config=self._config,
            training=self.training,
        )


class AvgPool3d(nn.Module):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        config=None,
    ) -> None:
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = make_ntuple(kernel_size, ndim=3)
        self.stride = make_ntuple(stride, ndim=3)
        self.padding = make_ntuple(padding, ndim=3)
        self._config = config

    def forward(self, input: SparseTensor) -> SparseTensor:
        return F.avg_pool3d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            config=self._config,
            training=self.training,
        )
