import torch
from torch import nn

from torchsparse import SparseTensor
from torchsparse.nn import functional as F
from torchsparse.utils import make_ntuple

__all__ = ["TrilinearUpsample3d", "NearestUpsample3d"]


class TrilinearUpsample3d(nn.Module):
    def __init__(
        self,
        channels: int,
        scale_factor=2,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.scale_factor = make_ntuple(scale_factor, ndim=3)
        self.align_corners = align_corners
        self.default_feat = nn.Parameter(torch.zeros(channels))

    def forward(self, input: SparseTensor) -> SparseTensor:
        return F.upsample_trilinear3d(
            input,
            scale_factor=self.scale_factor,
            align_corners=self.align_corners,
            default_feat=self.default_feat,
        )


class NearestUpsample3d(nn.Module):
    def __init__(
        self,
        channels: int,
        scale_factor=2,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.scale_factor = make_ntuple(scale_factor, ndim=3)
        self.align_corners = align_corners
        self.default_feat = nn.Parameter(torch.zeros(channels))

    def forward(self, input: SparseTensor) -> SparseTensor:
        return F.upsample_nearest3d(
            input,
            scale_factor=self.scale_factor,
            align_corners=self.align_corners,
            default_feat=self.default_feat,
        )
