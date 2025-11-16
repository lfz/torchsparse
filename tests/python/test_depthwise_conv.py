import numpy as np
import pytest
import torch

from torchsparse import SparseTensor
from torchsparse import nn as spnn
from torchsparse.nn.functional.conv.conv import _expand_depthwise_weight

from .test_utils import generate_feature_map


def _collapse_regular_grad(
    grad_weight: torch.Tensor, in_channels: int, depth_multiplier: int
) -> torch.Tensor:
    kernel_volume = grad_weight.shape[0]
    grad_view = grad_weight.view(
        kernel_volume, in_channels, in_channels, depth_multiplier
    )
    idx = torch.arange(in_channels, device=grad_weight.device)
    return grad_view[:, idx, idx, :]


def test_depthwise_conv_matches_dense():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for sparse depthwise convolution tests.")

    torch.manual_seed(0)
    device = torch.device("cuda")

    shape = (6, 6, 6)
    num_points = [70, 60]
    in_channels = 4
    depth_multiplier = 2
    kernel_size = 3

    sparse_dict = generate_feature_map(
        shape, num_points, in_channels, dtype=np.float32
    )
    coords = (
        torch.from_numpy(sparse_dict["coords"][:, [3, 0, 1, 2]])
        .int()
        .to(device)
    )
    base_feats = torch.from_numpy(sparse_dict["feats"]).to(torch.float32).to(device)

    feats_dw = base_feats.clone().detach().requires_grad_(True)
    feats_ref = base_feats.clone().detach().requires_grad_(True)

    spatial_range = (len(num_points), *shape)
    sparse_input_dw = SparseTensor(
        feats_dw, coords.clone(), spatial_range=spatial_range
    )
    sparse_input_ref = SparseTensor(
        feats_ref, coords.clone(), spatial_range=spatial_range
    )

    depthwise = spnn.DepthwiseConv3d(
        in_channels,
        kernel_size=kernel_size,
        depth_multiplier=depth_multiplier,
        bias=False,
    ).to(device)
    reference = spnn.Conv3d(
        in_channels,
        in_channels * depth_multiplier,
        kernel_size=kernel_size,
        bias=False,
    ).to(device)

    kernel = torch.randn(
        kernel_size**3, in_channels, depth_multiplier, device=device
    )
    depthwise.kernel.data.copy_(kernel)
    reference.kernel.data.copy_(
        _expand_depthwise_weight(kernel, depth_multiplier)
    )

    sparse_out_dw = depthwise(sparse_input_dw)
    sparse_out_ref = reference(sparse_input_ref)

    torch.testing.assert_close(sparse_out_dw.C, sparse_out_ref.C)
    torch.testing.assert_close(sparse_out_dw.F, sparse_out_ref.F, atol=1e-4, rtol=1e-4)

    grad_probe = torch.randn_like(sparse_out_dw.F)
    sparse_out_dw.F.backward(grad_probe)
    sparse_out_ref.F.backward(grad_probe)

    torch.testing.assert_close(
        feats_dw.grad, feats_ref.grad, atol=1e-4, rtol=1e-4
    )

    ref_grad_depthwise = _collapse_regular_grad(
        reference.kernel.grad, in_channels, depth_multiplier
    )
    torch.testing.assert_close(
        depthwise.kernel.grad,
        ref_grad_depthwise,
        atol=1e-4,
        rtol=1e-4,
    )
