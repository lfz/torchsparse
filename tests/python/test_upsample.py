import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F_torch

from torchsparse import SparseTensor
from torchsparse.nn import functional as F

from .test_utils import generate_feature_map


def _build_sparse_tensor(
    device, shape, num_points, channels, stride=(2, 2, 2)
) -> SparseTensor:
    sparse_dict = generate_feature_map(
        shape, num_points, channels, with_dense=False, dtype=np.float32
    )
    coords = torch.from_numpy(sparse_dict["coords"][:, [3, 0, 1, 2]]).int().to(device)
    feats = torch.randn(coords.size(0), channels, device=device, requires_grad=True)
    tensor = SparseTensor(
        feats=feats,
        coords=coords,
        stride=stride,
        spatial_range=(len(num_points), *shape),
    )
    return tensor


def _dense_reference(tensor: SparseTensor) -> torch.Tensor:
    dense = tensor.dense()
    return dense.permute(0, 4, 1, 2, 3).contiguous()


@pytest.mark.parametrize("align_corners", [False, True])
def test_sparse_trilinear_upsample_matches_dense(align_corners):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required for sparse upsample tests.")
    device = torch.device("cuda")
    scale_factor = 2
    shape = (4, 4, 4)
    channels = 4
    num_points = [36, 32]

    coarse = _build_sparse_tensor(
        device=device,
        shape=shape,
        num_points=num_points,
        channels=channels,
        stride=(2, 2, 2),
    )
    default_feat = torch.zeros(channels, device=device, requires_grad=True)
    upsampled = F.upsample_trilinear3d(
        coarse, scale_factor=scale_factor, align_corners=align_corners, default_feat=default_feat
    )

    dense_ref = _dense_reference(coarse)
    dense_ref = F_torch.interpolate(
        dense_ref,
        scale_factor=scale_factor,
        mode="trilinear",
        align_corners=align_corners,
    )
    dense_ref = dense_ref.permute(0, 2, 3, 4, 1).contiguous()
    ref_feats = dense_ref[
        upsampled.C[:, 0], upsampled.C[:, 1], upsampled.C[:, 2], upsampled.C[:, 3]
    ]

    torch.testing.assert_close(
        upsampled.F.cpu(), ref_feats.cpu(), atol=1e-4, rtol=1e-4
    )

    grad_probe = torch.randn_like(upsampled.F)
    loss_sparse = (upsampled.F * grad_probe).sum()
    loss_dense = (ref_feats * grad_probe).sum()
    loss_sparse.backward(retain_graph=True)
    sparse_grad = coarse.F.grad.detach().clone()
    if coarse.F.grad is not None:
        coarse.F.grad.zero_()
    if default_feat.grad is not None:
        default_feat.grad.zero_()
    loss_dense.backward()
    dense_grad = coarse.F.grad.detach().clone()

    torch.testing.assert_close(
        sparse_grad.cpu(), dense_grad.cpu(), atol=1e-4, rtol=1e-4
    )


def test_sparse_nearest_upsample_matches_dense():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required for sparse upsample tests.")
    device = torch.device("cuda")
    scale_factor = 2
    shape = (4, 4, 4)
    channels = 3
    num_points = [24]

    coarse = _build_sparse_tensor(
        device=device,
        shape=shape,
        num_points=num_points,
        channels=channels,
        stride=(2, 2, 2),
    )
    default_feat = torch.zeros(channels, device=device, requires_grad=True)
    upsampled = F.upsample_nearest3d(
        coarse, scale_factor=scale_factor, align_corners=False, default_feat=default_feat
    )

    dense_ref = _dense_reference(coarse)
    dense_ref = F_torch.interpolate(
        dense_ref,
        scale_factor=scale_factor,
        mode="nearest",
    )
    dense_ref = dense_ref.permute(0, 2, 3, 4, 1).contiguous()
    ref_feats = dense_ref[
        upsampled.C[:, 0], upsampled.C[:, 1], upsampled.C[:, 2], upsampled.C[:, 3]
    ]

    torch.testing.assert_close(
        upsampled.F.cpu(), ref_feats.cpu(), atol=1e-4, rtol=1e-4
    )

    grad_probe = torch.randn_like(upsampled.F)
    loss_sparse = (upsampled.F * grad_probe).sum()
    loss_dense = (ref_feats * grad_probe).sum()
    loss_sparse.backward(retain_graph=True)
    sparse_grad = coarse.F.grad.detach().clone()
    default_feat_grad = default_feat.grad.detach().clone()
    if coarse.F.grad is not None:
        coarse.F.grad.zero_()
    if default_feat.grad is not None:
        default_feat.grad.zero_()
    loss_dense.backward()
    dense_grad = coarse.F.grad.detach().clone()

    torch.testing.assert_close(
        sparse_grad.cpu(), dense_grad.cpu(), atol=1e-4, rtol=1e-4
    )
    assert torch.isfinite(default_feat_grad).all()


def test_trilinear_default_parameter_receives_gradient():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required for sparse upsample tests.")
    device = torch.device("cuda")
    channels = 2

    coords = torch.tensor([[0, 0, 0, 0]], device=device, dtype=torch.int32)
    feats = torch.randn(1, channels, device=device, requires_grad=True)
    tensor = SparseTensor(
        feats=feats,
        coords=coords,
        stride=(2, 2, 2),
        spatial_range=(1, 2, 2, 2),
    )
    default_feat = nn.Parameter(torch.zeros(channels, device=device))
    output = F.upsample_trilinear3d(
        tensor, scale_factor=2, align_corners=True, default_feat=default_feat
    )

    loss = output.F.sum()
    loss.backward()
    assert torch.all(default_feat.grad != 0)
