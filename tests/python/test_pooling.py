import numpy as np
import pytest
import torch

from torchsparse import SparseTensor
from torchsparse.nn import functional as F
from torchsparse.nn.functional.pooling import _build_pool_kmap

from .test_utils import generate_feature_map


def _prepare_inputs(device, shape, num_points, channels):
    sparse_dict = generate_feature_map(
        shape, num_points, channels, with_dense=False, dtype=np.float32
    )
    coords = torch.from_numpy(sparse_dict["coords"][:, [3, 0, 1, 2]]).int().to(device)
    total_points = coords.size(0)
    base_feats = torch.randn(total_points, channels, device=device)
    spatial_range = (len(num_points), *shape)
    return coords, base_feats.clone().detach(), spatial_range


def _reference_pool(feats: torch.Tensor, out_in_map: torch.Tensor, mode: str) -> torch.Tensor:
    outputs = []
    channels = feats.size(1)
    for row in out_in_map:
        idxs = row[row >= 0].long()
        if idxs.numel() == 0:
            outputs.append(torch.zeros(channels, device=feats.device, dtype=feats.dtype))
            continue
        vals = feats.index_select(0, idxs)
        if mode == "max":
            outputs.append(vals.max(dim=0).values)
        else:
            outputs.append(vals.mean(dim=0))
    if not outputs:
        return torch.zeros((0, channels), device=feats.device, dtype=feats.dtype)
    return torch.stack(outputs, dim=0)


def _run_pooling_consistency_test(pool_type: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required for sparse pooling tests.")

    device = torch.device("cuda")
    shape = (6, 6, 6)
    num_points = [70, 60]
    channels = 4
    kernel_size = 2
    stride = 2

    coords, base_feats, spatial_range = _prepare_inputs(device, shape, num_points, channels)

    feats_sparse_clone = base_feats.clone().detach().requires_grad_(True)
    feats_ref_clone = base_feats.clone().detach().requires_grad_(True)

    input_sparse = SparseTensor(
        feats_sparse_clone, coords, spatial_range=spatial_range
    )

    if pool_type == "max":
        sparse_out = F.max_pool3d(input_sparse, kernel_size=kernel_size, stride=stride)
        mode = "max"
    else:
        sparse_out = F.avg_pool3d(input_sparse, kernel_size=kernel_size, stride=stride)
        mode = "avg"

    kmap, _ = _build_pool_kmap(
        SparseTensor(
            feats_ref_clone, coords, spatial_range=spatial_range
        ),
        (kernel_size, kernel_size, kernel_size),
        (stride, stride, stride),
        (0, 0, 0),
        None,
        False,
    )

    num_outputs = kmap["coords"].shape[0]
    out_in_map = kmap["out_in_map"][:num_outputs].to(device)

    ref_out = _reference_pool(feats_ref_clone, out_in_map, mode)
    torch.testing.assert_close(
        sparse_out.C.cpu(), kmap["coords"].cpu(), msg="Coordinate mismatch."
    )
    torch.testing.assert_close(sparse_out.F.cpu(), ref_out.cpu(), atol=1e-4, rtol=1e-4)

    grad_probe = torch.randn_like(sparse_out.F)
    sparse_out.F.backward(grad_probe)
    ref_out.backward(grad_probe)
    torch.testing.assert_close(
        feats_sparse_clone.grad.cpu(),
        feats_ref_clone.grad.cpu(),
        atol=1e-4,
        rtol=1e-4,
    )


def test_sparse_max_pool_matches_dense():
    _run_pooling_consistency_test("max")


def test_sparse_avg_pool_matches_dense():
    _run_pooling_consistency_test("avg")
