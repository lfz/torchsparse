from typing import Tuple, Union

import torch

from torchsparse import SparseTensor
from torchsparse.nn.functional.devoxelize import spdevoxelize
from torchsparse.nn.functional.hash import sphash
from torchsparse.nn.functional.query import sphashquery
from torchsparse.utils import make_ntuple

__all__ = ["upsample_trilinear3d", "upsample_nearest3d"]

_CORNER_BITS = torch.tensor(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ],
    dtype=torch.long,
)


def _ensure_cuda_sparse_tensor(input: SparseTensor) -> None:
    if input.feats.device.type != "cuda":
        raise NotImplementedError("Sparse upsample currently supports CUDA tensors only.")
    if input.spatial_range is None:
        raise ValueError("Sparse upsample requires tensors with spatial_range specified.")


def _validate_scale_factor(scale_factor: Tuple[int, int, int]) -> None:
    for s in scale_factor:
        if not isinstance(s, int) or s < 1:
            raise ValueError(f"scale_factor must be positive integers. Got {scale_factor}.")


def _compute_output_sizes(
    spatial_range: Tuple[int, int, int, int],
    scale_factor: Tuple[int, int, int],
    align_corners: bool,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    size_in = tuple(int(spatial_range[d + 1]) for d in range(3))
    if align_corners:
        size_out = tuple(
            (size_in[d] - 1) * scale_factor[d] + 1 if size_in[d] > 0 else 0 for d in range(3)
        )
    else:
        size_out = tuple(size_in[d] * scale_factor[d] for d in range(3))
    spatial_range_out = (spatial_range[0], *size_out)
    return size_in, spatial_range_out


def _generate_output_coords(
    coords: torch.Tensor,
    scale_factor: Tuple[int, int, int],
    size_out: Tuple[int, int, int],
) -> torch.Tensor:
    if coords.numel() == 0:
        return coords.new_zeros((0, 4))

    device = coords.device
    dtype = coords.dtype
    scale_tensor = torch.tensor(scale_factor, dtype=dtype, device=device)

    offsets = torch.stack(
        torch.meshgrid(
            *[torch.arange(s, device=device, dtype=dtype) for s in scale_factor], indexing="ij"
        ),
        dim=-1,
    ).reshape(-1, 3)

    expanded = (
        coords[:, None, 1:].to(dtype)
        * scale_tensor.view(1, 1, 3)
        + offsets.view(1, -1, 3)
    )
    min_bound = torch.zeros((1, 1, 3), dtype=dtype, device=device)
    max_bound = (
        torch.tensor(size_out, dtype=dtype, device=device).view(1, 1, 3) - 1
    ).clamp_min_(0)
    expanded = torch.minimum(torch.maximum(expanded, min_bound), max_bound)

    batch = coords[:, 0].view(-1, 1, 1).expand(-1, offsets.size(0), 1)
    candidates = torch.cat([batch, expanded], dim=-1).reshape(-1, 4)
    unique_coords = torch.unique(candidates, dim=0)
    return unique_coords.to(dtype)


def _compute_axis_mappings(
    coords: torch.Tensor,
    size_in: Tuple[int, int, int],
    size_out: Tuple[int, int, int],
    scale_factor: Tuple[int, int, int],
    align_corners: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = coords.device
    coord_spatial = coords[:, 1:].to(torch.float32)
    lowers = []
    uppers = []
    ratios = []
    for axis in range(3):
        si = size_in[axis]
        so = size_out[axis]
        sf = scale_factor[axis]
        if si == 0:
            lower = torch.zeros_like(coord_spatial[:, axis])
            upper = lower
            ratio = torch.zeros_like(coord_spatial[:, axis])
        elif align_corners:
            if so <= 1 or si == 1:
                in_coord = torch.zeros_like(coord_spatial[:, axis])
            else:
                in_coord = coord_spatial[:, axis] * (si - 1) / (so - 1)
            lower = torch.floor(in_coord)
            upper = torch.clamp(lower + 1, max=si - 1)
            ratio = in_coord - lower
        else:
            if sf == 1 or si == 1:
                in_coord = coord_spatial[:, axis]
            else:
                in_coord = (coord_spatial[:, axis] + 0.5) / sf - 0.5
            lower = torch.floor(in_coord)
            upper = torch.clamp(lower + 1, max=si - 1)
            ratio = in_coord - lower
        lower = torch.clamp(lower, min=0, max=max(si - 1, 0))
        upper = torch.clamp(upper, min=0, max=max(si - 1, 0))
        ratio = torch.clamp(ratio, min=0.0, max=1.0)
        lowers.append(lower.to(torch.long))
        uppers.append(upper.to(torch.long))
        ratios.append(ratio)
    lower_idx = torch.stack(lowers, dim=1)
    upper_idx = torch.stack(uppers, dim=1)
    ratio_vals = torch.stack(ratios, dim=1)
    return lower_idx, upper_idx, ratio_vals


def _build_corner_coords(
    lower_idx: torch.Tensor,
    upper_idx: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    device = lower_idx.device
    corner_bits = _CORNER_BITS.to(device)
    corner_axes = []
    for axis in range(3):
        axis_vals = torch.stack(
            [lower_idx[:, axis], upper_idx[:, axis]],
            dim=1,
        )
        axis_corners = axis_vals[:, corner_bits[:, axis]]
        corner_axes.append(axis_corners)
    spatial = torch.stack(corner_axes, dim=2).to(lower_idx.dtype)
    batch = batch_idx.view(-1, 1).expand(-1, corner_bits.size(0)).unsqueeze(-1)
    return torch.cat([batch, spatial], dim=2)


def _compute_trilinear_weights(
    ratios: torch.Tensor, lower_idx: torch.Tensor, upper_idx: torch.Tensor
) -> torch.Tensor:
    device = ratios.device
    dtype = ratios.dtype
    corner_bits = _CORNER_BITS.to(device)
    weights = torch.ones((ratios.size(0), corner_bits.size(0)), device=device, dtype=dtype)
    for axis in range(3):
        lower_w = 1.0 - ratios[:, axis]
        upper_w = ratios[:, axis]
        same_mask = (lower_idx[:, axis] == upper_idx[:, axis]).to(dtype)
        lower_w = torch.where(same_mask.bool(), torch.ones_like(lower_w), lower_w)
        upper_w = torch.where(same_mask.bool(), torch.zeros_like(upper_w), upper_w)
        axis_weights = torch.stack([lower_w, upper_w], dim=1)
        weights *= axis_weights[:, corner_bits[:, axis]]
    return weights


def _compute_nearest_weights(
    ratios: torch.Tensor, lower_idx: torch.Tensor, upper_idx: torch.Tensor
) -> torch.Tensor:
    device = ratios.device
    corner_bits = _CORNER_BITS.to(device)
    nearest = torch.round(ratios + lower_idx.to(ratios.dtype))
    nearest = torch.clamp(nearest, min=0)
    bits = torch.zeros_like(nearest, dtype=torch.long)
    greater_mask = (nearest == upper_idx).long()
    bits = torch.where(greater_mask.bool(), torch.ones_like(bits), bits)
    corner_index = (
        bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]
    )
    weights = torch.zeros((ratios.size(0), corner_bits.size(0)), device=device, dtype=ratios.dtype)
    weights[torch.arange(ratios.size(0), device=device), corner_index] = 1.0
    return weights


def _lookup_indices(
    corner_coords: torch.Tensor,
    input_coords: torch.Tensor,
) -> torch.Tensor:
    if corner_coords.numel() == 0:
        return corner_coords.new_zeros((0, _CORNER_BITS.size(0)))
    ref_hash = sphash(input_coords)
    query_hash = sphash(corner_coords.reshape(-1, 4))
    indices = sphashquery(query_hash, ref_hash).view(corner_coords.size(0), -1)
    return indices


def _prepare_default_feat(
    default_feat: torch.Tensor, channels: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if default_feat is None:
        raise ValueError("default_feat must be provided for sparse upsample operations.")
    if default_feat.ndim != 1 or default_feat.numel() != channels:
        raise ValueError(
            f"default_feat must have shape ({channels},), but got {tuple(default_feat.shape)}."
        )
    if default_feat.device != device or default_feat.dtype != dtype:
        raise ValueError(
            "default_feat must live on the same device and share dtype with input features."
        )
    return default_feat


def _upsample_trilinear_align_corners_dense(
    input: SparseTensor,
    scale_factor: Tuple[int, int, int],
    default_feat: torch.Tensor,
) -> SparseTensor:
    from torch.nn.functional import interpolate

    device = input.feats.device
    channels = input.feats.size(1)
    size_in, spatial_range_out = _compute_output_sizes(
        input.spatial_range, scale_factor, True
    )
    coords_out = _generate_output_coords(input.coords, scale_factor, spatial_range_out[1:])
    if coords_out.numel() == 0:
        output = SparseTensor(
            coords=coords_out,
            feats=input.feats.new_zeros((0, channels)),
            stride=tuple(input.stride[axis] // scale_factor[axis] for axis in range(3)),
            spatial_range=spatial_range_out,
        )
        output._caches = input._caches
        return output

    dense = input.dense().permute(0, 4, 1, 2, 3).contiguous()
    dense_up = interpolate(
        dense,
        scale_factor=scale_factor,
        mode="trilinear",
        align_corners=True,
    )

    mask_feats = torch.ones(
        input.feats.size(0), 1, device=device, dtype=input.feats.dtype, requires_grad=False
    )
    mask_tensor = SparseTensor(
        mask_feats,
        input.coords,
        stride=input.stride,
        spatial_range=input.spatial_range,
    )
    mask_dense = mask_tensor.dense().permute(0, 4, 1, 2, 3).contiguous()
    mask_up = interpolate(
        mask_dense,
        scale_factor=scale_factor,
        mode="trilinear",
        align_corners=True,
    )

    dense_perm = dense_up.permute(0, 2, 3, 4, 1).contiguous()
    mask_perm = mask_up.permute(0, 2, 3, 4, 1).contiguous()
    b = coords_out[:, 0].long()
    x = coords_out[:, 1].long()
    y = coords_out[:, 2].long()
    z = coords_out[:, 3].long()
    gathered = dense_perm[b, x, y, z]
    mask_vals = mask_perm[b, x, y, z].clamp_(0.0, 1.0)
    mask_vals = mask_vals.expand_as(gathered)
    default_term = (1.0 - mask_vals) * default_feat.view(1, -1)
    feats_up = gathered + default_term

    output = SparseTensor(
        coords=coords_out.to(torch.int32),
        feats=feats_up,
        stride=tuple(input.stride[axis] // scale_factor[axis] for axis in range(3)),
        spatial_range=spatial_range_out,
    )
    output._caches = input._caches
    return output


def _upsample_sparse(
    input: SparseTensor,
    scale_factor: Union[int, Tuple[int, int, int]],
    align_corners: bool,
    mode: str,
    default_feat: torch.Tensor,
) -> SparseTensor:
    _ensure_cuda_sparse_tensor(input)
    scale_factor = make_ntuple(scale_factor, ndim=3)
    _validate_scale_factor(scale_factor)

    default_feat = _prepare_default_feat(
        default_feat, input.feats.size(1), input.feats.device, input.feats.dtype
    )
    if mode == "trilinear" and align_corners:
        return _upsample_trilinear_align_corners_dense(
            input, scale_factor, default_feat
        )

    for axis in range(3):
        if input.stride[axis] % scale_factor[axis] != 0:
            raise ValueError(
                f"Input stride {input.stride} is not divisible by scale_factor {scale_factor}."
            )
    effective_align = align_corners if mode == "trilinear" else False
    size_in, spatial_range_out = _compute_output_sizes(
        input.spatial_range, scale_factor, effective_align
    )
    coords_out = _generate_output_coords(input.coords, scale_factor, spatial_range_out[1:])
    if coords_out.numel() == 0:
        output = SparseTensor(
            coords=coords_out,
            feats=input.feats.new_zeros((0, input.feats.size(1))),
            stride=tuple(input.stride[axis] // scale_factor[axis] for axis in range(3)),
            spatial_range=spatial_range_out,
        )
        output._caches = input._caches
        return output

    lower_idx, upper_idx, ratios = _compute_axis_mappings(
        coords_out, size_in, spatial_range_out[1:], scale_factor, effective_align
    )
    corner_coords = _build_corner_coords(lower_idx, upper_idx, coords_out[:, 0])
    corner_coords = corner_coords.to(input.coords.dtype)
    indices = _lookup_indices(corner_coords, input.coords)
    channels = input.feats.size(1)
    if mode == "trilinear":
        weights = _compute_trilinear_weights(ratios, lower_idx, upper_idx)
    else:
        weights = _compute_nearest_weights(ratios, lower_idx, upper_idx)

    weights = weights.to(dtype=input.feats.dtype)
    default_index = input.feats.size(0)
    indices_clamped = indices.clone()
    missing_mask = indices_clamped < 0
    indices_clamped = indices_clamped.to(torch.int32)
    if missing_mask.any():
        indices_clamped[missing_mask] = default_index

    feats_ext = torch.cat([input.feats, default_feat.unsqueeze(0)], dim=0)
    feats_up = spdevoxelize(feats_ext, indices_clamped.contiguous(), weights.contiguous())

    output = SparseTensor(
        coords=coords_out.to(torch.int32),
        feats=feats_up,
        stride=tuple(input.stride[axis] // scale_factor[axis] for axis in range(3)),
        spatial_range=spatial_range_out,
    )
    output._caches = input._caches
    return output


def upsample_trilinear3d(
    input: SparseTensor,
    scale_factor: Union[int, Tuple[int, int, int]],
    align_corners: bool = False,
    default_feat: torch.Tensor = None,
) -> SparseTensor:
    return _upsample_sparse(
        input,
        scale_factor,
        align_corners,
        mode="trilinear",
        default_feat=default_feat,
    )


def upsample_nearest3d(
    input: SparseTensor,
    scale_factor: Union[int, Tuple[int, int, int]],
    align_corners: bool = False,
    default_feat: torch.Tensor = None,
) -> SparseTensor:
    return _upsample_sparse(
        input,
        scale_factor,
        align_corners=False,
        mode="nearest",
        default_feat=default_feat,
    )
