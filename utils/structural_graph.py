"""Local structural statistics for Sparse-view 3DGS Gaussians.

This module is intentionally independent from the training loop.  It operates
on plain tensors and returns local geometry statistics that later phases can
consume without changing densification, pruning, rendering, or losses.
"""

import torch
import torch.nn.functional as F

from utils.general_utils import build_rotation


def build_knn_graph(xyz, k=12):
    """Build a CPU cKDTree kNN graph for Gaussian centers.

    Args:
        xyz: Tensor[N, 3] Gaussian centers. The tensor may live on CPU or GPU.
        k: Requested neighbor count. The effective output width is
            min(k, N - 1), because a Gaussian is never its own neighbor.

    Returns:
        LongTensor[N, min(k, N - 1)] on xyz.device.

    Raises:
        ImportError: If scipy is unavailable.
        ValueError: If the input shape or k is invalid.
    """
    _validate_xyz(xyz)
    if k < 0:
        raise ValueError("k must be non-negative.")

    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError(
            "build_knn_graph requires scipy.spatial.cKDTree. "
            "Install scipy in the active environment to build Phase 5 kNN graphs."
        ) from exc

    n_points = xyz.shape[0]
    effective_k = min(int(k), max(n_points - 1, 0))
    if effective_k == 0:
        return torch.empty((n_points, 0), dtype=torch.long, device=xyz.device)

    xyz_np = xyz.detach().cpu().numpy()
    tree = cKDTree(xyz_np)
    _, raw_indices = tree.query(xyz_np, k=effective_k + 1)

    raw_indices = torch.as_tensor(raw_indices, dtype=torch.long)
    if raw_indices.ndim == 1:
        raw_indices = raw_indices[:, None]

    neighbors = []
    all_indices = torch.arange(n_points, dtype=torch.long)
    for i in range(n_points):
        row = raw_indices[i]
        row = row[(row >= 0) & (row < n_points) & (row != i)]
        if row.numel() < effective_k:
            filler = all_indices[all_indices != i]
            row = torch.cat((row, filler), dim=0)
        neighbors.append(row[:effective_k])

    return torch.stack(neighbors, dim=0).to(device=xyz.device)


def estimate_gaussian_normals(scales, rotations, eps=1e-12):
    """Estimate Gaussian normals from activated scales and quaternions.

    Args:
        scales: Tensor[N, 3] activated Gaussian spatial scales, for example
            GaussianModel.get_scaling. Do not pass log-scale _scaling.
        rotations: Tensor[N, 4] Gaussian quaternions in the repository's
            (w, x, y, z) representation.
        eps: Small constant for robust normalization.

    Returns:
        Tensor[N, 3] unit normals. The sign is not canonicalized.
    """
    _validate_scales(scales)
    _validate_rotations(rotations)
    if scales.shape[0] != rotations.shape[0]:
        raise ValueError("scales and rotations must have the same first dimension.")

    rotation_matrices = _build_rotation_matrices(rotations, eps=eps)
    min_axis = torch.argmin(scales, dim=-1)
    local_axes = F.one_hot(min_axis, num_classes=3).to(dtype=scales.dtype, device=scales.device)
    normals = torch.bmm(rotation_matrices, local_axes.unsqueeze(-1)).squeeze(-1)
    return F.normalize(normals, dim=-1, eps=eps)


def compute_geometric_turning(xyz, normals, neighbor_indices, sigma_d=None, eps=1e-8):
    """Compute local geometric turning K_i.

    Args:
        xyz: Tensor[N, 3] Gaussian centers.
        normals: Tensor[N, 3] unit normals.
        neighbor_indices: LongTensor[N, K] local neighbor indices.
        sigma_d: Optional distance scale. When None, the mean neighbor distance
            is used as a simple global scale for this call.
        eps: Small denominator constant.

    Returns:
        Tensor[N] with weighted local normal turning.
    """
    _validate_xyz(xyz)
    _validate_normals(normals, xyz.shape[0])
    _validate_neighbors(neighbor_indices, xyz.shape[0])

    if neighbor_indices.shape[1] == 0:
        return torch.zeros((xyz.shape[0],), dtype=xyz.dtype, device=xyz.device)

    neighbor_xyz = xyz[neighbor_indices]
    neighbor_normals = normals[neighbor_indices]
    offsets = neighbor_xyz - xyz[:, None, :]
    dist2 = (offsets * offsets).sum(dim=-1)
    sigma = _resolve_sigma(dist2, sigma_d, eps)
    weights = torch.exp(-dist2 / (2.0 * sigma * sigma + eps))
    normal_agreement = (normals[:, None, :] * neighbor_normals).sum(dim=-1).abs().clamp(max=1.0)
    turning = 1.0 - normal_agreement
    return (weights * turning).sum(dim=-1) / (weights.sum(dim=-1) + eps)


def compute_continuity_defect(
    xyz,
    normals,
    neighbor_indices,
    normal_threshold=0.9,
    sigma_d=None,
    eps=1e-8,
):
    """Compute normalized local continuity defect D_i.

    Only neighbors with |n_i dot n_j| > normal_threshold are treated as belonging
    to the same local surface before measuring normal-direction offsets.

    Args:
        xyz: Tensor[N, 3] Gaussian centers.
        normals: Tensor[N, 3] unit normals.
        neighbor_indices: LongTensor[N, K] local neighbor indices.
        normal_threshold: Same-surface normal agreement threshold.
        sigma_d: Optional distance scale for neighbor weights. When None, the
            mean neighbor distance is used as a simple global scale.
        eps: Small denominator constant.

    Returns:
        Tensor[N] with normalized continuity defect.
    """
    _validate_xyz(xyz)
    _validate_normals(normals, xyz.shape[0])
    _validate_neighbors(neighbor_indices, xyz.shape[0])

    if neighbor_indices.shape[1] == 0:
        return torch.zeros((xyz.shape[0],), dtype=xyz.dtype, device=xyz.device)

    neighbor_xyz = xyz[neighbor_indices]
    neighbor_normals = normals[neighbor_indices]
    offsets = neighbor_xyz - xyz[:, None, :]
    dist = torch.linalg.norm(offsets, dim=-1)
    dist2 = dist * dist
    sigma = _resolve_sigma(dist2, sigma_d, eps)
    weights = torch.exp(-dist2 / (2.0 * sigma * sigma + eps))

    normal_agreement = (normals[:, None, :] * neighbor_normals).sum(dim=-1).abs()
    same_surface = normal_agreement > normal_threshold
    masked_weights = weights * same_surface.to(dtype=weights.dtype)

    plane_offsets = (offsets * normals[:, None, :]).sum(dim=-1).abs()
    weighted_offsets = (masked_weights * plane_offsets).sum(dim=-1)
    valid_counts = same_surface.sum(dim=-1)
    r_bar = torch.where(
        valid_counts > 0,
        (dist * same_surface.to(dtype=dist.dtype)).sum(dim=-1) / valid_counts.clamp_min(1).to(dtype=dist.dtype),
        torch.zeros((xyz.shape[0],), dtype=xyz.dtype, device=xyz.device),
    )
    denominator = r_bar * masked_weights.sum(dim=-1) + eps
    defect = weighted_offsets / denominator
    return torch.where(valid_counts > 0, defect, torch.zeros_like(defect))


def compute_redundancy(xyz, scales, neighbor_indices, eps=1e-8):
    """Compute local Gaussian redundancy R_i.

    Args:
        xyz: Tensor[N, 3] Gaussian centers.
        scales: Tensor[N, 3] activated Gaussian spatial scales.
        neighbor_indices: LongTensor[N, K] local neighbor indices.
        eps: Small denominator constant.

    Returns:
        Tensor[N] with overlap-based local redundancy.
    """
    _validate_xyz(xyz)
    _validate_scales(scales)
    _validate_neighbors(neighbor_indices, xyz.shape[0])
    if scales.shape[0] != xyz.shape[0]:
        raise ValueError("xyz and scales must have the same first dimension.")

    if neighbor_indices.shape[1] == 0:
        return torch.zeros((xyz.shape[0],), dtype=xyz.dtype, device=xyz.device)

    neighbor_xyz = xyz[neighbor_indices]
    offsets = neighbor_xyz - xyz[:, None, :]
    dist2 = (offsets * offsets).sum(dim=-1)
    mean_scales = scales.mean(dim=-1)
    neighbor_mean_scales = mean_scales[neighbor_indices]
    overlap_scale = mean_scales[:, None] + neighbor_mean_scales
    weights = torch.exp(-dist2 / (2.0 * overlap_scale.clamp_min(eps).pow(2)))
    return weights.sum(dim=-1)


def _build_rotation_matrices(rotations, eps=1e-12):
    """Reuse the repository helper on CUDA; use the same formula on CPU."""
    if rotations.is_cuda:
        return build_rotation(rotations).to(dtype=rotations.dtype, device=rotations.device)
    return _build_rotation_device_safe(rotations, eps=eps)


def _build_rotation_device_safe(rotations, eps=1e-12):
    """CPU-safe fallback matching the repository's build_rotation formula."""
    q = F.normalize(rotations, dim=-1, eps=eps)
    matrix = torch.zeros((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    matrix[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[:, 0, 1] = 2 * (x * y - r * z)
    matrix[:, 0, 2] = 2 * (x * z + r * y)
    matrix[:, 1, 0] = 2 * (x * y + r * z)
    matrix[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[:, 1, 2] = 2 * (y * z - r * x)
    matrix[:, 2, 0] = 2 * (x * z - r * y)
    matrix[:, 2, 1] = 2 * (y * z + r * x)
    matrix[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def _resolve_sigma(dist2, sigma_d, eps):
    if sigma_d is not None:
        return torch.as_tensor(sigma_d, dtype=dist2.dtype, device=dist2.device).clamp_min(eps)
    positive = dist2[dist2 > 0]
    if positive.numel() == 0:
        return torch.as_tensor(1.0, dtype=dist2.dtype, device=dist2.device)
    return torch.sqrt(positive.mean()).clamp_min(eps)


def _validate_xyz(xyz):
    if not torch.is_tensor(xyz) or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be a Tensor[N, 3].")


def _validate_scales(scales):
    if not torch.is_tensor(scales) or scales.ndim != 2 or scales.shape[1] != 3:
        raise ValueError("scales must be a Tensor[N, 3].")


def _validate_rotations(rotations):
    if not torch.is_tensor(rotations) or rotations.ndim != 2 or rotations.shape[1] != 4:
        raise ValueError("rotations must be a Tensor[N, 4].")


def _validate_normals(normals, n_points):
    if not torch.is_tensor(normals) or normals.shape != (n_points, 3):
        raise ValueError("normals must be a Tensor[N, 3] matching xyz.")


def _validate_neighbors(neighbor_indices, n_points):
    if not torch.is_tensor(neighbor_indices) or neighbor_indices.ndim != 2:
        raise ValueError("neighbor_indices must be a LongTensor[N, K].")
    if neighbor_indices.dtype != torch.long:
        raise ValueError("neighbor_indices must have dtype torch.long.")
    if neighbor_indices.shape[0] != n_points:
        raise ValueError("neighbor_indices first dimension must match N.")
    if neighbor_indices.numel() > 0:
        if neighbor_indices.min() < 0 or neighbor_indices.max() >= n_points:
            raise ValueError("neighbor_indices contains an out-of-range index.")
