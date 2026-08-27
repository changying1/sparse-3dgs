"""Gestalt continuity losses for Sparse-view 3DGS Phase 10.

The edge gate is intentionally discrete and non-differentiable.  The selected
edges are then reused by differentiable plane and normal continuity losses.
"""

import math

import torch

from utils.structural_graph import estimate_gaussian_normals


def select_same_surface_edges(
    xyz,
    normals,
    neighbor_indices,
    visible_view_count=None,
    normal_threshold=0.9,
    max_distance=None,
):
    """Select same-surface edges from an existing kNN graph.

    Args:
        xyz: Tensor[N, 3] Gaussian centers.
        normals: Tensor[N, 3] unit Gaussian normals.
        neighbor_indices: LongTensor[N, K] candidate neighbors.
        visible_view_count: Optional Tensor[N]. When provided, both endpoints
            must have at least one visible view.
        normal_threshold: Minimum absolute normal agreement.
        max_distance: Optional maximum edge length.

    Returns:
        LongTensor[E, 2] with rows [source_index, target_index].
    """
    _validate_xyz(xyz)
    _validate_normals(normals, xyz.shape[0])
    _validate_neighbors(neighbor_indices, xyz.shape[0])
    _validate_threshold(normal_threshold)
    _validate_same_device(xyz, normals, "normals")
    _validate_same_device(xyz, neighbor_indices, "neighbor_indices")
    if max_distance is not None:
        _validate_max_distance(max_distance)
    if visible_view_count is not None:
        _validate_visible_view_count(visible_view_count, xyz.shape[0])
        _validate_same_device(xyz, visible_view_count, "visible_view_count")

    n_points = xyz.shape[0]
    if n_points == 0 or neighbor_indices.shape[1] == 0:
        return torch.empty((0, 2), dtype=torch.long, device=neighbor_indices.device)

    with torch.no_grad():
        xyz_detached = xyz.detach()
        normals_detached = normals.detach()
        neighbors = neighbor_indices
        sources = torch.arange(n_points, dtype=torch.long, device=neighbors.device)[:, None].expand_as(neighbors)

        same_surface = sources != neighbors
        neighbor_normals = normals_detached[neighbors]
        normal_agreement = (normals_detached[:, None, :] * neighbor_normals).sum(dim=-1).abs()
        same_surface = same_surface & (normal_agreement >= normal_threshold)

        if visible_view_count is not None:
            visible = visible_view_count.detach() > 0
            same_surface = same_surface & visible[sources] & visible[neighbors]

        if max_distance is not None:
            offsets = xyz_detached[neighbors] - xyz_detached[:, None, :]
            distances = torch.linalg.norm(offsets, dim=-1)
            same_surface = same_surface & (distances <= max_distance)

        return torch.stack((sources[same_surface], neighbors[same_surface]), dim=-1)


def compute_plane_continuity_loss(xyz, normals, edges, eps=1e-8):
    """Compute normalized local plane continuity loss for selected edges."""
    _validate_xyz(xyz)
    _validate_normals(normals, xyz.shape[0])
    _validate_edges(edges, xyz.shape[0])
    _validate_eps(eps)

    if edges.shape[0] == 0:
        return _zero_like_inputs(xyz, normals)

    src = edges[:, 0]
    dst = edges[:, 1]
    offsets = xyz[dst] - xyz[src]
    detached_distances = torch.linalg.norm(offsets.detach(), dim=-1)
    r_bar = _source_mean_edge_distance(detached_distances, src, xyz.shape[0], eps)

    plane_offsets = (offsets * normals[src]).sum(dim=-1)
    normalized_offsets = plane_offsets / (r_bar[src] + eps)
    loss = normalized_offsets.square().mean()
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


def compute_normal_continuity_loss(normals, edges):
    """Compute sign-agnostic normal continuity loss for selected edges."""
    if not torch.is_tensor(normals) or normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError("normals must be a Tensor[N, 3].")
    _validate_edges(edges, normals.shape[0])

    if edges.shape[0] == 0:
        return normals.sum() * 0.0

    src = edges[:, 0]
    dst = edges[:, 1]
    dot = (normals[src] * normals[dst]).sum(dim=-1).clamp(min=-1.0, max=1.0)
    loss = (1.0 - dot.abs()).mean()
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


def compute_gestalt_loss(
    xyz,
    scales,
    rotations,
    neighbor_indices,
    visible_view_count=None,
    normal_threshold=0.9,
    max_distance=None,
    lambda_normal=1.0,
    eps=1e-8,
):
    """Compute the Phase 10 Gestalt continuity loss.

    Returns a dict with total loss, component losses, and selected edge count.
    """
    _validate_xyz(xyz)
    _validate_scales(scales, xyz.shape[0])
    _validate_rotations(rotations, xyz.shape[0])
    _validate_neighbors(neighbor_indices, xyz.shape[0])
    _validate_lambda_normal(lambda_normal)
    _validate_eps(eps)

    normals = estimate_gaussian_normals(scales, rotations)
    edges = select_same_surface_edges(
        xyz.detach(),
        normals.detach(),
        neighbor_indices,
        visible_view_count=visible_view_count,
        normal_threshold=normal_threshold,
        max_distance=max_distance,
    )
    plane_loss = compute_plane_continuity_loss(xyz, normals, edges, eps=eps)
    normal_loss = compute_normal_continuity_loss(normals, edges)
    total_loss = plane_loss + float(lambda_normal) * normal_loss
    return {
        "loss": total_loss,
        "plane_loss": plane_loss,
        "normal_loss": normal_loss,
        "edge_count": int(edges.shape[0]),
    }


def subsample_edges(edges, max_edges):
    """Deterministically subsample edges with uniform coverage."""
    _validate_edges_tensor(edges)
    if type(max_edges) is not int:
        raise ValueError("max_edges must be an integer.")
    if max_edges <= 0:
        raise ValueError("max_edges must be positive.")

    edge_count = edges.shape[0]
    if edge_count <= max_edges:
        return edges

    indices = torch.linspace(0, edge_count - 1, steps=max_edges, device=edges.device)
    indices = indices.round().to(dtype=torch.long)
    return edges[indices]


def _source_mean_edge_distance(distances, sources, n_points, eps):
    sums = torch.zeros((n_points,), dtype=distances.dtype, device=distances.device)
    counts = torch.zeros((n_points,), dtype=distances.dtype, device=distances.device)
    sums.scatter_add_(0, sources, distances)
    counts.scatter_add_(0, sources, torch.ones_like(distances))
    means = sums / counts.clamp_min(1.0)
    return means.clamp_min(eps)


def _zero_like_inputs(xyz, normals):
    return xyz.sum() * 0.0 + normals.sum() * 0.0


def _validate_xyz(xyz):
    if not torch.is_tensor(xyz) or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be a Tensor[N, 3].")


def _validate_scales(scales, n_points):
    if not torch.is_tensor(scales) or scales.shape != (n_points, 3):
        raise ValueError("scales must be a Tensor[N, 3] matching xyz.")


def _validate_rotations(rotations, n_points):
    if not torch.is_tensor(rotations) or rotations.shape != (n_points, 4):
        raise ValueError("rotations must be a Tensor[N, 4] matching xyz.")


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


def _validate_edges(edges, n_points):
    _validate_edges_tensor(edges)
    if edges.numel() > 0:
        if edges.min() < 0 or edges.max() >= n_points:
            raise ValueError("edges contains an out-of-range index.")
        if torch.any(edges[:, 0] == edges[:, 1]):
            raise ValueError("edges must not contain self edges.")


def _validate_edges_tensor(edges):
    if not torch.is_tensor(edges) or edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must be a LongTensor[E, 2].")
    if edges.dtype != torch.long:
        raise ValueError("edges must have dtype torch.long.")


def _validate_visible_view_count(visible_view_count, n_points):
    if not torch.is_tensor(visible_view_count) or visible_view_count.ndim != 1 or visible_view_count.shape != (n_points,):
        raise ValueError("visible_view_count must be a Tensor[N].")


def _validate_same_device(reference, tensor, name):
    if tensor.device != reference.device:
        raise ValueError(f"{name} must be on the same device as xyz.")


def _validate_threshold(normal_threshold):
    if not isinstance(normal_threshold, (int, float)) or not math.isfinite(float(normal_threshold)):
        raise ValueError("normal_threshold must be finite.")
    if not 0.0 <= float(normal_threshold) <= 1.0:
        raise ValueError("normal_threshold must be in [0, 1].")


def _validate_max_distance(max_distance):
    if not isinstance(max_distance, (int, float)) or not math.isfinite(float(max_distance)):
        raise ValueError("max_distance must be finite when provided.")
    if float(max_distance) < 0.0:
        raise ValueError("max_distance must be non-negative.")


def _validate_lambda_normal(lambda_normal):
    if not isinstance(lambda_normal, (int, float)) or not math.isfinite(float(lambda_normal)):
        raise ValueError("lambda_normal must be finite.")
    if float(lambda_normal) < 0.0:
        raise ValueError("lambda_normal must be non-negative.")


def _validate_eps(eps):
    if not isinstance(eps, (int, float)) or not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be a positive finite value.")
