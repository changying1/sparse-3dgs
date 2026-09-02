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


def compute_normal_reliability(scales, eps=1e-8):
    """Estimate detached normal reliability from activated Gaussian scales."""
    if not torch.is_tensor(scales) or scales.ndim != 2 or scales.shape[1] != 3:
        raise ValueError("scales must be a Tensor[N, 3].")
    _validate_eps(eps)
    if not torch.isfinite(scales).all():
        raise ValueError("scales must be finite.")

    with torch.no_grad():
        sorted_scales = torch.sort(scales.detach(), dim=-1).values
        s_min = sorted_scales[:, 0]
        s_mid = sorted_scales[:, 1]
        confidence = 1.0 - s_min / (s_mid + float(eps))
        confidence = torch.nan_to_num(confidence, nan=0.0, posinf=1.0, neginf=0.0)
        confidence = confidence.clamp(0.0, 1.0)
    return confidence.detach()


def compute_edge_reliability(point_confidence, edges):
    """Compute detached edge reliability as min endpoint reliability."""
    if not torch.is_tensor(point_confidence) or point_confidence.ndim != 1:
        raise ValueError("point_confidence must be a Tensor[N].")
    _validate_edges(edges, point_confidence.shape[0])
    if not torch.isfinite(point_confidence).all():
        raise ValueError("point_confidence must be finite.")

    if edges.shape[0] == 0:
        return point_confidence.detach().new_empty((0,))

    with torch.no_grad():
        src = edges[:, 0]
        dst = edges[:, 1]
        edge_confidence = torch.minimum(point_confidence.detach()[src], point_confidence.detach()[dst])
        edge_confidence = torch.nan_to_num(edge_confidence, nan=0.0, posinf=1.0, neginf=0.0)
        edge_confidence = edge_confidence.clamp(0.0, 1.0)
    return edge_confidence.detach()


def compute_common_neighbor_support(knn_indices, edges, eps=1e-8):
    """Compute detached common-neighbor overlap support for sampled edges."""
    if not torch.is_tensor(knn_indices) or knn_indices.ndim != 2:
        raise ValueError("knn_indices must be a LongTensor[N, K].")
    if knn_indices.dtype != torch.long:
        raise ValueError("knn_indices must have dtype torch.long.")
    _validate_edges(edges, knn_indices.shape[0])
    _validate_eps(eps)
    _validate_same_device(knn_indices, edges, "edges")
    if knn_indices.numel() > 0:
        if knn_indices.min() < 0 or knn_indices.max() >= knn_indices.shape[0]:
            raise ValueError("knn_indices contains an out-of-range index.")

    if edges.shape[0] == 0:
        return torch.empty((0,), dtype=torch.float32, device=knn_indices.device)
    if knn_indices.shape[1] == 0:
        return torch.zeros((edges.shape[0],), dtype=torch.float32, device=knn_indices.device)

    with torch.no_grad():
        src_neighbors = knn_indices.detach()[edges[:, 0]]
        dst_neighbors = knn_indices.detach()[edges[:, 1]]
        shared = src_neighbors[:, :, None] == dst_neighbors[:, None, :]
        intersection = shared.any(dim=-1).sum(dim=-1).to(dtype=torch.float32)
        min_degree = torch.full_like(intersection, float(knn_indices.shape[1]))
        support = intersection / (min_degree + float(eps))
        support = torch.nan_to_num(support, nan=0.0, posinf=1.0, neginf=0.0)
        support = support.clamp(0.0, 1.0)
    return support.detach()


def compute_surface_relation_reliability(normal_edge_reliability, relation_confidence):
    """Combine detached normal and surface-relation edge reliability."""
    if not torch.is_tensor(normal_edge_reliability) or normal_edge_reliability.ndim != 1:
        raise ValueError("normal_edge_reliability must be a Tensor[E].")
    if not torch.is_tensor(relation_confidence) or relation_confidence.shape != normal_edge_reliability.shape:
        raise ValueError("relation_confidence must be a Tensor[E] matching normal_edge_reliability.")
    _validate_same_device(normal_edge_reliability, relation_confidence, "relation_confidence")

    with torch.no_grad():
        normal = torch.nan_to_num(normal_edge_reliability.detach().float(), nan=0.0, posinf=1.0, neginf=0.0)
        relation = torch.nan_to_num(relation_confidence.detach().float(), nan=0.0, posinf=1.0, neginf=0.0)
        gate = normal.clamp(0.0, 1.0) * relation.clamp(0.0, 1.0)
        gate = torch.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return gate.detach().to(dtype=torch.promote_types(normal_edge_reliability.dtype, relation_confidence.dtype))


def compute_weighted_plane_continuity_loss(xyz, normals, edges, edge_weights, eps=1e-8):
    """Compute edge-weighted normalized local plane continuity loss."""
    _validate_xyz(xyz)
    _validate_normals(normals, xyz.shape[0])
    _validate_edges(edges, xyz.shape[0])
    _validate_edge_weights(edge_weights, edges.shape[0])
    _validate_eps(eps)
    _validate_same_device(xyz, edge_weights, "edge_weights")

    if edges.shape[0] == 0:
        return _zero_like_inputs(xyz, normals)

    weights = edge_weights.detach().to(dtype=xyz.dtype, device=xyz.device)
    weight_mass = weights.sum()
    if weight_mass.item() <= float(eps):
        return _zero_like_inputs(xyz, normals)

    src = edges[:, 0]
    dst = edges[:, 1]
    offsets = xyz[dst] - xyz[src]
    detached_distances = torch.linalg.norm(offsets.detach(), dim=-1)
    r_bar = _source_mean_edge_distance(detached_distances, src, xyz.shape[0], eps)

    plane_offsets = (offsets * normals[src]).sum(dim=-1)
    normalized_offsets = plane_offsets / (r_bar[src] + eps)
    loss = (weights * normalized_offsets.square()).sum() / (weight_mass + float(eps))
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


def compute_gated_plane_continuity_loss(xyz, normals, edges, edge_weights, eps=1e-8):
    """Compute confidence-gated plane continuity loss as mean(w * per-edge loss)."""
    _validate_xyz(xyz)
    _validate_normals(normals, xyz.shape[0])
    _validate_edges(edges, xyz.shape[0])
    _validate_edge_weights(edge_weights, edges.shape[0])
    _validate_eps(eps)
    _validate_same_device(xyz, edge_weights, "edge_weights")

    if edges.shape[0] == 0:
        return _zero_like_inputs(xyz, normals)

    weights = edge_weights.detach().to(dtype=xyz.dtype, device=xyz.device)
    src = edges[:, 0]
    dst = edges[:, 1]
    offsets = xyz[dst] - xyz[src]
    detached_distances = torch.linalg.norm(offsets.detach(), dim=-1)
    r_bar = _source_mean_edge_distance(detached_distances, src, xyz.shape[0], eps)

    plane_offsets = (offsets * normals[src]).sum(dim=-1)
    normalized_offsets = plane_offsets / (r_bar[src] + eps)
    loss = (weights * normalized_offsets.square()).mean()
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


def compute_weighted_normal_continuity_loss(normals, edges, edge_weights, eps=1e-8):
    """Compute edge-weighted sign-agnostic normal continuity loss."""
    if not torch.is_tensor(normals) or normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError("normals must be a Tensor[N, 3].")
    _validate_edges(edges, normals.shape[0])
    _validate_edge_weights(edge_weights, edges.shape[0])
    _validate_eps(eps)
    _validate_same_device(normals, edge_weights, "edge_weights")

    if edges.shape[0] == 0:
        return normals.sum() * 0.0

    weights = edge_weights.detach().to(dtype=normals.dtype, device=normals.device)
    weight_mass = weights.sum()
    if weight_mass.item() <= float(eps):
        return normals.sum() * 0.0

    src = edges[:, 0]
    dst = edges[:, 1]
    dot = (normals[src] * normals[dst]).sum(dim=-1).clamp(min=-1.0, max=1.0)
    per_edge_loss = 1.0 - dot.abs()
    loss = (weights * per_edge_loss).sum() / (weight_mass + float(eps))
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


def compute_gated_normal_continuity_loss(normals, edges, edge_weights, eps=1e-8):
    """Compute confidence-gated normal continuity loss as mean(w * per-edge loss)."""
    if not torch.is_tensor(normals) or normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError("normals must be a Tensor[N, 3].")
    _validate_edges(edges, normals.shape[0])
    _validate_edge_weights(edge_weights, edges.shape[0])
    _validate_eps(eps)
    _validate_same_device(normals, edge_weights, "edge_weights")

    if edges.shape[0] == 0:
        return normals.sum() * 0.0

    weights = edge_weights.detach().to(dtype=normals.dtype, device=normals.device)
    src = edges[:, 0]
    dst = edges[:, 1]
    dot = (normals[src] * normals[dst]).sum(dim=-1).clamp(min=-1.0, max=1.0)
    per_edge_loss = 1.0 - dot.abs()
    loss = (weights * per_edge_loss).mean()
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
    use_normal_reliability=False,
    weighting_mode=None,
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
    if weighting_mode is None:
        weighting_mode = "normalized" if use_normal_reliability else "none"
    if weighting_mode not in ("none", "normalized", "gated", "relation_gated"):
        raise ValueError("weighting_mode must be one of: none, normalized, gated, relation_gated.")
    if use_normal_reliability and weighting_mode == "none":
        weighting_mode = "normalized"

    if weighting_mode in ("normalized", "gated", "relation_gated"):
        point_confidence = compute_normal_reliability(scales, eps=eps)
        normal_edge_confidence = compute_edge_reliability(point_confidence, edges)
        edge_confidence = normal_edge_confidence
        relation_confidence = None
        if weighting_mode == "relation_gated":
            relation_confidence = compute_common_neighbor_support(neighbor_indices, edges, eps=eps)
            edge_confidence = compute_surface_relation_reliability(normal_edge_confidence, relation_confidence)
        if weighting_mode == "normalized":
            plane_loss = compute_weighted_plane_continuity_loss(xyz, normals, edges, edge_confidence, eps=eps)
            normal_loss = compute_weighted_normal_continuity_loss(normals, edges, edge_confidence, eps=eps)
        else:
            plane_loss = compute_gated_plane_continuity_loss(xyz, normals, edges, edge_confidence, eps=eps)
            normal_loss = compute_gated_normal_continuity_loss(normals, edges, edge_confidence, eps=eps)
    else:
        point_confidence = None
        edge_confidence = None
        plane_loss = compute_plane_continuity_loss(xyz, normals, edges, eps=eps)
        normal_loss = compute_normal_continuity_loss(normals, edges)
    total_loss = plane_loss + float(lambda_normal) * normal_loss
    result = {
        "loss": total_loss,
        "plane_loss": plane_loss,
        "normal_loss": normal_loss,
        "edge_count": int(edges.shape[0]),
    }
    if weighting_mode in ("normalized", "gated", "relation_gated"):
        result.update(_confidence_diagnostics(point_confidence, normal_edge_confidence))
        if weighting_mode == "relation_gated":
            result.update(_relation_diagnostics(relation_confidence, edge_confidence))
        result["weighting_mode"] = weighting_mode
    return result


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


def _confidence_diagnostics(point_confidence, edge_confidence):
    point_stats = _detached_stats(point_confidence)
    edge_stats = _detached_stats(edge_confidence)
    return {
        "confidence_mean": point_stats["mean"],
        "confidence_median": point_stats["median"],
        "confidence_q10": point_stats["q10"],
        "confidence_q90": point_stats["q90"],
        "edge_confidence_mean": edge_stats["mean"],
        "edge_confidence_median": edge_stats["median"],
        "edge_confidence_q10": edge_stats["q10"],
        "edge_confidence_q90": edge_stats["q90"],
        "edge_conf_mean": edge_stats["mean"],
        "edge_conf_median": edge_stats["median"],
        "edge_conf_q10": edge_stats["q10"],
        "edge_conf_q90": edge_stats["q90"],
        "effective_edge_mass": float(edge_confidence.detach().sum().item()) if edge_confidence.numel() else 0.0,
        "effective_strength_ratio": float(edge_confidence.detach().mean().item()) if edge_confidence.numel() else 0.0,
    }


def _relation_diagnostics(relation_confidence, final_gate):
    relation_stats = _detached_extended_stats(relation_confidence)
    gate_stats = _detached_extended_stats(final_gate)
    gate = final_gate.detach().float()
    nearzero_ratio = float((gate <= 1e-8).float().mean().item()) if gate.numel() else 0.0
    return {
        "relation_conf_mean": relation_stats["mean"],
        "relation_conf_min": relation_stats["min"],
        "relation_conf_max": relation_stats["max"],
        "relation_conf_q25": relation_stats["q25"],
        "relation_conf_q50": relation_stats["q50"],
        "relation_conf_q75": relation_stats["q75"],
        "final_gate_mean": gate_stats["mean"],
        "final_gate_min": gate_stats["min"],
        "final_gate_max": gate_stats["max"],
        "final_gate_q25": gate_stats["q25"],
        "final_gate_q50": gate_stats["q50"],
        "final_gate_q75": gate_stats["q75"],
        "zero_or_nearzero_gate_ratio": nearzero_ratio,
    }


def _detached_stats(values):
    detached = values.detach()
    if detached.numel() == 0:
        return {"mean": 0.0, "median": 0.0, "q10": 0.0, "q90": 0.0}
    detached = torch.nan_to_num(detached.float(), nan=0.0, posinf=1.0, neginf=0.0)
    return {
        "mean": float(detached.mean().item()),
        "median": float(detached.quantile(0.5).item()),
        "q10": float(detached.quantile(0.1).item()),
        "q90": float(detached.quantile(0.9).item()),
    }


def _detached_extended_stats(values):
    detached = values.detach()
    if detached.numel() == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "q25": 0.0, "q50": 0.0, "q75": 0.0}
    detached = torch.nan_to_num(detached.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return {
        "mean": float(detached.mean().item()),
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
        "q25": float(detached.quantile(0.25).item()),
        "q50": float(detached.quantile(0.5).item()),
        "q75": float(detached.quantile(0.75).item()),
    }


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


def _validate_edge_weights(edge_weights, edge_count):
    if not torch.is_tensor(edge_weights) or edge_weights.ndim != 1 or edge_weights.shape != (edge_count,):
        raise ValueError("edge_weights must be a Tensor[E].")
    if not torch.isfinite(edge_weights).all():
        raise ValueError("edge_weights must be finite.")
    if edge_weights.numel() > 0 and ((edge_weights < 0.0).any() or (edge_weights > 1.0).any()):
        raise ValueError("edge_weights must be in [0, 1].")


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
