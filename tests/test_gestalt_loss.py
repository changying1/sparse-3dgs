import math

import pytest
import torch

from utils.gestalt_loss import (
    compute_gestalt_loss,
    compute_normal_continuity_loss,
    compute_plane_continuity_loss,
    select_same_surface_edges,
)


def _grid_plane(width=3, spacing=1.0, z=0.0):
    xs, ys = torch.meshgrid(
        torch.arange(width, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack((xs.reshape(-1) * spacing, ys.reshape(-1) * spacing, torch.full((width * width,), z)), dim=-1)


def _manual_knn(xyz, k):
    if xyz.shape[0] == 0:
        return torch.empty((0, 0), dtype=torch.long)
    effective_k = min(k, max(xyz.shape[0] - 1, 0))
    if effective_k == 0:
        return torch.empty((xyz.shape[0], 0), dtype=torch.long)
    neighbors = []
    for i in range(xyz.shape[0]):
        dist = torch.linalg.norm(xyz - xyz[i], dim=-1)
        order = torch.argsort(dist)
        order = order[order != i]
        neighbors.append(order[:effective_k])
    return torch.stack(neighbors, dim=0)


def _z_normals(n_points):
    return torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).repeat(n_points, 1)


def test_planar_geometry_has_near_zero_plane_loss():
    xyz = _grid_plane(width=3)
    normals = _z_normals(xyz.shape[0])
    edges = select_same_surface_edges(xyz, normals, _manual_knn(xyz, k=4))

    loss = compute_plane_continuity_loss(xyz, normals, edges)

    assert torch.isfinite(loss)
    assert loss.item() < 1e-8


def test_displaced_double_layer_plane_has_larger_plane_loss():
    single = _grid_plane(width=3)
    double = torch.cat((single, _grid_plane(width=3, z=0.35)), dim=0)
    single_normals = _z_normals(single.shape[0])
    double_normals = _z_normals(double.shape[0])

    single_edges = select_same_surface_edges(single, single_normals, _manual_knn(single, k=4))
    double_edges = select_same_surface_edges(double, double_normals, _manual_knn(double, k=6))
    single_loss = compute_plane_continuity_loss(single, single_normals, single_edges)
    double_loss = compute_plane_continuity_loss(double, double_normals, double_edges)

    assert torch.isfinite(double_loss)
    assert double_loss > single_loss + 0.01


def test_aligned_normals_have_near_zero_normal_loss():
    normals = _z_normals(4)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)

    loss = compute_normal_continuity_loss(normals, edges)

    assert torch.isfinite(loss)
    assert loss.item() < 1e-8


def test_perturbed_normals_produce_larger_normal_loss():
    aligned = _z_normals(2)
    perturbed = torch.tensor([[0.0, 0.0, 1.0], [math.sqrt(0.5), 0.0, math.sqrt(0.5)]], dtype=torch.float32)
    edges = torch.tensor([[0, 1]], dtype=torch.long)

    aligned_loss = compute_normal_continuity_loss(aligned, edges)
    perturbed_loss = compute_normal_continuity_loss(perturbed, edges)

    assert torch.isfinite(perturbed_loss)
    assert perturbed_loss > aligned_loss + 0.1


def test_normal_sign_ambiguity_is_abs_dot_invariant():
    normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=torch.float32)
    edges = torch.tensor([[0, 1]], dtype=torch.long)

    loss = compute_normal_continuity_loss(normals, edges)

    assert torch.isfinite(loss)
    assert loss.item() < 1e-8


def test_right_angle_corner_filtering_excludes_cross_plane_edges():
    xy = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    yz = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=torch.float32)
    xyz = torch.cat((xy, yz), dim=0)
    normals = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    neighbors = torch.tensor([[1, 2], [0, 2], [3, 0], [2, 1]], dtype=torch.long)

    edges = select_same_surface_edges(xyz, normals, neighbors, normal_threshold=0.9)
    selected = {tuple(edge) for edge in edges.tolist()}

    assert (0, 2) not in selected
    assert (1, 2) not in selected
    assert (2, 0) not in selected
    assert (3, 1) not in selected
    assert selected == {(0, 1), (1, 0), (2, 3), (3, 2)}


def test_visibility_evidence_gate_filters_unobserved_gaussians():
    xyz = _grid_plane(width=2)
    normals = _z_normals(xyz.shape[0])
    neighbors = _manual_knn(xyz, k=3)
    visible_view_count = torch.tensor([1, 0, 1, 1], dtype=torch.long)

    edges = select_same_surface_edges(xyz, normals, neighbors, visible_view_count=visible_view_count)

    assert edges.numel() > 0
    assert 1 not in edges.reshape(-1).tolist()


def test_invalid_visible_view_count_shape_is_rejected():
    xyz = _grid_plane(width=2)
    normals = _z_normals(xyz.shape[0])
    neighbors = _manual_knn(xyz, k=3)
    visible_view_count = torch.ones((xyz.shape[0], 1), dtype=torch.long)

    with pytest.raises(ValueError, match="visible_view_count must be a Tensor\\[N\\]"):
        select_same_surface_edges(xyz, normals, neighbors, visible_view_count=visible_view_count)


def test_max_distance_filters_distant_pairs_only_when_provided():
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    normals = _z_normals(xyz.shape[0])
    neighbors = torch.tensor([[1, 2], [0, 2], [0, 1]], dtype=torch.long)

    unfiltered = select_same_surface_edges(xyz, normals, neighbors, max_distance=None)
    filtered = select_same_surface_edges(xyz, normals, neighbors, max_distance=0.5)
    unfiltered_pairs = {tuple(edge) for edge in unfiltered.tolist()}
    filtered_pairs = {tuple(edge) for edge in filtered.tolist()}

    assert (0, 1) in filtered_pairs
    assert (1, 0) in filtered_pairs
    assert (0, 2) in unfiltered_pairs
    assert (2, 0) in unfiltered_pairs
    assert (0, 2) not in filtered_pairs
    assert (2, 0) not in filtered_pairs


def test_zero_valid_edges_returns_zero_finite_losses():
    xyz = _grid_plane(width=2)
    normals = _z_normals(xyz.shape[0])
    visible_view_count = torch.zeros((xyz.shape[0],), dtype=torch.long)
    edges = select_same_surface_edges(
        xyz,
        normals,
        _manual_knn(xyz, k=3),
        visible_view_count=visible_view_count,
        normal_threshold=0.99,
    )
    plane_loss = compute_plane_continuity_loss(xyz, normals, edges)
    normal_loss = compute_normal_continuity_loss(normals, edges)
    total = plane_loss + normal_loss

    assert edges.shape == (0, 2)
    assert plane_loss.item() == 0.0
    assert normal_loss.item() == 0.0
    assert total.item() == 0.0
    assert torch.isfinite(total)


def test_single_gaussian_and_empty_neighbor_graph_return_zero():
    xyz = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    scales = torch.tensor([[1.0, 1.0, 0.1]], dtype=torch.float32)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    neighbors = torch.empty((1, 0), dtype=torch.long)

    result = compute_gestalt_loss(xyz, scales, rotations, neighbors)

    assert result["edge_count"] == 0
    assert result["plane_loss"].item() == 0.0
    assert result["normal_loss"].item() == 0.0
    assert result["loss"].item() == 0.0


def test_backward_preserves_xyz_and_rotation_gradients():
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.25], [0.0, 1.0, -0.15]],
        dtype=torch.float32,
        requires_grad=True,
    )
    scales = torch.tensor([[1.0, 1.0, 0.1], [1.0, 1.0, 0.1], [1.0, 1.0, 0.1]], dtype=torch.float32)
    rotations = torch.tensor(
        [[0.98, 0.04, 0.10, 0.02], [0.99, -0.02, 0.08, 0.03], [0.97, 0.03, 0.12, -0.01]],
        dtype=torch.float32,
        requires_grad=True,
    )
    neighbors = _manual_knn(xyz.detach(), k=2)

    result = compute_gestalt_loss(xyz, scales, rotations, neighbors, normal_threshold=0.8)
    result["loss"].backward()

    assert result["loss"].item() > 0.0
    assert xyz.grad is not None
    assert rotations.grad is not None
    assert torch.isfinite(xyz.grad).all()
    assert torch.isfinite(rotations.grad).all()


def test_extreme_small_scale_numerical_stability():
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1e-6, 0.0, 1e-7], [0.0, 1e-6, -1e-7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    scales = torch.full((3, 3), 1e-9, dtype=torch.float64)
    scales[:, 2] = 1e-12
    rotations = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.999, 0.001, 0.002, 0.0], [0.998, -0.002, 0.001, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    neighbors = _manual_knn(xyz.detach().to(torch.float32), k=2)

    result = compute_gestalt_loss(xyz, scales, rotations, neighbors, normal_threshold=0.9)
    result["loss"].backward()

    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["plane_loss"])
    assert torch.isfinite(result["normal_loss"])
    assert torch.isfinite(xyz.grad).all()
    assert torch.isfinite(rotations.grad).all()
