import math

import pytest
import torch

from utils.structural_graph import (
    build_knn_graph,
    compute_continuity_defect,
    compute_geometric_turning,
    compute_redundancy,
    estimate_gaussian_normals,
)


def _grid_plane(width=4, spacing=1.0, z=0.0):
    xs, ys = torch.meshgrid(
        torch.arange(width, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack((xs.reshape(-1) * spacing, ys.reshape(-1) * spacing, torch.full((width * width,), z)), dim=-1)


def _manual_knn(xyz, k):
    neighbors = []
    for i in range(xyz.shape[0]):
        dist = torch.linalg.norm(xyz - xyz[i], dim=-1)
        order = torch.argsort(dist)
        order = order[order != i]
        neighbors.append(order[: min(k, xyz.shape[0] - 1)])
    return torch.stack(neighbors, dim=0)


def test_knn_graph_shape_dtype_no_self_and_small_n():
    pytest.importorskip("scipy")
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    neighbors = build_knn_graph(xyz, k=5)

    assert neighbors.shape == (3, 2)
    assert neighbors.dtype == torch.long
    assert torch.isfinite(neighbors.to(torch.float32)).all()
    for i in range(xyz.shape[0]):
        assert i not in neighbors[i].tolist()
        assert set(neighbors[i].tolist()) == set(range(xyz.shape[0])) - {i}


def test_gaussian_normals_follow_minimum_scale_axis():
    scales = torch.tensor(
        [
            [1.0, 2.0, 0.25],
            [0.2, 2.0, 3.0],
        ],
        dtype=torch.float32,
    )
    angle = math.pi / 2.0
    rotations = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)],
        ],
        dtype=torch.float32,
    )

    normals = estimate_gaussian_normals(scales, rotations)

    assert normals.shape == (2, 3)
    assert torch.isfinite(normals).all()
    assert torch.allclose(torch.linalg.norm(normals, dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(normals[0].abs(), torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)
    assert torch.allclose(normals[1].abs(), torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)


def test_plane_has_low_turning_and_low_continuity_defect():
    xyz = _grid_plane(width=4)
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).repeat(xyz.shape[0], 1)
    neighbors = _manual_knn(xyz, k=4)

    k_values = compute_geometric_turning(xyz, normals, neighbors)
    d_values = compute_continuity_defect(xyz, normals, neighbors)

    assert torch.isfinite(k_values).all()
    assert torch.isfinite(d_values).all()
    assert k_values.mean() < 1e-4
    assert d_values.mean() < 1e-4


def test_right_angle_corner_has_higher_turning_without_huge_defect():
    plane_xy = _grid_plane(width=4)
    plane_yz = torch.stack((torch.zeros_like(plane_xy[:, 0]), plane_xy[:, 0], plane_xy[:, 1]), dim=-1)
    xyz = torch.cat((plane_xy, plane_yz), dim=0)
    normals = torch.cat(
        (
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).repeat(plane_xy.shape[0], 1),
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32).repeat(plane_yz.shape[0], 1),
        ),
        dim=0,
    )
    neighbors = _manual_knn(xyz, k=8)

    k_values = compute_geometric_turning(xyz, normals, neighbors)
    d_values = compute_continuity_defect(xyz, normals, neighbors, normal_threshold=0.9)
    corner_mask = (xyz[:, 0].abs() < 1e-6) & (xyz[:, 2].abs() < 1e-6)
    planar_mask = (xyz[:, 0] > 1.5) | (xyz[:, 2] > 1.5)

    assert torch.isfinite(k_values).all()
    assert torch.isfinite(d_values).all()
    assert k_values[corner_mask].mean() > k_values[planar_mask].mean()
    assert d_values[corner_mask].mean() < 0.5


def test_double_layer_plane_has_higher_continuity_defect_than_single_plane():
    single = _grid_plane(width=4)
    double = torch.cat((single, _grid_plane(width=4, z=0.35)), dim=0)
    single_normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).repeat(single.shape[0], 1)
    double_normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).repeat(double.shape[0], 1)

    single_d = compute_continuity_defect(single, single_normals, _manual_knn(single, k=4))
    double_d = compute_continuity_defect(double, double_normals, _manual_knn(double, k=6))

    assert torch.isfinite(single_d).all()
    assert torch.isfinite(double_d).all()
    assert double_d.mean() > single_d.mean() + 0.05


def test_redundancy_is_higher_for_close_large_overlapping_gaussians():
    dense = torch.tensor(
        [[0.00, 0.0, 0.0], [0.05, 0.0, 0.0], [0.00, 0.05, 0.0], [0.05, 0.05, 0.0]],
        dtype=torch.float32,
    )
    sparse = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
        dtype=torch.float32,
    )
    dense_scales = torch.full((dense.shape[0], 3), 0.4)
    sparse_scales = torch.full((sparse.shape[0], 3), 0.05)

    dense_r = compute_redundancy(dense, dense_scales, _manual_knn(dense, k=3))
    sparse_r = compute_redundancy(sparse, sparse_scales, _manual_knn(sparse, k=3))

    assert torch.isfinite(dense_r).all()
    assert torch.isfinite(sparse_r).all()
    assert dense_r.mean() > sparse_r.mean()
