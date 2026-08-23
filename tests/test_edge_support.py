import pytest
import torch

from utils.edge_support import (
    aggregate_multiview_edge_support,
    camera_from_projection,
    compute_edge_map,
    project_gaussians_to_camera,
    sample_edge_support,
)


def _identity_projection_camera(width=8, height=6):
    full_proj = torch.eye(4, dtype=torch.float32)
    world_view = torch.eye(4, dtype=torch.float32)
    return camera_from_projection(full_proj, width, height, world_view_transform=world_view, uid=10)


def test_constant_image_has_near_zero_edges():
    image = torch.ones((3, 16, 16), dtype=torch.float32) * 0.5

    edge = compute_edge_map(image)

    assert edge.shape == (16, 16)
    assert torch.isfinite(edge).all()
    assert edge.mean() < 1e-3


def test_vertical_step_edge_has_high_center_response():
    image = torch.zeros((3, 16, 16), dtype=torch.float32)
    image[:, :, 8:] = 1.0

    edge = compute_edge_map(image)
    center_response = edge[:, 7:9].mean()
    far_response = torch.cat((edge[:, :3].reshape(-1), edge[:, 13:].reshape(-1))).mean()

    assert torch.isfinite(edge).all()
    assert center_response > 0.5
    assert far_response < 0.1


def test_projection_center_outside_and_behind_points():
    camera = _identity_projection_camera(width=8, height=6)
    xyz = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
    )

    pixel_coords, valid = project_gaussians_to_camera(xyz, camera=camera)

    assert torch.isfinite(pixel_coords).all()
    assert valid.tolist() == [True, False, False]
    assert torch.allclose(pixel_coords[0], torch.tensor([4.0, 3.0]), atol=1e-5)


def test_projection_y_direction_matches_rasterizer_ndc2pix():
    camera = _identity_projection_camera(width=8, height=6)
    xyz = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, -0.5, 1.0],
        ],
        dtype=torch.float32,
    )

    pixel_coords, valid = project_gaussians_to_camera(xyz, camera=camera)

    assert valid.tolist() == [True, True, True]
    assert pixel_coords[1, 1] > pixel_coords[0, 1]
    assert pixel_coords[2, 1] < pixel_coords[0, 1]


def test_bilinear_sampling_integer_and_half_pixel_positions():
    edge_map = torch.arange(16, dtype=torch.float32).view(4, 4) / 15.0
    pixel_coords = torch.tensor(
        [
            [1.5, 1.5],
            [2.0, 2.0],
        ],
        dtype=torch.float32,
    )
    valid = torch.tensor([True, True])

    support = sample_edge_support(edge_map, pixel_coords, valid)

    expected_center = edge_map[1, 1]
    expected_half = (edge_map[1, 1] + edge_map[1, 2] + edge_map[2, 1] + edge_map[2, 2]) / 4.0
    assert torch.isfinite(support).all()
    assert torch.allclose(support[0], expected_center, atol=1e-6)
    assert torch.allclose(support[1], expected_half, atol=1e-6)


def test_invalid_projection_samples_zero_without_nan():
    edge_map = torch.ones((4, 4), dtype=torch.float32)
    pixel_coords = torch.tensor([[2.0, 2.0], [1.0, 1.0]], dtype=torch.float32)
    valid = torch.tensor([False, True])

    support = sample_edge_support(edge_map, pixel_coords, valid)

    assert torch.isfinite(support).all()
    assert support[0] == 0.0
    assert support[1] > 0.0


def test_multiview_aggregation_uses_visibility_history():
    supports = torch.tensor(
        [
            [0.2, 0.8],
            [1.0, 0.4],
        ],
        dtype=torch.float32,
    )
    valid_masks = torch.ones_like(supports, dtype=torch.bool)
    visibility_history = torch.tensor(
        [
            [True, False],
            [True, True],
        ]
    )

    boundary, counts = aggregate_multiview_edge_support(
        sampled_supports=supports,
        valid_masks=valid_masks,
        visibility_history=visibility_history,
        return_counts=True,
    )

    assert torch.isfinite(boundary).all()
    assert torch.allclose(boundary, torch.tensor([0.2, 0.6]), atol=1e-6)
    assert torch.equal(counts, torch.tensor([1.0, 2.0]))

    cameras = [
        camera_from_projection(torch.eye(4), 4, 4, uid=42),
        camera_from_projection(torch.eye(4), 4, 4, uid=7),
    ]
    uid_visibility = torch.tensor([[False, True]])
    uid_boundary = aggregate_multiview_edge_support(
        sampled_supports=torch.tensor([[0.9], [0.1]], dtype=torch.float32),
        valid_masks=torch.ones((2, 1), dtype=torch.bool),
        visibility_history=uid_visibility,
        cameras=cameras,
        camera_uid_to_train_index={42: 1, 7: 0},
    )
    assert torch.allclose(uid_boundary, torch.tensor([0.9]), atol=1e-6)


def test_zero_visible_views_returns_zero():
    supports = torch.tensor([[0.7], [0.9]], dtype=torch.float32)
    valid_masks = torch.ones_like(supports, dtype=torch.bool)
    visibility_history = torch.tensor([[False, False]])

    boundary, counts = aggregate_multiview_edge_support(
        sampled_supports=supports,
        valid_masks=valid_masks,
        visibility_history=visibility_history,
        return_counts=True,
    )

    assert torch.isfinite(boundary).all()
    assert boundary.item() == 0.0
    assert counts.item() == 0.0


def test_real_camera_path_rejects_edge_map_camera_resolution_mismatch():
    camera = _identity_projection_camera(width=8, height=6)
    xyz = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    edge_map = torch.zeros((5, 8), dtype=torch.float32)

    with pytest.raises(ValueError, match="edge_map size HxW=5x8, camera size HxW=6x8"):
        aggregate_multiview_edge_support(
            xyz=xyz,
            cameras=[camera],
            edge_maps=[edge_map],
        )


def test_real_camera_path_rejects_camera_edge_map_count_mismatch():
    cameras = [_identity_projection_camera(width=8, height=6), _identity_projection_camera(width=8, height=6)]
    xyz = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    edge_map = torch.zeros((6, 8), dtype=torch.float32)

    with pytest.raises(ValueError, match="got 2 cameras and 1 edge maps"):
        aggregate_multiview_edge_support(
            xyz=xyz,
            cameras=cameras,
            edge_maps=[edge_map],
        )


def test_real_camera_path_requires_uid_mapping_with_visibility_history():
    camera = _identity_projection_camera(width=8, height=6)
    xyz = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    edge_map = torch.zeros((6, 8), dtype=torch.float32)
    visibility_history = torch.tensor([[True]])

    with pytest.raises(ValueError, match="camera.uid -> Phase 4 train index mapping"):
        aggregate_multiview_edge_support(
            xyz=xyz,
            cameras=[camera],
            edge_maps=[edge_map],
            visibility_history=visibility_history,
        )
