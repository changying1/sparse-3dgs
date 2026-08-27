from pathlib import Path

import pytest

import numpy as np

from scene.dataset_readers import (
    CameraInfo,
    SceneInfo,
    apply_sparse_view_split,
    getNerfppNorm,
    load_sparse_init_point_cloud,
    storePly,
)
from utils.graphics_utils import BasicPointCloud
from scripts.create_sparse_view_split import nested_sparse_positions, uniform_indices


def _camera(name):
    return CameraInfo(
        uid=0,
        R=None,
        T=None,
        FovY=0.0,
        FovX=0.0,
        depth_params=None,
        image_path=name,
        image_name=name,
        depth_path="",
        width=1,
        height=1,
        is_test=False,
    )


def _posed_camera(name, x):
    return CameraInfo(
        uid=0,
        R=np.eye(3),
        T=np.array([-x, 0.0, 0.0]),
        FovY=0.0,
        FovX=0.0,
        depth_params=None,
        image_path=name,
        image_name=name,
        depth_path="",
        width=1,
        height=1,
        is_test=False,
    )


def _scene_info(count=12):
    cameras = [_camera(f"image_{index:03d}.png") for index in range(count)]
    return SceneInfo(
        point_cloud=None,
        train_cameras=cameras,
        test_cameras=[],
        nerf_normalization={"radius": 1.0},
        ply_path="points3D.ply",
        is_nerf_synthetic=False,
    )


def _point_cloud(count):
    xyz = np.stack(
        [
            np.arange(count, dtype=np.float32),
            np.zeros(count, dtype=np.float32),
            np.ones(count, dtype=np.float32),
        ],
        axis=1,
    )
    return BasicPointCloud(
        points=xyz,
        colors=np.zeros((count, 3), dtype=np.float32),
        normals=np.zeros((count, 3), dtype=np.float32),
    )


def _write_sparse_init(path, count):
    path.mkdir(parents=True)
    xyz = _point_cloud(count).points
    rgb = np.zeros((count, 3), dtype=np.uint8)
    storePly(str(path / "points3D.ply"), xyz, rgb)


def _write_names(path, names):
    Path(path).write_text("\n".join(names) + "\n")


def _names(cameras):
    return [camera.image_name for camera in cameras]


def test_sparse_split_loads_exactly_three_train_cameras(tmp_path):
    train_path = tmp_path / "train_views_3.txt"
    _write_names(train_path, ["image_000.png", "image_005.png", "image_011.png"])

    split = apply_sparse_view_split(_scene_info(), str(train_path), "")

    assert _names(split.train_cameras) == ["image_000.png", "image_005.png", "image_011.png"]
    assert len(split.train_cameras) == 3
    assert len(split.test_cameras) == 9


def test_sparse_split_loads_exactly_six_train_cameras(tmp_path):
    train_names = [f"image_{index:03d}.png" for index in [0, 2, 4, 7, 9, 11]]
    train_path = tmp_path / "train_views_6.txt"
    _write_names(train_path, train_names)

    split = apply_sparse_view_split(_scene_info(), str(train_path), "")

    assert _names(split.train_cameras) == train_names
    assert len(split.train_cameras) == 6


def test_sparse_split_loads_exactly_nine_train_cameras(tmp_path):
    train_names = [f"image_{index:03d}.png" for index in [0, 1, 3, 4, 6, 7, 8, 10, 11]]
    train_path = tmp_path / "train_views_9.txt"
    _write_names(train_path, train_names)

    split = apply_sparse_view_split(_scene_info(), str(train_path), "")

    assert _names(split.train_cameras) == train_names
    assert len(split.train_cameras) == 9


def test_sparse_split_rejects_train_test_overlap(tmp_path):
    train_path = tmp_path / "train_views.txt"
    test_path = tmp_path / "test_views.txt"
    _write_names(train_path, ["image_000.png"])
    _write_names(test_path, ["image_000.png"])

    with pytest.raises(ValueError, match="overlap"):
        apply_sparse_view_split(_scene_info(), str(train_path), str(test_path))


def test_sparse_split_repeated_load_is_deterministic(tmp_path):
    train_names = ["image_000.png", "image_005.png", "image_011.png"]
    train_path = tmp_path / "train_views.txt"
    _write_names(train_path, train_names)

    first = apply_sparse_view_split(_scene_info(), str(train_path), "")
    second = apply_sparse_view_split(_scene_info(), str(train_path), "")

    assert _names(first.train_cameras) == _names(second.train_cameras)
    assert _names(first.test_cameras) == _names(second.test_cameras)


def test_sparse_split_unspecified_keeps_full_view_behavior():
    scene_info = _scene_info()

    split = apply_sparse_view_split(scene_info, "", "")

    assert split is scene_info
    assert len(split.train_cameras) == 12
    assert len(split.test_cameras) == 0


def test_sparse_init_path_unspecified_keeps_full_point_cloud():
    scene_info = _scene_info()
    scene_info = scene_info._replace(point_cloud=_point_cloud(149606), ply_path="full/points3D.ply")

    loaded = load_sparse_init_point_cloud(scene_info, "")

    assert loaded is scene_info
    assert len(loaded.point_cloud.points) == 149606
    assert loaded.ply_path == "full/points3D.ply"


def test_sparse_init_path_overrides_only_point_cloud(tmp_path):
    sparse_init = tmp_path / "sparse_3" / "0"
    _write_sparse_init(sparse_init, 57)
    scene_info = _scene_info()._replace(point_cloud=_point_cloud(149606), ply_path="full/points3D.ply")

    loaded = load_sparse_init_point_cloud(scene_info, str(sparse_init))

    assert len(loaded.point_cloud.points) == 57
    assert loaded.ply_path == str(sparse_init / "points3D.ply")
    assert _names(loaded.train_cameras) == _names(scene_info.train_cameras)
    assert _names(loaded.test_cameras) == _names(scene_info.test_cameras)


@pytest.mark.parametrize("count", [57, 185, 270])
def test_sparse_init_counts_match_frozen_protocol(tmp_path, count):
    sparse_init = tmp_path / f"sparse_{count}" / "0"
    _write_sparse_init(sparse_init, count)

    loaded = load_sparse_init_point_cloud(_scene_info(), str(sparse_init))

    assert len(loaded.point_cloud.points) == count


def test_sparse_init_missing_directory_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="sparse_init_model_path"):
        load_sparse_init_point_cloud(_scene_info(), str(tmp_path / "missing" / "0"))


def test_sparse_init_missing_points_file_fails(tmp_path):
    sparse_init = tmp_path / "sparse_3" / "0"
    sparse_init.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="points3D.ply"):
        load_sparse_init_point_cloud(_scene_info(), str(sparse_init))


def test_sparse_init_empty_points_fail(tmp_path):
    sparse_init = tmp_path / "sparse_3" / "0"
    _write_sparse_init(sparse_init, 0)

    with pytest.raises(ValueError, match="no vertices"):
        load_sparse_init_point_cloud(_scene_info(), str(sparse_init))


def test_sparse_init_non_finite_xyz_fails(tmp_path):
    sparse_init = tmp_path / "sparse_3" / "0"
    sparse_init.mkdir(parents=True)
    xyz = np.array([[0.0, np.nan, 1.0]], dtype=np.float32)
    rgb = np.zeros((1, 3), dtype=np.uint8)
    storePly(str(sparse_init / "points3D.ply"), xyz, rgb)

    with pytest.raises(ValueError, match="non-finite"):
        load_sparse_init_point_cloud(_scene_info(), str(sparse_init))


def test_sparse_split_normalization_uses_split_train_cameras(tmp_path):
    train_path = tmp_path / "train_views.txt"
    _write_names(train_path, ["image_000.png", "image_010.png"])
    cameras = [_posed_camera("image_000.png", 0.0), _posed_camera("image_010.png", 10.0), _posed_camera("image_100.png", 100.0)]
    scene_info = SceneInfo(
        point_cloud=None,
        train_cameras=cameras,
        test_cameras=[],
        nerf_normalization=getNerfppNorm(cameras),
        ply_path="points3D.ply",
        is_nerf_synthetic=False,
    )

    split = apply_sparse_view_split(scene_info, str(train_path), "")
    split_norm = getNerfppNorm(split.train_cameras)

    assert split_norm["radius"] == pytest.approx(5.5)
    assert scene_info.nerf_normalization["radius"] == pytest.approx(69.666664)


def test_sparse_split_rejects_unknown_camera_name(tmp_path):
    train_path = tmp_path / "train_views.txt"
    _write_names(train_path, ["missing.png"])

    with pytest.raises(ValueError, match="unknown camera"):
        apply_sparse_view_split(_scene_info(), str(train_path), "")


def test_sparse_split_rejects_duplicate_camera_name(tmp_path):
    train_path = tmp_path / "train_views.txt"
    _write_names(train_path, ["image_000.png", "image_000.png"])

    with pytest.raises(ValueError, match="Duplicate camera name"):
        apply_sparse_view_split(_scene_info(), str(train_path), "")


def test_sparse_split_train_and_test_are_disjoint_when_test_is_implicit(tmp_path):
    train_names = ["image_000.png", "image_005.png", "image_011.png"]
    train_path = tmp_path / "train_views.txt"
    _write_names(train_path, train_names)

    split = apply_sparse_view_split(_scene_info(), str(train_path), "")

    assert set(_names(split.train_cameras)).isdisjoint(_names(split.test_cameras))


def test_uniform_indices_are_deterministic_and_nested():
    nine = uniform_indices(251, 9)
    six = [nine[index] for index in nested_sparse_positions(9, 6)]
    three = [nine[index] for index in nested_sparse_positions(9, 3)]

    assert nine == [0, 31, 62, 94, 125, 156, 188, 219, 250]
    assert six == [0, 62, 125, 156, 188, 250]
    assert three == [0, 125, 250]
    assert set(three).issubset(six)
    assert set(six).issubset(nine)
