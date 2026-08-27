import argparse
import importlib.util
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_COLMAP_LOADER_SPEC = importlib.util.spec_from_file_location(
    "colmap_loader", REPO_ROOT / "scene" / "colmap_loader.py"
)
_COLMAP_LOADER = importlib.util.module_from_spec(_COLMAP_LOADER_SPEC)
_COLMAP_LOADER_SPEC.loader.exec_module(_COLMAP_LOADER)
read_extrinsics_binary = _COLMAP_LOADER.read_extrinsics_binary
read_intrinsics_binary = _COLMAP_LOADER.read_intrinsics_binary
read_points3D_binary = _COLMAP_LOADER.read_points3D_binary


COLMAP_PAIR_ID_BASE = 2147483647


def main():
    parser = argparse.ArgumentParser(description="Create split-specific fixed-pose COLMAP initialization.")
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--train_views_file", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--colmap_path", required=True)
    args = parser.parse_args()

    report = create_split_specific_initialization(
        source_path=Path(args.source_path),
        train_views_file=Path(args.train_views_file),
        output_model_path=Path(args.output_model_path),
        colmap_path=Path(args.colmap_path),
    )
    print(json.dumps(report, indent=2))


def create_split_specific_initialization(source_path, train_views_file, output_model_path, colmap_path):
    source_path = source_path.resolve()
    output_model_path = output_model_path.resolve()
    colmap_path = colmap_path.resolve()
    full_sparse_path = source_path / "sparse" / "0"
    full_database_path = source_path / "database.db"
    images_path = source_path / "images"

    _validate_paths(source_path, full_sparse_path, full_database_path, images_path, output_model_path, colmap_path)
    train_names = _read_train_names(train_views_file)
    level_dir = output_model_path.parent
    input_model_path = level_dir / "input_model"
    filtered_database_path = level_dir / "database.db"
    log_path = level_dir / "triangulation.log"
    metadata_path = level_dir / "initialization_metadata.json"

    if level_dir.exists():
        shutil.rmtree(level_dir)
    input_model_path.mkdir(parents=True)
    output_model_path.mkdir(parents=True)

    cameras = read_intrinsics_binary(str(full_sparse_path / "cameras.bin"))
    images = read_extrinsics_binary(str(full_sparse_path / "images.bin"))
    images_by_name = {image.name: image for image in images.values()}
    missing = [name for name in train_names if name not in images_by_name]
    if missing:
        raise ValueError("Train views missing from full COLMAP model: " + ", ".join(missing))

    selected_images = [images_by_name[name] for name in train_names]
    selected_camera_ids = sorted({image.camera_id for image in selected_images})
    selected_cameras = {camera_id: cameras[camera_id] for camera_id in selected_camera_ids}
    selected_image_ids = {image.id for image in selected_images}

    _write_text_input_model(input_model_path, selected_cameras, selected_images)
    db_stats = _write_filtered_database(full_database_path, filtered_database_path, selected_image_ids, selected_camera_ids)

    command = [
        str(colmap_path),
        "point_triangulator",
        "--database_path",
        str(filtered_database_path),
        "--image_path",
        str(images_path),
        "--input_path",
        str(input_model_path),
        "--output_path",
        str(output_model_path),
        "--clear_points",
        "1",
        "--refine_intrinsics",
        "0",
        "--Mapper.fix_existing_frames",
        "1",
    ]
    run_result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(run_result.stdout)
    if run_result.returncode != 0:
        raise RuntimeError(f"COLMAP point_triangulator failed with exit code {run_result.returncode}. See {log_path}.")

    _ensure_ply(output_model_path)
    validation = validate_model(output_model_path, train_names, selected_image_ids)

    metadata = {
        "view_count": len(train_names),
        "train_images": train_names,
        "registered_images": validation["registered_images"],
        "initial_points": validation["points3D_bin"],
        "camera_count": validation["camera_count"],
        "heldout_track_references": validation["heldout_track_references"],
        "database_source": str(full_database_path),
        "filtered_database": str(filtered_database_path),
        "poses_source": "full COLMAP model, train images only",
        "triangulation": "train-only fixed-pose",
        "refine_intrinsics": False,
        "colmap_version": _colmap_version(colmap_path),
        "command_line": command,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "original_image_ids": {image.name: image.id for image in selected_images},
        "database_stats": db_stats,
        "validation": validation,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def _validate_paths(source_path, full_sparse_path, full_database_path, images_path, output_model_path, colmap_path):
    if not full_sparse_path.exists():
        raise FileNotFoundError(full_sparse_path)
    if not full_database_path.exists():
        raise FileNotFoundError(full_database_path)
    if not images_path.exists():
        raise FileNotFoundError(images_path)
    if not colmap_path.exists():
        raise FileNotFoundError(colmap_path)
    forbidden = (source_path / "sparse" / "0").resolve()
    if output_model_path == forbidden or forbidden in output_model_path.parents:
        raise ValueError("Refusing to write into the full-view sparse/0 model.")


def _read_train_names(path):
    names = []
    seen = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        if name in seen:
            raise ValueError(f"Duplicate train view '{name}' in {path}:{line_number}.")
        seen.add(name)
        names.append(name)
    if not names:
        raise ValueError("train_views_file is empty.")
    return names


def _write_text_input_model(path, cameras, images):
    with open(path / "cameras.txt", "w") as file:
        for camera_id in sorted(cameras):
            camera = cameras[camera_id]
            params = " ".join(_format_float(value) for value in camera.params)
            file.write(f"{camera.id} {camera.model} {camera.width} {camera.height} {params}\n")
    with open(path / "images.txt", "w") as file:
        for image in sorted(images, key=lambda item: item.id):
            qvec = " ".join(_format_float(value) for value in image.qvec)
            tvec = " ".join(_format_float(value) for value in image.tvec)
            file.write(f"{image.id} {qvec} {tvec} {image.camera_id} {image.name}\n\n")
    (path / "points3D.txt").write_text("")


def _write_filtered_database(source_db, output_db, selected_image_ids, selected_camera_ids):
    shutil.copy2(source_db, output_db)
    con = sqlite3.connect(output_db)
    cur = con.cursor()
    image_ids = tuple(sorted(selected_image_ids))
    camera_ids = tuple(sorted(selected_camera_ids))
    pair_ids = tuple(sorted(_pair_id(a, b) for index, a in enumerate(image_ids) for b in image_ids[index + 1:]))
    _delete_not_in(cur, "keypoints", "image_id", image_ids)
    _delete_not_in(cur, "descriptors", "image_id", image_ids)
    _delete_not_in(cur, "images", "image_id", image_ids)
    _delete_not_in(cur, "cameras", "camera_id", camera_ids)
    _delete_not_in(cur, "matches", "pair_id", pair_ids)
    _delete_not_in(cur, "two_view_geometries", "pair_id", pair_ids)
    _delete_not_in_optional(cur, "frame_data", "data_id", image_ids)
    con.commit()
    stats = {
        "images": cur.execute("select count(*) from images").fetchone()[0],
        "cameras": cur.execute("select count(*) from cameras").fetchone()[0],
        "keypoints": cur.execute("select count(*) from keypoints").fetchone()[0],
        "descriptors": cur.execute("select count(*) from descriptors").fetchone()[0],
        "matches": cur.execute("select count(*) from matches").fetchone()[0],
        "two_view_geometries": cur.execute("select count(*) from two_view_geometries").fetchone()[0],
        "nonempty_matches": cur.execute("select count(*) from matches where rows > 0").fetchone()[0],
        "nonempty_two_view_geometries": cur.execute("select count(*) from two_view_geometries where rows > 0").fetchone()[0],
    }
    con.close()
    return stats


def _delete_not_in(cur, table, column, values):
    placeholders = ",".join("?" for _ in values)
    cur.execute(f"delete from {table} where {column} not in ({placeholders})", values)


def _delete_not_in_optional(cur, table, column, values):
    exists = cur.execute("select count(*) from sqlite_master where type='table' and name=?", (table,)).fetchone()[0]
    if exists:
        _delete_not_in(cur, table, column, values)


def _pair_id(image_id1, image_id2):
    low, high = sorted((int(image_id1), int(image_id2)))
    return COLMAP_PAIR_ID_BASE * low + high


def _ensure_ply(model_path):
    bin_path = model_path / "points3D.bin"
    ply_path = model_path / "points3D.ply"
    xyz, rgb, _ = read_points3D_binary(str(bin_path))
    _write_ascii_ply(ply_path, xyz, rgb)


def validate_model(model_path, train_names, train_image_ids):
    cameras = read_intrinsics_binary(str(model_path / "cameras.bin"))
    images = read_extrinsics_binary(str(model_path / "images.bin"))
    xyz, rgb, errors = _read_points3d_with_tracks(model_path / "points3D.bin")
    ply_vertex_count = _read_ply_vertex_count(model_path / "points3D.ply")
    registered_names = [image.name for image in sorted(images.values(), key=lambda item: item.name)]
    heldout_refs = sum(
        1
        for point in errors["tracks"]
        for image_id in point
        if image_id not in train_image_ids
    )
    finite_xyz = bool(np.isfinite(xyz).all()) if xyz.size else True
    finite_error = bool(np.isfinite(errors["values"]).all()) if errors["values"].size else True
    rgb_valid = bool(((rgb >= 0) & (rgb <= 255)).all()) if rgb.size else True
    return {
        "registered_images": len(images),
        "camera_count": len(cameras),
        "registered_names": registered_names,
        "matches_train_file": registered_names == sorted(train_names),
        "points3D_bin": int(xyz.shape[0]),
        "points3D_ply": int(ply_vertex_count),
        "ply_matches_bin": int(ply_vertex_count) == int(xyz.shape[0]),
        "heldout_track_references": heldout_refs,
        "xyz_finite": finite_xyz,
        "rgb_valid": rgb_valid,
        "error_finite": finite_error,
    }


def _write_ascii_ply(path, xyz, rgb):
    with open(path, "w", encoding="ascii") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {xyz.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property float nx\n")
        file.write("property float ny\n")
        file.write("property float nz\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")
        normals = np.zeros_like(xyz)
        for point, normal, color in zip(xyz, normals, rgb):
            file.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{normal[0]:.8f} {normal[1]:.8f} {normal[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _read_ply_vertex_count(path):
    with open(path, "r", encoding="ascii") as file:
        for line in file:
            if line.startswith("element vertex "):
                return int(line.split()[2])
            if line.strip() == "end_header":
                break
    raise ValueError(f"Could not find vertex count in {path}.")


def _read_points3d_with_tracks(path):
    import struct

    xyzs = []
    rgbs = []
    errors = []
    tracks = []
    with open(path, "rb") as file:
        num_points = struct.unpack("<Q", file.read(8))[0]
        for _ in range(num_points):
            point = struct.unpack("<QdddBBBd", file.read(43))
            xyzs.append(point[1:4])
            rgbs.append(point[4:7])
            errors.append(point[7])
            track_length = struct.unpack("<Q", file.read(8))[0]
            track_elems = struct.unpack("<" + "ii" * track_length, file.read(8 * track_length)) if track_length else ()
            tracks.append(list(track_elems[0::2]))
    return (
        np.array(xyzs, dtype=np.float64).reshape((num_points, 3)),
        np.array(rgbs, dtype=np.uint8).reshape((num_points, 3)),
        {"values": np.array(errors, dtype=np.float64), "tracks": tracks},
    )


def _colmap_version(colmap_path):
    result = subprocess.run([str(colmap_path), "-h"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    first_line = result.stdout.splitlines()[0] if result.stdout else ""
    return first_line.strip()


def _format_float(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Cannot write non-finite camera parameter.")
    return f"{value:.17g}"


if __name__ == "__main__":
    main()
