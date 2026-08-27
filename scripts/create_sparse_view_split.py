import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.colmap_loader import read_extrinsics_binary, read_extrinsics_text


def load_colmap_image_names(source_path):
    sparse_dir = Path(source_path) / "sparse" / "0"
    binary_path = sparse_dir / "images.bin"
    text_path = sparse_dir / "images.txt"
    if binary_path.exists():
        extrinsics = read_extrinsics_binary(str(binary_path))
    elif text_path.exists():
        extrinsics = read_extrinsics_text(str(text_path))
    else:
        raise FileNotFoundError(f"No COLMAP images.bin or images.txt found in {sparse_dir}.")
    return sorted(image.name for image in extrinsics.values())


def uniform_indices(total_count, view_count):
    if view_count <= 0:
        raise ValueError("view_count must be positive.")
    if view_count > total_count:
        raise ValueError(f"Cannot select {view_count} views from {total_count} cameras.")
    return np.linspace(0, total_count - 1, num=view_count).round().astype(int).tolist()


def nested_sparse_positions(max_view_count, view_count):
    if max_view_count == 9 and view_count == 3:
        return [0, 4, 8]
    if max_view_count == 9 and view_count == 6:
        return [0, 2, 4, 5, 6, 8]
    return uniform_indices(max_view_count, view_count)


def create_sparse_view_split(source_path, output_dir=None, view_counts=(3, 6, 9)):
    names = load_colmap_image_names(source_path)
    output = Path(output_dir) if output_dir else Path(source_path) / "sparse_view_splits"
    output.mkdir(parents=True, exist_ok=True)

    max_view_count = max(view_counts)
    selected_by_count = {}
    selected_indices_by_count = {}
    base_indices = uniform_indices(len(names), max_view_count)
    for view_count in sorted(view_counts):
        positions = nested_sparse_positions(max_view_count, view_count)
        selected_indices = [base_indices[position] for position in positions]
        selected_by_count[view_count] = [names[index] for index in selected_indices]
        selected_indices_by_count[view_count] = selected_indices
        write_names(output / f"train_views_{view_count}.txt", selected_by_count[view_count])

    max_train = set(selected_by_count[max_view_count])
    test_names = [name for name in names if name not in max_train]
    write_names(output / "test_views.txt", test_names)

    metadata = {
        "source_path": os.path.abspath(source_path),
        "total_cameras": len(names),
        "ordering": "sorted by COLMAP image_name, matching scene/dataset_readers.py",
        "sampling": f"deterministic uniform sampling via numpy.linspace over the {max_view_count}-view split",
        "train_indices": {str(count): selected_indices_by_count[count] for count in sorted(view_counts)},
        "train_views": {str(count): selected_by_count[count] for count in sorted(view_counts)},
        "test_count": len(test_names),
    }
    with open(output / "split_metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)
    return metadata


def write_names(path, names):
    with open(path, "w") as file:
        for name in names:
            file.write(f"{name}\n")


def main():
    parser = argparse.ArgumentParser(description="Create deterministic sparse-view train/test splits.")
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("-o", "--output_dir", default=None)
    args = parser.parse_args()
    metadata = create_sparse_view_split(args.source_path, args.output_dir)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
