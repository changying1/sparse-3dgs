import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description="Precompute continuous edge maps for a directory of images.")
    parser.add_argument("--input_dir", required=True, help="Directory containing training images.")
    parser.add_argument("--output_dir", required=True, help="Directory where .pt edge maps will be written.")
    parser.add_argument("--operator", default="scharr", choices=("scharr", "sobel"), help="Gradient operator.")
    parser.add_argument("--png_preview", action="store_true", help="Also save uint8 PNG previews for manual inspection.")
    args = parser.parse_args()

    import torch
    from PIL import Image

    from utils.edge_support import compute_edge_map
    from utils.general_utils import PILtoTorch

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        tensor = PILtoTorch(image, image.size)[:3]
        edge_map = compute_edge_map(tensor, operator=args.operator).cpu()

        output_path = output_dir / f"{image_path.stem}.pt"
        torch.save(edge_map, output_path)

        if args.png_preview:
            preview = (edge_map.clamp(0.0, 1.0) * 255.0).to(torch.uint8).numpy()
            Image.fromarray(preview).save(output_dir / f"{image_path.stem}.png")

        print(f"{image_path.name} -> {output_path.name}")


if __name__ == "__main__":
    main()
