"""Generate sample images from an exported ONNX generator.

Input contract (matches submission spec):
    z  (B, 512)  float32  → image  (B, 3, 1024, 1024)  float32  in [-1, 1]

CLI:
    python generate_from_onnx.py --onnx submission.onnx --out samples/onnx --n 16
    python generate_from_onnx.py --onnx submission.onnx --out samples/onnx --n 4 --grid
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def make_session(onnx_path: Path) -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    provider_used = session.get_providers()[0]
    print(f"ONNX Runtime provider: {provider_used}")
    return session


def generate_images(
    session: ort.InferenceSession,
    num_samples: int,
    seed: int,
    batch_size: int,
) -> np.ndarray:
    """Return float32 array (N, H, W, 3) in [0, 1]."""
    rng = np.random.default_rng(seed)
    all_images: list[np.ndarray] = []

    for start in range(0, num_samples, batch_size):
        batch_n = min(batch_size, num_samples - start)
        z = rng.standard_normal((batch_n, 512)).astype(np.float32)
        output = session.run(["image"], {"z": z})[0]  # (B, 3, H, W), [-1, 1]
        images_hwc = (output.transpose(0, 2, 3, 1) + 1.0) / 2.0  # (B, H, W, 3)
        all_images.append(images_hwc.clip(0.0, 1.0))
        print(f"Generated {start + batch_n}/{num_samples}")

    return np.concatenate(all_images, axis=0)


def save_individual(images: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img_f32 in enumerate(images):
        img_uint8 = (img_f32 * 255).astype(np.uint8)
        path = out_dir / f"sample_{i:04d}.png"
        Image.fromarray(img_uint8).save(path)
    print(f"Saved {len(images)} images → {out_dir}/")


def save_grid(images: np.ndarray, out_path: Path, nrow: int) -> None:
    n, h, w, c = images.shape
    ncol = (n + nrow - 1) // nrow
    grid = np.zeros((ncol * h, nrow * w, c), dtype=np.float32)
    for idx, img in enumerate(images):
        row, col = divmod(idx, nrow)
        grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = img
    grid_uint8 = (grid * 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid_uint8).save(out_path)
    print(f"Saved grid ({nrow}×{ncol}) → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--onnx", type=Path, required=True, help="Path to .onnx file")
    parser.add_argument("--out", type=Path, default=Path("samples/onnx_out"),
                        help="Output directory (individual PNGs) or grid path")
    parser.add_argument("--n", type=int, default=16, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Inference batch size (reduce if OOM)")
    parser.add_argument("--grid", action="store_true",
                        help="Save a single grid image instead of individual files")
    parser.add_argument("--nrow", type=int, default=4,
                        help="Columns per row in grid mode")
    args = parser.parse_args()

    session = make_session(args.onnx)
    images = generate_images(session, args.n, args.seed, args.batch_size)

    if args.grid:
        out_path = args.out if args.out.suffix else args.out / "grid.png"
        save_grid(images, out_path, nrow=args.nrow)
    else:
        save_individual(images, args.out)


if __name__ == "__main__":
    main()
