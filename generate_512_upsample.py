"""Generate a 512 sample grid and bilinear-upsample to 1024 for visual inspection.

For pure 512 checkpoints (512 stabilize phase) where you want to see
what the ONNX submission will look like after bilinear upsampling.

    python generate_512_upsample.py --ckpt runs/ckpt.pt --out sample.png
    python generate_512_upsample.py --ckpt runs/ckpt.pt --out sample.png --n 16 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.utils as vutils

from src.model import Generator, GeneratorConfig, build_baseline_256_generator


def load_generator(ckpt_path: Path, device: str, use_ema: bool) -> tuple[Generator, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "meta" in ckpt and "generator_config" in ckpt["meta"]:
        g_cfg = GeneratorConfig.from_dict(ckpt["meta"]["generator_config"])
        G = Generator(g_cfg).to(device).eval()
    else:
        G = build_baseline_256_generator().to(device).eval()

    state = ckpt.get("G_ema_state") if use_ema else None
    state = state or ckpt.get("G_state")
    if state is None:
        raise RuntimeError("Checkpoint has neither G_ema_state nor G_state")
    G.load_state_dict(state)

    n_params = sum(p.numel() for p in G.parameters())
    print(f"Loaded {'G_ema' if use_ema and 'G_ema_state' in ckpt else 'G'} "
          f"({n_params/1e6:.2f}M params, z_dim={G.z_dim})")
    return G, ckpt


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("sample_512_upsample.png"))
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--nrow", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--psi", type=float, default=1.0,
                        help="Truncation strength (z *= psi). 1.0 = no truncation.")
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    G, ckpt = load_generator(args.ckpt, device=device, use_ema=not args.no_ema)

    g_for_z = torch.Generator(device="cpu").manual_seed(args.seed)
    z = torch.randn(args.n, G.z_dim, generator=g_for_z).to(device) * args.psi
    if args.psi != 1.0:
        print(f"Truncation: psi={args.psi}")

    progressive_state = ckpt.get("progressive_state", {}) if isinstance(ckpt, dict) else {}
    resolution = progressive_state.get("resolution") if isinstance(progressive_state, dict) else None
    alpha = progressive_state.get("alpha", 1.0) if isinstance(progressive_state, dict) else 1.0

    if getattr(G.cfg, "progressive", False) and resolution is not None:
        resolution = int(resolution)
        alpha = float(alpha)
        print(f"Progressive generation: resolution={resolution}, alpha={alpha:.3f}")
        fake = G(z, resolution=resolution, alpha=alpha)
    else:
        fake = G(z)

    fake_1024 = F.interpolate(
        fake, size=(1024, 1024), mode="bilinear", align_corners=False
    )                                                              # (N, 3, 1024, 1024)

    x = ((fake_1024 + 1.0) / 2.0).clamp(0.0, 1.0)
    grid = vutils.make_grid(x, nrow=args.nrow, padding=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid, args.out)
    print(f"Native res: {tuple(fake.shape[-2:])} -> upsampled to {tuple(fake_1024.shape[-2:])}")
    print(f"Saved {args.n} samples -> {args.out}")


if __name__ == "__main__":
    main()
