"""Export a Generator to ONNX with truncation (z*psi) baked in, then upsample.

Same submission contract as export_onnx.py:
    input  z      shape (B, 512), dtype float32
    output image  shape (B, 3, 1024, 1024), dtype float32, range [-1, 1]

Difference from export_onnx.py: the wrapper scales z by `psi` *before* feeding
it to G. Truncation (psi < 1.0) pulls z toward the origin, biasing samples
toward higher per-image fidelity at the cost of diversity. Since the grader
only supplies raw z, this scaling must be baked into the exported graph —
hence a dedicated wrapper rather than a CLI flag bolted onto the baseline.

Pick `psi` with verify_fid.py (sweeps psi, reports FID per value) rather than
by eye — the psi that looks nicest is not always the psi with the lowest FID.

CLI:
    python export_onnx_truncated.py \
        --ckpt /path/to/ckpt.pt --out submission.onnx --psi 0.7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import Generator, GeneratorConfig


TARGET_RESOLUTION = 1024


class TruncatedSubmissionWrapper(nn.Module):
    """Run G(z*psi) at the checkpoint's native resolution, then resize to 1024×1024."""

    def __init__(self, G: nn.Module, psi: float, resolution: int | None, alpha: float):
        super().__init__()
        self.G = G
        self.psi = psi
        self.resolution = resolution
        self.alpha = alpha

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_trunc = z * self.psi                 # (B, 512) -> (B, 512)
        if self.resolution is not None and getattr(self.G, "cfg", None) and getattr(self.G.cfg, "progressive", False):
            x = self.G(z_trunc, resolution=self.resolution, alpha=self.alpha)
        else:
            x = self.G(z_trunc)                # (B, 3, R, R), [-1, 1]
        x = F.interpolate(
            x,
            size=(TARGET_RESOLUTION, TARGET_RESOLUTION),
            mode="bilinear",
            align_corners=False,
        )
        return x


def export_to_onnx(
    G: nn.Module,
    out_path: str | Path,
    *,
    psi: float,
    resolution: int | None = None,
    alpha: float = 1.0,
    opset: int = 17,
    batch_size: int = 1,
) -> None:
    """Export `G` wrapped as (B, 512) -> (B, 3, 1024, 1024) with truncation baked in.

    The batch dimension is exported dynamic; other dimensions are static.
    `G.z_dim` must equal 512 (assignment spec).
    """
    if getattr(G, "z_dim", None) != 512:
        raise ValueError(
            f"G.z_dim must be 512 (assignment spec). Got {getattr(G, 'z_dim', None)!r}."
        )

    G.eval()
    wrapper = TruncatedSubmissionWrapper(G, psi=psi, resolution=resolution, alpha=alpha).eval()

    dummy_z = torch.randn(batch_size, 512)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_z,
        str(out_path),
        input_names=["z"],
        output_names=["image"],
        opset_version=opset,
        dynamic_axes={"z": {0: "batch"}, "image": {0: "batch"}},
        dynamo=False,  # legacy tracer — avoids the onnxscript dependency
    )

    with torch.no_grad():
        ref_out = wrapper(dummy_z)
    print(f"Saved ONNX → {out_path}")
    print(f"  psi           {psi}")
    print(f"  input  z      (B, 512)")
    print(f"  output image  {tuple(ref_out.shape)} (B dynamic), range "
          f"[{ref_out.min():.3f}, {ref_out.max():.3f}]")


def _load_g(ckpt_path: Path) -> tuple[nn.Module, int | None, float]:
    """Load G_ema and progressive_state (resolution, alpha) from a student ckpt."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "meta" not in ckpt or "generator_config" not in ckpt.get("meta", {}):
        raise RuntimeError(
            "Checkpoint has no meta.generator_config — this script targets "
            "scaled-up student architectures, not the 256 baseline."
        )
    g_cfg = GeneratorConfig.from_dict(ckpt["meta"]["generator_config"])
    G = Generator(g_cfg)
    state = ckpt.get("G_ema_state") or ckpt.get("G_state")
    if state is None:
        raise RuntimeError("Checkpoint has neither G_ema_state nor G_state")
    G.load_state_dict(state)
    prog = ckpt.get("progressive_state") or {}
    resolution = int(prog["resolution"]) if "resolution" in prog else None
    alpha = float(prog.get("alpha", 1.0))
    print(f"progressive_state: resolution={resolution}, alpha={alpha:.3f}")
    return G, resolution, alpha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("submission_truncated.onnx"))
    parser.add_argument(
        "--psi", type=float, required=True,
        help="Truncation strength baked into the graph (z *= psi). "
             "Pick via verify_fid.py — do not guess.",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    G, resolution, alpha = _load_g(args.ckpt)
    export_to_onnx(G, args.out, psi=args.psi, resolution=resolution, alpha=alpha,
                   opset=args.opset, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
