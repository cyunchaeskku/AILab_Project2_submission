"""Fine-tune script

Two start modes:

1) Fine-tune from the distributed 256 baseline (most common):
       python scripts/train.py --config configs/baseline_256.yaml \
                               --init-from ckpt/ffhq256_baseline.pt

2) Resume your own training run from a full ckpt you saved earlier:
       python scripts/train.py --config configs/baseline_256.yaml \
                               --resume runs/my_run/ckpt_001000000.pt

   `--resume` restores G, D, G_ema, both optimizers, and RNG state — bit-for-bit
   continuation (assuming the same architecture).

Recipe (the one that worked after three divergences):
- ResNet GAN: GN on G, Spectral Norm on D, self-attention at 32×32
- Non-saturating logistic loss + R1 (lazy every 16 D steps, γ=10)
- DiffAug 'color,translation' (cutout disabled — too aggressive)
- Adam β=(0, 0.9), G lr = D lr = 1e-3 (avoid TTUR until you observe a problem)
- EMA G (half-life 10k images)
- fp32 throughout

Logging via wandb if installed and not disabled.
FID is intentionally not measured here — measure it yourself between checkpoints.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import shutil
import sys
import threading
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import yaml
from PIL import Image
from torch.utils.data import DataLoader

# wandb is optional — keep training runnable on environments without it.
try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    wandb = None
    _HAS_WANDB = False

from src.augment import diff_augment
from src.dataset import ZipImageDataset, infinite_loader
from src.losses import ns_logistic_g, r1_penalty
from src.model import (
    Discriminator,
    DiscriminatorConfig,
    EMA,
    Generator,
    GeneratorConfig,
)


def _detect_gpu_config_key() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("Auto config requires CUDA GPU, but torch.cuda.is_available() is False")
    name = torch.cuda.get_device_name(0)
    upper = name.upper()
    if "A100" in upper:
        return "a100"
    if "L4" in upper:
        return "l4"
    raise RuntimeError(
        f"Unsupported GPU for auto config: {name!r}. Expected NVIDIA A100 or L4."
    )


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    auto_cfg = cfg.get("auto_config") if isinstance(cfg, dict) else None
    if auto_cfg:
        key = _detect_gpu_config_key()
        if key not in auto_cfg:
            raise KeyError(f"auto_config missing key for detected GPU: {key}")
        selected_path = Path(auto_cfg[key])
        if not selected_path.is_absolute():
            selected_path = path.parent / selected_path
        print(
            f"Auto config selected: {selected_path} "
            f"(gpu={torch.cuda.get_device_name(0)})",
            flush=True,
        )
        with open(selected_path) as selected_f:
            cfg = yaml.safe_load(selected_f)
        cfg.setdefault("meta", {})
        cfg["meta"]["selected_config"] = str(selected_path)
        cfg["meta"]["detected_gpu"] = torch.cuda.get_device_name(0)
    return cfg


def set_seed(seed: int) -> None:
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def copy_checkpoint(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def save_checkpoint(path: Path, state: dict, backup_path: Path | None = None) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    if backup_path is not None:
        copy_checkpoint(path, backup_path)


def async_save_checkpoint(
    path: Path,
    state: dict,
    backup_path: Path | None = None,
) -> threading.Thread:
    t = threading.Thread(target=save_checkpoint, args=(path, state, backup_path), daemon=False)
    t.start()
    return t


def latest_checkpoint(search_dirs: list[Path]) -> Path | None:
    candidates: list[Path] = []
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        candidates.extend(directory.glob("ckpt_*.pt"))
        final_path = directory / "final.pt"
        if final_path.is_file():
            candidates.append(final_path)
    if not candidates:
        return None
    def _step_key(p: Path) -> int:
        m = re.match(r"ckpt_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else float("inf")  # final.pt → inf

    return max(candidates, key=_step_key)


@torch.no_grad()
def save_sample_grid(
    G: torch.nn.Module,
    sample_z: torch.Tensor,
    out_path: Path,
    nrow: int = 8,
    *,
    resolution: int | None = None,
    alpha: float = 1.0,
    max_image_resolution: int | None = None,
) -> None:
    G.eval()
    fake = G(sample_z, resolution=resolution, alpha=alpha) if resolution is not None else G(sample_z)
    if max_image_resolution is not None and fake.shape[-1] > max_image_resolution:
        fake = F.interpolate(
            fake,
            size=(max_image_resolution, max_image_resolution),
            mode="bilinear",
            align_corners=False,
        )
    x = ((fake + 1.0) / 2.0).clamp(0.0, 1.0)
    grid = vutils.make_grid(x, nrow=nrow, padding=2)
    vutils.save_image(grid, out_path)


def build_loader(
    *,
    train_zip: str,
    flip: bool,
    batch_size: int,
    num_workers: int,
    device: str,
) -> tuple[ZipImageDataset, DataLoader, object]:
    dataset = ZipImageDataset(train_zip, flip=flip)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )
    return dataset, loader, infinite_loader(loader)


def build_progressive_schedule(cfg: dict, train_cfg: dict) -> list[dict]:
    stages = cfg.get("stages")
    if not stages:
        raise ValueError("progressive.enabled requires progressive.stages")
    schedule = []
    cursor = 0
    default_zip = train_cfg["train_zip"]
    default_batch = train_cfg["batch_size"]
    for idx, stage in enumerate(stages):
        images = int(stage["images"])
        if images <= 0:
            raise ValueError(f"progressive.stages[{idx}].images must be positive")
        resolution = int(stage["resolution"])
        phase = str(stage.get("phase", "stabilize"))
        if phase not in ("fade", "stabilize"):
            raise ValueError(f"Unknown progressive phase: {phase!r}")
        entry = {
            "index": idx,
            "resolution": resolution,
            "phase": phase,
            "images": images,
            "start": cursor,
            "end": cursor + images,
            "batch_size": int(stage.get("batch_size", default_batch)),
            "train_zip": str(stage.get("train_zip", default_zip)),
            "lr_g": float(stage["lr_g"]) if "lr_g" in stage else None,
            "lr_d": float(stage["lr_d"]) if "lr_d" in stage else None,
            "ckpt_every": int(stage["ckpt_every"]) if "ckpt_every" in stage else None,
        }
        schedule.append(entry)
        cursor += images
    return schedule


def stage_for_images(schedule: list[dict], images_seen: int) -> tuple[dict, int]:
    for stage in schedule:
        if images_seen < stage["end"]:
            return stage, images_seen - stage["start"]
    stage = schedule[-1]
    return stage, stage["images"]


def alpha_for_stage(stage: dict, phase_seen: int) -> float:
    if stage["phase"] != "fade":
        return 1.0
    return max(0.0, min(1.0, phase_seen / max(stage["images"], 1)))


def batch_diversity_loss(
    fake: torch.Tensor,
    *,
    target: float,
    max_resolution: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize only very low within-batch image variation.

    This is intentionally bounded with a hinge. Once the generated batch has
    enough pixel-level spread, the regularizer becomes zero instead of pushing
    the generator toward noisy outputs.
    """
    x = fake.float()
    if x.shape[-1] > max_resolution:
        x = F.interpolate(
            x,
            size=(max_resolution, max_resolution),
            mode="bilinear",
            align_corners=False,
        )
    flat = x.flatten(1)
    diversity = flat.std(dim=0, unbiased=False).mean()
    return F.relu(fake.new_tensor(float(target)) - diversity), diversity.detach()


def resize_real(real: torch.Tensor, resolution: int) -> torch.Tensor:
    if real.shape[-1] == resolution and real.shape[-2] == resolution:
        return real
    return F.interpolate(real, size=(resolution, resolution), mode="bilinear", align_corners=False)


def extract_validation_subset(
    *,
    zip_path: str,
    out_dir: Path,
    max_images: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png")) + list(out_dir.glob("*.jpeg"))
    if len(existing) >= max_images:
        return len(existing)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = sorted(
            n for n in zf.namelist()
            if n.lower().endswith((".png", ".jpg", ".jpeg"))
        )[:max_images]
        for idx, name in enumerate(names):
            suffix = Path(name).suffix.lower() or ".jpg"
            dst = out_dir / f"real_{idx:06d}{suffix}"
            if dst.exists():
                continue
            with zf.open(name, "r") as src:
                dst.write_bytes(src.read())
    return len(names)


@torch.no_grad()
def write_fake_validation_images(
    *,
    G: torch.nn.Module,
    out_dir: Path,
    z_dim: int,
    n_images: int,
    batch_size: int,
    device: str,
    seed: int,
    resolution: int | None,
    alpha: float,
) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    G.eval()
    written = 0
    while written < n_images:
        b = min(batch_size, n_images - written)
        z = torch.randn(b, z_dim, generator=gen).to(device)
        fake = G(z, resolution=resolution, alpha=alpha) if resolution is not None else G(z)
        fake = ((fake + 1.0) / 2.0).clamp(0.0, 1.0).cpu()
        for i in range(b):
            array = (fake[i].permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
            Image.fromarray(array).save(out_dir / f"fake_{written + i:06d}.png")
        written += b


def run_pytorch_fid(fake_dir: Path, real_dir: Path, device: str) -> float | None:
    cmd = [
        sys.executable,
        "-m",
        "pytorch_fid",
        str(fake_dir),
        str(real_dir),
        "--device",
        "cuda" if device == "cuda" else "cpu",
    ]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        print("[val] python executable not found; skipping FID", flush=True)
        return None

    output = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode != 0:
        print("[val] pytorch-fid failed; install with `pip install pytorch-fid scipy`", flush=True)
        print(output.strip()[-1000:], flush=True)
        return None
    for line in output.splitlines():
        if "FID:" in line:
            try:
                return float(line.split("FID:")[-1].strip())
            except ValueError:
                continue
    print("[val] could not parse FID output", flush=True)
    print(output.strip()[-1000:], flush=True)
    return None


def validation_zip_for_resolution(validation_cfg: dict, resolution: int) -> str | None:
    zips = validation_cfg.get("valid_zips", {}) or {}
    return zips.get(str(resolution)) or zips.get(int(resolution))


def build_checkpoint(
    *,
    images_seen: int,
    step: int,
    G: torch.nn.Module,
    D: torch.nn.Module,
    G_ema: EMA,
    optG: torch.optim.Optimizer,
    optD: torch.optim.Optimizer,
    g_cfg: GeneratorConfig,
    d_cfg: DiscriminatorConfig,
    training_cfg: dict,
    wandb_run_id: str | None,
    progressive_state: dict | None = None,
) -> dict:
    ckpt = {
        "images_seen": images_seen,
        "step": step,
        "G_state": G.state_dict(),
        "D_state": D.state_dict(),
        "G_ema_state": G_ema.state_dict(),
        "optG_state": optG.state_dict(),
        "optD_state": optD.state_dict(),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
        },
        "wandb_run_id": wandb_run_id,
        "meta": {
            "generator_config": asdict(g_cfg),
            "discriminator_config": asdict(d_cfg),
            "training_config": training_cfg,
        },
    }
    if progressive_state is not None:
        ckpt["progressive_state"] = progressive_state
    return ckpt


def load_matching_weights(
    model: torch.nn.Module,
    state_dict: dict,
    name: str,
) -> None:
    """Load only tensors whose names and shapes match the current model.

    This lets a 512/1024 generator reuse the existing lower-resolution blocks
    from a 256/512 checkpoint while leaving new high-resolution layers random.
    """
    model_state = model.state_dict()
    matched = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            matched[key] = value
        else:
            skipped.append(key)

    model_state.update(matched)
    model.load_state_dict(model_state)
    print(f"  {name}: loaded {len(matched)} tensors, skipped {len(skipped)}")


def checkpoint_resolution(ckpt: dict, kind: str) -> int:
    meta = ckpt.get("meta", {}) if isinstance(ckpt.get("meta", {}), dict) else {}
    if kind == "G":
        cfg = meta.get("generator_config")
        if isinstance(cfg, dict) and cfg.get("resolutions"):
            return int(cfg["resolutions"][-1])
    if kind == "D":
        cfg = meta.get("discriminator_config")
        if isinstance(cfg, dict) and cfg.get("resolutions"):
            return int(cfg["resolutions"][0])
    return 256


def _replace_stage_index(key: str, shift: int) -> str:
    parts = key.split(".")
    if len(parts) >= 3 and parts[0] == "stages" and parts[1].isdigit():
        parts[1] = str(int(parts[1]) + shift)
        return ".".join(parts)
    return key


def progressive_target_key(
    model: torch.nn.Module,
    key: str,
    *,
    source_resolution: int,
    kind: str,
) -> str:
    if kind == "G":
        target_resolution = model.cfg.resolutions[-1]
        if source_resolution > target_resolution:
            # higher-res ckpt → lower-res model: remap prev_to_rgbs/prev_out_norms at target_res
            pfx = f"prev_to_rgbs.{target_resolution}."
            if key.startswith(pfx):
                return f"to_rgb.{key[len(pfx):]}"
            pfx = f"prev_out_norms.{target_resolution}."
            if key.startswith(pfx):
                return f"out_norm.{key[len(pfx):]}"
            return key  # stages.0..N match directly; extra stages skipped by shape check
        elif source_resolution < target_resolution:
            if key.startswith("to_rgb."):
                return f"prev_to_rgbs.{source_resolution}.{key[len('to_rgb.'):]}"
            if key.startswith("out_norm."):
                return f"prev_out_norms.{source_resolution}.{key[len('out_norm.'):]}"
        return key

    if kind == "D":
        target_resolution = model.cfg.resolutions[0]
        if source_resolution > target_resolution:
            # higher-res ckpt → lower-res model: shift stage indices down
            stage_shift = source_resolution.bit_length() - target_resolution.bit_length()
            pfx = f"prev_from_rgbs.{target_resolution}."
            if key.startswith(pfx):
                return f"from_rgb.{key[len(pfx):]}"
            if key.startswith("stages."):
                parts = key.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    new_idx = int(parts[1]) - stage_shift
                    if new_idx < 0:
                        return key  # above target resolution, will be skipped
                    parts[1] = str(new_idx)
                    return ".".join(parts)
            return key
        elif source_resolution < target_resolution:
            if key.startswith("from_rgb."):
                return f"prev_from_rgbs.{source_resolution}.{key[len('from_rgb.'):]}"
            if key.startswith("stages."):
                shift = int(model.stage_unit_start_idx[source_resolution])
                return _replace_stage_index(key, shift)
        return key

    return key


def load_progressive_weights(
    model: torch.nn.Module,
    state_dict: dict,
    name: str,
    *,
    source_resolution: int,
) -> None:
    model_state = model.state_dict()
    matched = {}
    skipped = []
    for key, value in state_dict.items():
        target_key = progressive_target_key(
            model,
            key,
            source_resolution=source_resolution,
            kind=name,
        )
        if target_key in model_state and model_state[target_key].shape == value.shape:
            matched[target_key] = value
        else:
            skipped.append((key, target_key))

    model_state.update(matched)
    model.load_state_dict(model_state)
    print(
        f"  {name}: loaded {len(matched)} tensors with progressive remap "
        f"(source_res={source_resolution}), skipped {len(skipped)}"
    )


def init_from_checkpoint(
    init_path: Path,
    G: torch.nn.Module,
    D: torch.nn.Module,
    G_ema: EMA,
    device: str,
) -> None:
    """Partial init from a lower-resolution or same-resolution checkpoint."""
    print(f"Initializing from checkpoint: {init_path}")
    ckpt = torch.load(init_path, map_location=device, weights_only=False)

    if "G_state" not in ckpt:
        raise RuntimeError(f"Checkpoint has no G_state: {init_path}")
    g_source_resolution = checkpoint_resolution(ckpt, "G")
    d_source_resolution = checkpoint_resolution(ckpt, "D")
    if getattr(G.cfg, "progressive", False):
        load_progressive_weights(
            G,
            ckpt["G_state"],
            "G",
            source_resolution=g_source_resolution,
        )
    else:
        load_matching_weights(G, ckpt["G_state"], "G")

    if "G_ema_state" in ckpt:
        if getattr(G_ema.shadow.cfg, "progressive", False):
            load_progressive_weights(
                G_ema.shadow,
                ckpt["G_ema_state"],
                "G",
                source_resolution=g_source_resolution,
            )
        else:
            load_matching_weights(G_ema.shadow, ckpt["G_ema_state"], "G_ema")
    else:
        if getattr(G_ema.shadow.cfg, "progressive", False):
            load_progressive_weights(
                G_ema.shadow,
                ckpt["G_state"],
                "G",
                source_resolution=g_source_resolution,
            )
        else:
            load_matching_weights(G_ema.shadow, ckpt["G_state"], "G_ema")

    if "D_state" in ckpt:
        if getattr(D.cfg, "progressive", False):
            load_progressive_weights(
                D,
                ckpt["D_state"],
                "D",
                source_resolution=d_source_resolution,
            )
        else:
            load_matching_weights(D, ckpt["D_state"], "D")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--init-from", type=Path, default=None,
        help="Path to a (possibly slim) baseline ckpt. Partial load with "
             "strict=False; optimizers/RNG start fresh.",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to a full ckpt saved by this same script. Restores "
             "G/D/G_ema/optimizers/RNG/wandb run id.",
    )
    parser.add_argument(
        "--auto-resume", action="store_true",
        help="Resume from latest ckpt in out.run_dir or out.backup_dir if present; "
             "otherwise fall back to --init-from.",
    )
    parser.add_argument("--total-images", type=int, default=None)
    parser.add_argument(
        "--new-wandb-run", action="store_true",
        help="When --resume, start a fresh wandb run instead of reattaching.",
    )
    args = parser.parse_args()

    if args.init_from is not None and args.resume is not None:
        raise SystemExit("Use either --init-from or --resume, not both.")

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    if args.total_images is not None:
        train_cfg["total_images"] = args.total_images

    out_cfg = cfg.get("out", {})
    run_dir = Path(out_cfg["run_dir"])
    backup_dir = Path(out_cfg["backup_dir"]) if out_cfg.get("backup_dir") else None
    if args.auto_resume and args.resume is None:
        search_dirs = [run_dir]
        if backup_dir is not None:
            search_dirs.insert(0, backup_dir)
        found = latest_checkpoint(search_dirs)
        if found is not None:
            args.resume = found
            args.init_from = None
            print(f"Auto-resume found checkpoint: {found}")
        else:
            print("Auto-resume found no checkpoint; starting from --init-from.")

    set_seed(train_cfg["seed"])
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    g_cfg = GeneratorConfig.from_dict(cfg["generator"])
    d_cfg = DiscriminatorConfig.from_dict(cfg["discriminator"])
    G = Generator(g_cfg).to(device)
    D = Discriminator(d_cfg).to(device)
    print(f"Generator: {sum(p.numel() for p in G.parameters())/1e6:.2f}M params")
    print(f"Discriminator: {sum(p.numel() for p in D.parameters())/1e6:.2f}M params")

    lr_g = float(train_cfg.get("lr_g", train_cfg.get("lr")))
    lr_d = float(train_cfg.get("lr_d", train_cfg.get("lr")))
    optG = torch.optim.Adam(
        G.parameters(), lr=lr_g,
        betas=(train_cfg["beta1"], train_cfg["beta2"]), eps=1e-8,
        weight_decay=train_cfg["weight_decay"],
    )
    optD = torch.optim.Adam(
        D.parameters(), lr=lr_d,
        betas=(train_cfg["beta1"], train_cfg["beta2"]), eps=1e-8,
        weight_decay=train_cfg["weight_decay"],
    )
    print(f"Optimizers: G lr={lr_g}, D lr={lr_d}")

    G_ema = EMA(G, half_life=train_cfg["ema_half_life"])
    G_ema.shadow.to(device)

    num_workers = train_cfg["num_workers"]
    progressive_cfg = cfg.get("progressive", {}) or {}
    progressive_enabled = bool(progressive_cfg.get("enabled", False))
    progressive_schedule = (
        build_progressive_schedule(progressive_cfg, train_cfg)
        if progressive_enabled else []
    )
    if progressive_enabled:
        schedule_total = progressive_schedule[-1]["end"]
        if args.total_images is None:
            train_cfg["total_images"] = schedule_total
        print("Progressive schedule:")
        for stage in progressive_schedule:
            print(
                f"  #{stage['index']} {stage['phase']} res={stage['resolution']} "
                f"images={stage['images']} batch={stage['batch_size']} "
                f"zip={stage['train_zip']}"
            )

    sample_gen = torch.Generator(device="cpu").manual_seed(train_cfg["sample_seed"])
    sample_z = torch.randn(train_cfg["sample_n"], g_cfg.z_dim, generator=sample_gen).to(device)

    run_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoint backup dir: {backup_dir}")

    images_seen = 0
    step = 0
    wandb_run_id: str | None = None

    if args.init_from is not None:
        init_from_checkpoint(args.init_from, G, D, G_ema, device=device)

    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["G_state"])
        D.load_state_dict(ckpt["D_state"])
        G_ema.load_state_dict(ckpt["G_ema_state"])
        if "optG_state" in ckpt:
            optG.load_state_dict(ckpt["optG_state"])
        if "optD_state" in ckpt:
            optD.load_state_dict(ckpt["optD_state"])
        # Force yaml LR onto the loaded optimizer state.
        for pg in optG.param_groups:
            pg["lr"] = lr_g
        for pg in optD.param_groups:
            pg["lr"] = lr_d
        images_seen = ckpt.get("images_seen", 0)
        step = ckpt.get("step", 0)
        wandb_run_id = None if args.new_wandb_run else ckpt.get("wandb_run_id")
        rng = ckpt.get("rng_state", {})
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu())
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])

    # wandb
    wandb_cfg = cfg.get("wandb", {})
    wandb_mode = wandb_cfg.get("mode", "online") if _HAS_WANDB else "disabled"
    run = None
    if wandb_mode != "disabled":
        init_kwargs = {
            "project": wandb_cfg.get("project", "ffhqgen-student"),
            "name": wandb_cfg.get("name"),
            "mode": wandb_mode,
            "config": cfg,
        }
        if wandb_run_id is not None:
            init_kwargs["id"] = wandb_run_id
            init_kwargs["resume"] = "must"
        run = wandb.init(**init_kwargs)
        wandb_run_id = run.id

    validation_cfg = cfg.get("validation", {}) or {}
    validation_enabled = bool(validation_cfg.get("enabled", False))
    val_every = int(validation_cfg.get("every_images", 100_000))
    val_n_fake = int(validation_cfg.get("n_fake", 512))
    val_n_real = int(validation_cfg.get("n_real", 2_000))
    val_batch_size = int(validation_cfg.get("batch_size", 8))
    val_seed = int(validation_cfg.get("seed", 2026))
    val_keep_fake = bool(validation_cfg.get("keep_fake", False))
    validation_dir = run_dir / "validation"

    total_images = train_cfg["total_images"]
    z_dim = g_cfg.z_dim
    r1_gamma = train_cfg["r1_gamma"]
    r1_lazy_every = train_cfg["r1_lazy_every"]
    log_every = train_cfg["log_every"]
    ckpt_every = train_cfg["ckpt_every"]
    sample_every = int(train_cfg.get("sample_every", ckpt_every))
    sample_grid_max_resolution = train_cfg.get("sample_grid_max_resolution")
    sample_grid_max_resolution = (
        int(sample_grid_max_resolution)
        if sample_grid_max_resolution is not None else None
    )
    diversity_cfg = train_cfg.get("diversity_regularizer", {}) or {}
    diversity_weight = float(diversity_cfg.get("weight", 0.0))
    diversity_target = float(diversity_cfg.get("target", 0.12))
    diversity_every = max(1, int(diversity_cfg.get("every", 1)))
    diversity_max_resolution = int(diversity_cfg.get("max_resolution", 128))
    if diversity_weight > 0:
        print(
            "Diversity regularizer: "
            f"weight={diversity_weight} target={diversity_target} "
            f"every={diversity_every} max_res={diversity_max_resolution}",
            flush=True,
        )
    grad_clip_g = float(train_cfg.get("grad_clip_g", float("inf")))
    grad_clip_d = float(train_cfg.get("grad_clip_d", float("inf")))
    precision = train_cfg.get("precision", "fp32")
    if precision not in ("bf16", "fp32"):
        raise ValueError(f"precision must be 'bf16' or 'fp32', got {precision!r}")
    use_amp = precision == "bf16"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    print(f"Precision: {precision} ({'autocast bf16' if use_amp else 'fp32 throughout'})")
    augment_policy = train_cfg.get("augment", "") or ""
    print(f"Augment policy: {augment_policy!r}")

    dataset = None
    loader = None
    inf_loader = None
    active_stage_index: int | None = None
    current_resolution: int | None = None
    current_alpha = 1.0
    current_phase = "fixed"
    current_stage: dict | None = None
    if not progressive_enabled:
        dataset, loader, inf_loader = build_loader(
            train_zip=train_cfg["train_zip"],
            flip=train_cfg["flip"],
            batch_size=train_cfg["batch_size"],
            num_workers=num_workers,
            device=device,
        )
        print(f"Dataset: {len(dataset)} images")

    last_ckpt = images_seen
    last_sample = images_seen
    last_validation = images_seen
    save_threads: list[threading.Thread] = []
    window_t0 = time.perf_counter()
    window_imgs = 0
    last_r1_value: float | None = None

    batch_desc = "stage-dependent" if progressive_enabled else str(train_cfg["batch_size"])
    print(
        f"Training: images_seen={images_seen} → {total_images} "
        f"(batch={batch_desc}, device={device})"
    )

    while images_seen < total_images:
        if progressive_enabled:
            current_stage, phase_seen = stage_for_images(progressive_schedule, images_seen)
            current_resolution = int(current_stage["resolution"])
            current_alpha = alpha_for_stage(current_stage, phase_seen)
            current_phase = str(current_stage["phase"])
            if active_stage_index != current_stage["index"]:
                dataset, loader, inf_loader = build_loader(
                    train_zip=current_stage["train_zip"],
                    flip=train_cfg["flip"],
                    batch_size=current_stage["batch_size"],
                    num_workers=num_workers,
                    device=device,
                )
                active_stage_index = current_stage["index"]
                print(
                    f"[stage] #{active_stage_index} {current_phase} "
                    f"res={current_resolution} alpha={current_alpha:.3f} "
                    f"batch={current_stage['batch_size']} dataset={len(dataset)}"
                )
                if current_stage.get("lr_g") is not None:
                    for pg in optG.param_groups:
                        pg["lr"] = current_stage["lr_g"]
                if current_stage.get("lr_d") is not None:
                    for pg in optD.param_groups:
                        pg["lr"] = current_stage["lr_d"]
                if current_stage.get("lr_g") is not None or current_stage.get("lr_d") is not None:
                    print(
                        f"[stage] lr_g={optG.param_groups[0]['lr']:.6g} "
                        f"lr_d={optD.param_groups[0]['lr']:.6g}"
                    )

        if inf_loader is None:
            raise RuntimeError("Training dataloader was not initialized")
        real = next(inf_loader).to(device, non_blocking=True)
        if current_resolution is not None:
            real = resize_real(real, current_resolution)
        b = real.size(0)

        # --- D step ---
        z = torch.randn(b, z_dim, device=device)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            with torch.no_grad():
                fake = G(z, resolution=current_resolution, alpha=current_alpha)
            d_real = D(
                diff_augment(real, augment_policy),
                resolution=current_resolution,
                alpha=current_alpha,
            )
            d_fake = D(
                diff_augment(fake.detach(), augment_policy),
                resolution=current_resolution,
                alpha=current_alpha,
            )
            l_d_real = F.softplus(-d_real).mean()
            l_d_fake = F.softplus(d_fake).mean()
            l_d = l_d_real + l_d_fake
        optD.zero_grad(set_to_none=True)
        l_d.backward()

        if (step + 1) % r1_lazy_every == 0:
            l_r1 = r1_lazy_every * r1_penalty(
                D,
                diff_augment(real.float(), augment_policy),
                gamma=r1_gamma,
                resolution=current_resolution,
                alpha=current_alpha,
            )
            l_r1.backward()
            last_r1_value = float(l_r1.item()) / r1_lazy_every

        grad_norm_d = float(
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=grad_clip_d)
        )
        optD.step()

        # --- G step ---
        z = torch.randn(b, z_dim, device=device)
        diversity_value = None
        diversity_penalty = None
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            fake = G(z, resolution=current_resolution, alpha=current_alpha)
            d_fake_g = D(
                diff_augment(fake, augment_policy),
                resolution=current_resolution,
                alpha=current_alpha,
            )
            l_g = ns_logistic_g(d_fake_g)
            if diversity_weight > 0 and (step + 1) % diversity_every == 0:
                diversity_penalty, diversity_value = batch_diversity_loss(
                    fake,
                    target=diversity_target,
                    max_resolution=diversity_max_resolution,
                )
                l_g = l_g + diversity_weight * diversity_penalty
        optG.zero_grad(set_to_none=True)
        l_g.backward()
        grad_norm_g = float(
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=grad_clip_g)
        )
        optG.step()

        G_ema.update(G, b)

        images_seen += b
        window_imgs += b
        step += 1

        if step % log_every == 0:
            now = time.perf_counter()
            elapsed = max(now - window_t0, 1e-6)
            throughput = window_imgs / elapsed
            window_t0 = now
            window_imgs = 0
            log = {
                "images_seen": images_seen,
                "throughput/imgs_per_sec": throughput,
                "loss/D_total": float(l_d.item()),
                "loss/D_real": float(l_d_real.item()),
                "loss/D_fake": float(l_d_fake.item()),
                "loss/G": float(l_g.item()),
                "D_out/real_mean": float(d_real.float().mean().item()),
                "D_out/fake_mean": float(d_fake.float().mean().item()),
                "grad_norm/G": grad_norm_g,
                "grad_norm/D": grad_norm_d,
                "lr": optG.param_groups[0]["lr"],
            }
            if diversity_value is not None and diversity_penalty is not None:
                log["diversity/batch_std"] = float(diversity_value.item())
                log["diversity/penalty"] = float(diversity_penalty.item())
            if last_r1_value is not None:
                log["loss/R1"] = last_r1_value
            if progressive_enabled:
                log.update({
                    "progressive/resolution": current_resolution,
                    "progressive/alpha": current_alpha,
                    "progressive/stage": active_stage_index,
                    "progressive/is_fade": 1 if current_phase == "fade" else 0,
                })
            if wandb_mode != "disabled":
                wandb.log(log, step=step)
            progress_msg = (
                f"step={step} imgs={images_seen} thr={throughput:.1f}img/s "
                f"l_d={l_d.item():.3f} l_g={l_g.item():.3f} "
                f"d_real={d_real.float().mean().item():.3f} "
                f"d_fake={d_fake.float().mean().item():.3f} "
                f"gn_g={grad_norm_g:.2f} gn_d={grad_norm_d:.2f}"
            )
            if progressive_enabled:
                progress_msg += (
                    f" res={current_resolution} phase={current_phase} "
                    f"alpha={current_alpha:.3f}"
                )
            print(progress_msg, flush=True)

        if images_seen - last_sample >= sample_every:
            grid_path = samples_dir / f"grid_{images_seen:09d}.png"
            save_sample_grid(
                G_ema.shadow,
                sample_z,
                grid_path,
                nrow=8,
                resolution=current_resolution,
                alpha=current_alpha,
                max_image_resolution=sample_grid_max_resolution,
            )
            if wandb_mode != "disabled":
                wandb.log({"samples/grid": wandb.Image(str(grid_path))}, step=step)
            print(f"[sample] {grid_path.name}", flush=True)
            last_sample = images_seen

        if validation_enabled and images_seen - last_validation >= val_every:
            if current_resolution is None:
                val_resolution = int(train_cfg.get("resolution", g_cfg.resolutions[-1]))
            else:
                val_resolution = int(current_resolution)
            valid_zip = validation_zip_for_resolution(validation_cfg, val_resolution)
            if valid_zip is None:
                print(f"[val] no valid zip configured for res={val_resolution}; skipping", flush=True)
                last_validation = images_seen
            elif not Path(valid_zip).is_file():
                print(f"[val] valid zip not found: {valid_zip}; skipping", flush=True)
                last_validation = images_seen
            else:
                print(
                    f"[val] start imgs={images_seen} res={val_resolution} "
                    f"fake={val_n_fake} real={val_n_real}",
                    flush=True,
                )
                real_dir = validation_dir / f"real_{val_resolution}_{val_n_real}"
                fake_dir = validation_dir / f"fake_{images_seen:09d}_{val_resolution}"
                n_real = extract_validation_subset(
                    zip_path=valid_zip,
                    out_dir=real_dir,
                    max_images=val_n_real,
                )
                write_fake_validation_images(
                    G=G_ema.shadow,
                    out_dir=fake_dir,
                    z_dim=z_dim,
                    n_images=val_n_fake,
                    batch_size=val_batch_size,
                    device=device,
                    seed=val_seed,
                    resolution=val_resolution if progressive_enabled else None,
                    alpha=current_alpha if progressive_enabled else 1.0,
                )
                fid = run_pytorch_fid(fake_dir, real_dir, device=device)
                if fid is not None:
                    print(f"[val] fid={fid:.4f} imgs={images_seen} res={val_resolution}", flush=True)
                    if wandb_mode != "disabled":
                        wandb.log(
                            {
                                "validation/fid": fid,
                                "validation/resolution": val_resolution,
                                "validation/n_fake": val_n_fake,
                                "validation/n_real": n_real,
                            },
                            step=step,
                        )
                if not val_keep_fake:
                    shutil.rmtree(fake_dir, ignore_errors=True)
                last_validation = images_seen

        stage_ckpt_every = (
            current_stage.get("ckpt_every")
            if progressive_enabled and current_stage is not None
            else None
        )
        effective_ckpt_every = stage_ckpt_every or ckpt_every
        if images_seen - last_ckpt >= effective_ckpt_every:
            ckpt = build_checkpoint(
                images_seen=images_seen, step=step,
                G=G, D=D, G_ema=G_ema, optG=optG, optD=optD,
                g_cfg=g_cfg, d_cfg=d_cfg, training_cfg=train_cfg,
                wandb_run_id=wandb_run_id,
                progressive_state={
                    "stage": active_stage_index,
                    "resolution": current_resolution,
                    "phase": current_phase,
                    "alpha": current_alpha,
                } if progressive_enabled else None,
            )
            ckpt_path = run_dir / f"ckpt_{images_seen:09d}.pt"
            backup_ckpt_path = (
                backup_dir / ckpt_path.name if backup_dir is not None else None
            )
            grid_path = samples_dir / f"grid_{images_seen:09d}.png"
            save_threads = [t for t in save_threads if t.is_alive()]
            save_threads.append(async_save_checkpoint(ckpt_path, ckpt, backup_ckpt_path))
            save_sample_grid(
                G_ema.shadow,
                sample_z,
                grid_path,
                nrow=8,
                resolution=current_resolution,
                alpha=current_alpha,
                max_image_resolution=sample_grid_max_resolution,
            )
            if wandb_mode != "disabled":
                wandb.log({"samples/grid": wandb.Image(str(grid_path))}, step=step)
            print(f"[ckpt+grid] {ckpt_path.name} / {grid_path.name}")
            last_ckpt = images_seen

    print("Training complete. Saving final ckpt...")
    final_ckpt = build_checkpoint(
        images_seen=images_seen, step=step,
        G=G, D=D, G_ema=G_ema, optG=optG, optD=optD,
        g_cfg=g_cfg, d_cfg=d_cfg, training_cfg=train_cfg,
        wandb_run_id=wandb_run_id,
        progressive_state={
            "stage": active_stage_index,
            "resolution": current_resolution,
            "phase": current_phase,
            "alpha": current_alpha,
        } if progressive_enabled else None,
    )
    final_path = run_dir / "final.pt"
    backup_final_path = backup_dir / "final.pt" if backup_dir is not None else None
    save_checkpoint(final_path, final_ckpt, backup_final_path)
    for t in save_threads:
        t.join()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
