# AILab Project 2 — Progressive GAN (PGGAN) → FFHQ 1024, ONNX submission

## Checkpoints
- Training ckpts store `G_ema_state` (use for inference), `G_state`, `meta.generator_config`.
  Load: `Generator(GeneratorConfig.from_dict(ckpt["meta"]["generator_config"]))` then `load_state_dict(ckpt["G_ema_state"])`.
- Ckpts live in Google Drive on Colab: `/content/drive/MyDrive/OpenAILab/project2/<run>/ckpt_*.pt` (~558MB: G+D+EMA+optims).

## Generator.forward (src/model.py)
- `G(z)` = full native-1024 path (alpha=1.0). `G(z, resolution=R, alpha=a)` = progressive.
- alpha<1 → fade blend `(1-a)*nearest_up(toRGB_prev) + a*toRGB_R`. Mid-fade samples are NOT reproduced by bare `G(z)`.

## ONNX submission
- Contract: input z `(B,512)` fp32 → output `(B,3,1024,1024)` in [-1,1]. Export via `export_onnx.py`. **.onnx ≤ 200MB.**
- Grader supplies z, so truncation (`z*psi`) and any fixed alpha/upsample MUST be baked into the wrapper graph.

## Reuse & environment
- `train.py` is import-safe (`__main__` guard, wandb optional). Reuse: `extract_validation_subset`, `write_fake_validation_images`, `run_pytorch_fid`, `build_checkpoint`, `latest_checkpoint`.
- FID: `pip install pytorch-fid scipy`; subprocess over two image dirs; Inception resizes to 299, so 1024 fine detail barely affects FID.
- Progressive config `stages` support per-stage overrides: `batch_size, train_zip, lr_g, lr_d, ckpt_every`.
- Local Mac has no torch/numpy/onnxruntime (Pylance import warnings are noise) — can't run scripts here; syntax-check with `python -m py_compile <file>`. Training/running is on Colab.
