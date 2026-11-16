"""Train LeWM (encoder and action-conditioned predictor, jointly, from scratch) on random-walk windows.

Local smoke run:  uv run python scripts/train.py --data <shard> --run-name smoke --max-steps 200 --val-batches 5
Cloud run:        uv run python scripts/train.py --data pretrain_100 pretrain_101 pretrain_102 pretrain_103 --out <persistent dir> --run-name run1
Resume:           add --resume. Preflight only: add --preflight-only.
Outputs: <out>/<run-name>/{checkpoints, csv, tb, env.json, preflight.json, summary.json}.
No goals or planning exist in this model (INT-11): the loss is next-latent prediction plus SIGReg."""

import argparse
import dataclasses
import datetime
import json
import os
import platform
import shutil
import time
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, OnExceptionCheckpoint, Timer
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from stable_pretraining.callbacks import HardwareMonitor

from hjepa_agent.callbacks import CollapseGuard
from hjepa_agent.config import OUTPUTS_DIR
from hjepa_agent.data import WindowDataset, compute_stats
from hjepa_agent.model import LeWMModule
from hjepa_agent.recipe import Recipe
from hjepa_agent.weights import make_sampler, window_weights


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class FailAt(pl.Callback):
    """Test hook: raise at a given step to prove that failed runs still leave checkpoints and a summary."""

    def __init__(self, step):
        self.step = step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step >= self.step:
            raise RuntimeError(f"injected failure at step {self.step}")


def latest_checkpoint(ck, which):
    """The newest checkpoint of any kind (last, on_exception, final, step), or last_good when asked."""
    if which == "last_good":
        p = ck / "last_good.ckpt"
        return p if p.exists() else None
    found = [p for p in ck.glob("*.ckpt") if p.name != "last_good.ckpt"]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def loaders(args, recipe, stats):
    train = WindowDataset(args.data, "train", recipe.frameskip, stats)
    val = WindowDataset(args.data, "val", recipe.frameskip, stats)
    sampler = make_sampler(train, window_weights(train)) if args.weighted else None
    nw = args.num_workers if args.num_workers is not None else (0 if os.name == "nt" else min(8, os.cpu_count() or 1))
    kw = dict(batch_size=recipe.batch_size, num_workers=nw, persistent_workers=nw > 0, pin_memory=torch.cuda.is_available(), drop_last=True)
    train_dl = torch.utils.data.DataLoader(train, shuffle=sampler is None, sampler=sampler, **kw)
    return train, val, train_dl, torch.utils.data.DataLoader(val, shuffle=False, **kw)


def preflight(args, recipe, out, train, train_dl, total_steps):
    """Fail fast, before any long run: hardware, disk, data shapes, real forward and backward steps, checkpoint roundtrip, time estimate."""
    res, t0 = {}, time.time()
    res["torch"], res["cuda"] = torch.__version__, torch.cuda.is_available()
    if args.require_gpu and not torch.cuda.is_available():
        raise SystemExit("preflight FAILED: no CUDA device (use --no-require-gpu for a CPU run)")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        res["gpu"], res["gpu_mem_gb"] = p.name, round(p.total_memory / 1e9, 1)
        if p.total_memory < 3.5e9:
            raise SystemExit(f"preflight FAILED: GPU memory {res['gpu_mem_gb']} GB is too small")
    res["disk_free_gb"] = round(shutil.disk_usage(out).free / 1e9, 1)
    if res["disk_free_gb"] < 5:
        raise SystemExit(f"preflight FAILED: only {res['disk_free_gb']} GB free in {out}")
    for s in train.shards:
        if tuple(s.frames.shape[2:]) != (48, 64, 3) or s.T < 3 * recipe.frameskip:
            raise SystemExit(f"preflight FAILED: shard {s.name} has frames {s.frames.shape}")
    res["windows_train"], res["total_steps"] = len(train), total_steps
    (out / "write_test.txt").write_text("ok")
    module = LeWMModule(recipe, total_steps)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    module.to(dev).train()
    opt = module.configure_optimizers()["optimizer"]
    it, times = iter(train_dl), []
    for i in range(6):
        b = {k: v.to(dev) for k, v in next(it).items() if k != "index"}
        s = time.time()
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            loss = module.training_step(b, i)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if dev == "cuda":
            torch.cuda.synchronize()
        if not torch.isfinite(loss):
            raise SystemExit(f"preflight FAILED: non-finite loss {float(loss)} at dry step {i}")
        times.append(time.time() - s)
    step_s = sum(times[2:]) / len(times[2:])
    res["step_seconds"], res["samples_per_second"] = round(step_s, 3), round(recipe.batch_size / step_s, 1)
    res["estimated_minutes"] = round(total_steps * step_s / 60, 1)
    if dev == "cuda":
        res["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    torch.save(module.state_dict(), out / "roundtrip.pt")
    LeWMModule(recipe, total_steps).load_state_dict(torch.load(out / "roundtrip.pt"))
    (out / "roundtrip.pt").unlink()
    res["seconds"] = round(time.time() - t0, 1)
    (out / "preflight.json").write_text(json.dumps(res, indent=1))
    if res["estimated_minutes"] > 1.2 * recipe.time_cap_minutes and not args.force:
        raise SystemExit(f"preflight FAILED: estimated {res['estimated_minutes']} min exceeds the {recipe.time_cap_minutes} min cap (use --force)")
    log(f"preflight OK: {res}")
    del module, opt
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=OUTPUTS_DIR)
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-steps", type=int, help="stop after this many steps (smoke runs)")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--sigreg-weight", type=float)
    ap.add_argument("--val-batches", type=int)
    ap.add_argument("--weighted", action="store_true", help="importance-weighted sampling for uniform coverage (INT-56)")
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--resume", nargs="?", const="latest", choices=["latest", "last_good"], help="resume from the newest checkpoint, or from last_good")
    ap.add_argument("--fail-at-step", type=int, help="test hook: raise at this step")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--no-require-gpu", dest="require_gpu", action="store_false")
    ap.add_argument("--force", action="store_true", help="ignore the preflight time-cap check")
    args = ap.parse_args()

    given = dict(epochs=args.epochs, batch_size=args.batch_size, sigreg_weight=args.sigreg_weight, val_batches=args.val_batches)
    recipe = dataclasses.replace(Recipe(), **{k: v for k, v in given.items() if v is not None})
    pl.seed_everything(recipe.seed, workers=True)
    out = args.out / args.run_name
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    env = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "lightning": pl.__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "args": {k: str(v) for k, v in vars(args).items()}, "recipe": dataclasses.asdict(recipe)}
    (out / "env.json").write_text(json.dumps(env, indent=1))
    log(f"output directory {out}")
    stats = compute_stats(args.data)
    train, val, train_dl, val_dl = loaders(args, recipe, stats)
    steps_per_epoch = len(train) // recipe.batch_size
    total_steps = args.max_steps or recipe.epochs * steps_per_epoch
    log(f"{len(train)} train windows, {len(val)} val windows, {steps_per_epoch} steps per epoch, {total_steps} steps in total")
    preflight(args, recipe, out, train, train_dl, total_steps)
    if args.preflight_only:
        return

    module = LeWMModule(recipe, total_steps)
    ck = out / "checkpoints"
    guard = CollapseGuard(ck, recipe.embed_dim, recipe.collapse_sigreg_max, recipe.collapse_rank_min_frac, recipe.collapse_patience, recipe.collapse_grace_steps)
    trainer = pl.Trainer(
        accelerator="auto", devices=1, precision=recipe.precision, gradient_clip_val=recipe.grad_clip,
        max_epochs=recipe.epochs if not args.max_steps else -1, max_steps=args.max_steps or -1,
        val_check_interval=min(recipe.val_every_steps, total_steps), check_val_every_n_epoch=None, limit_val_batches=recipe.val_batches,
        log_every_n_steps=10, logger=[CSVLogger(out, name="csv"), TensorBoardLogger(out, name="tb")],
        callbacks=[ModelCheckpoint(dirpath=ck, filename="step{step_idx:.0f}", monitor="step_idx", mode="max", save_top_k=3,
                                   every_n_train_steps=recipe.ckpt_every_steps, save_last=True, auto_insert_metric_name=False),
                   OnExceptionCheckpoint(dirpath=ck, filename="on_exception"),
                   Timer(duration=datetime.timedelta(minutes=recipe.time_cap_minutes)),
                   LearningRateMonitor(logging_interval="step"), HardwareMonitor(), guard] + ([FailAt(args.fail_at_step)] if args.fail_at_step else []),
        default_root_dir=out)
    status, t0 = "complete", time.time()
    resume = latest_checkpoint(ck, args.resume) if args.resume else None
    if args.resume:
        log(f"resuming from {resume}" if resume else "nothing to resume from; starting fresh")
    try:
        trainer.fit(module, train_dl, val_dl, ckpt_path=str(resume) if resume else None)
        if guard.stopped_for:
            status = guard.stopped_for
        elif trainer.global_step < total_steps:
            status = "stopped early (time cap or interrupt)"
    except BaseException as e:  # keep what we have, then re-raise
        status = f"failed: {type(e).__name__}: {e}"
        raise
    finally:
        trainer.save_checkpoint(ck / "final.ckpt")
        m = {k: float(v) for k, v in trainer.callback_metrics.items()}
        last_good = ck / "last_good.ckpt"
        summary = {"status": status, "steps": trainer.global_step, "total_steps": total_steps,
                   "wall_minutes": round((time.time() - t0) / 60, 1), "metrics": m, "last_good": str(last_good) if last_good.exists() else None}
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        log(f"finished: {status}; summary at {out / 'summary.json'}")


if __name__ == "__main__":
    main()
