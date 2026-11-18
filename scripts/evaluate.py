"""Offline evaluation of a checkpoint: rollout error by horizon, layer probes, history probe, transfer sets.

uv run python scripts/evaluate.py --ckpt <path> --out <dir> [--data map_pretrain_200 ...] [--transfer map_transfer_chasers ...]
Writes <out>/evaluation.json and <out>/rollout.png. Labels are privileged and used only here."""

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hjepa_agent import evaluation as ev
from hjepa_agent.data import compute_stats, frame_labels, load_frames, sample_frame_indices

PRETRAIN = ["map_pretrain_200", "map_pretrain_201", "map_pretrain_202", "map_pretrain_203"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", nargs="+", default=PRETRAIN)
    ap.add_argument("--transfer", nargs="*", default=["map_transfer_chasers", "map_transfer_patch", "map_transfer_floor1"])
    ap.add_argument("--episodes", type=int, default=512, help="rollout sequences per data set")
    ap.add_argument("--horizon", type=int, default=16, help="rollout steps after the 3-frame context")
    ap.add_argument("--train-n", type=int, default=20000)
    ap.add_argument("--val-n", type=int, default=5000)
    ap.add_argument("--parts", nargs="+", default=["rollout", "layers", "history", "transfer"])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, stats, res = ev.load_module(a.ckpt, dev), compute_stats(a.data), {"ckpt": str(a.ckpt), "data": a.data}
    length = 3 + a.horizon

    if "rollout" in a.parts:
        px, ac, _ = ev.episode_sequences(a.data, "val", a.episodes, length, stats=stats, seed=1)
        res["rollout"] = ev.rollout_errors(m, px, ac)
        log(f"rollout: model/copy at h=1 {res['rollout']['model_over_copy'][0]:.3f}, h={a.horizon} {res['rollout']['model_over_copy'][-1]:.3f}")

    if "layers" in a.parts:
        itr, iva = sample_frame_indices(a.data, "train", a.train_n, 0), sample_frame_indices(a.data, "val", a.val_n, 1)
        ytr, yva = frame_labels(a.data, "train", itr), frame_labels(a.data, "val", iva)
        ttr, tva = ev.identifiable_targets(ytr), ev.identifiable_targets(yva)
        xtr, xva = load_frames(a.data, itr, stats), load_frames(a.data, iva, stats)
        ftr, fva = ev.layer_features(m, xtr), ev.layer_features(m, xva)
        ftr["patch_all"], fva["patch_all"] = ev.reduce(ftr["patch_all"], fva["patch_all"])
        ftr["pixels"], fva["pixels"] = ev.reduce(xtr.flatten(1).numpy(), xva.flatten(1).numpy())
        res["layers"] = {}
        for layer in ftr:
            r = {"room_auc": ev.room_auc(ftr[layer], ytr["room2"], fva[layer], yva["room2"])}
            for k in ttr:
                r[f"{k}_ridge"] = ev.probe(ftr[layer], ttr[k], fva[layer], tva[k], "ridge")
                if k in ("cos4h", "sin4h", "abs_pos_y", "pos_x"):
                    r[f"{k}_mlp"] = ev.probe(ftr[layer], ttr[k], fva[layer], tva[k], "mlp")
            res["layers"][layer] = r
            log(f"layer {layer}: cos4h mlp {r['cos4h_mlp']:.3f}, room AUC {r['room_auc']:.3f}")

    if "history" in a.parts:
        ftr, ytr = ev.history_features(m, a.data, "train", a.train_n, stats, seed=0)
        fva, yva = ev.history_features(m, a.data, "val", a.val_n, stats, seed=1)
        res["history"] = {k: ev.probe(ftr, ytr[k], fva, yva[k], "mlp") for k in ("heading_cos", "heading_sin", "pos_y")}
        log(f"history (3 frames + actions), MLP: {res['history']}")

    if "transfer" in a.parts:
        res["transfer"] = {}
        for t in a.transfer:
            px, ac, _ = ev.episode_sequences(t, "val", min(a.episodes, 128), length, stats=stats, seed=1)
            res["transfer"][t] = ev.rollout_errors(m, px, ac)
            log(f"transfer {t}: model/copy at h=1 {res['transfer'][t]['model_over_copy'][0]:.3f}")

    (a.out / "evaluation.json").write_text(json.dumps(res, indent=1))
    if "rollout" in res:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        sets = {"pretrain val": res["rollout"], **res.get("transfer", {})}
        for name, r in sets.items():
            ax[0].plot(r["horizon"], r["model"], marker=".", label=f"{name}: model")
            ax[1].plot(r["horizon"], r["model_over_copy"], marker=".", label=name)
        ax[0].plot(res["rollout"]["horizon"], res["rollout"]["copy"], "k--", label="pretrain val: copy")
        ax[0].plot(res["rollout"]["horizon"], res["rollout"]["shuffled"], "k:", label="pretrain val: shuffled actions")
        ax[0].set(xlabel="rollout step (x2 env steps)", ylabel="latent MSE", yscale="log", title="Rollout error")
        ax[1].axhline(1, color="r", lw=0.8); ax[1].set(xlabel="rollout step", ylabel="model / copy", title="Relative to copying")
        for x in ax: x.grid(alpha=0.3); x.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(a.out / "rollout.png", dpi=90)
    log(f"wrote {a.out / 'evaluation.json'}")


if __name__ == "__main__":
    main()
