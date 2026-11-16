"""Linear probes on frozen per-frame features against ground truth, with baselines.

uv run python scripts/eval_probes.py --data pretrain_100 --features pixels|random|ckpt [--ckpt path]
Writes outputs/probes_<data>_<features>.json. Features: raw pixels (PCA 256); the LeWM frame latent (CLS token through
the projector) of a randomly initialised model; or of a trained Lightning checkpoint. Labels are privileged and for evaluation only."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hjepa_agent.config import OUTPUTS_DIR
from hjepa_agent.data import compute_stats, frame_labels, load_frames, load_stats, sample_frame_indices
from hjepa_agent.model import build_lewm
from hjepa_agent.recipe import Recipe

REG = ["pos_x", "pos_y", "heading_cos", "heading_sin"]
FLAGS = ["room2", "patch_visible", "food_visible"]
CHASER_REG, CHASER_FLAGS = ["chaser_dist", "chaser_bearing_cos", "chaser_bearing_sin"], ["chaser_visible"]


def take(names, split, n, seed, stats=None):
    idx = sample_frame_indices(names, split, n, seed=seed)
    return load_frames(names, idx, stats), frame_labels(names, split, idx)


@torch.no_grad()
def lewm_latents(x, ckpt=None, device="cuda", bs=512):
    """The LeWM frame latent of each frame; weights from a Lightning checkpoint, or random when ckpt is None."""
    m = build_lewm(Recipe()).to(device)
    if ckpt:
        sd = torch.load(ckpt, map_location=device, weights_only=False)["state_dict"]
        m.load_state_dict({k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")})
    m.eval()
    return torch.cat([m.encode({"pixels": x[i:i + bs, None].to(device)})["emb"][:, 0].float().cpu() for i in range(0, len(x), bs)]).numpy()


def collapse(f):
    ev = np.linalg.svd(f - f.mean(0), compute_uv=False) ** 2
    ev = ev / ev.sum()
    return {"participation_ratio": float(1 / (ev ** 2).sum()), "dims_for_90pct": int((ev.cumsum() < 0.9).sum()) + 1, "dim": f.shape[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--features", choices=["pixels", "random", "ckpt"], required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--stats-from", nargs="+", help="shards whose normalisation statistics to use (default: --data); use the TRAINING shards for transfer sets")
    ap.add_argument("--out", type=Path, help="result file; default outputs/probes_<data>_<features>.json")
    ap.add_argument("--train-n", type=int, default=20000)
    ap.add_argument("--val-n", type=int, default=5000)
    a = ap.parse_args()
    src = a.stats_from or a.data
    try:
        stats = None if a.features == "pixels" else load_stats(src)
    except FileNotFoundError:
        stats = compute_stats(src)
    xtr, ytr = take(a.data, "train", a.train_n, 0, stats)
    xva, yva = take(a.data, "val", a.val_n, 1, stats)
    if a.features == "pixels":
        pca = PCA(256, random_state=0).fit(xtr.flatten(1).numpy())
        ftr, fva = pca.transform(xtr.flatten(1).numpy()), pca.transform(xva.flatten(1).numpy())
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = a.ckpt if a.features == "ckpt" else None
        ftr, fva = lewm_latents(xtr, ckpt, dev), lewm_latents(xva, ckpt, dev)
    res = {"features": a.features, "data": a.data, "collapse": collapse(fva), "r2": {}, "auc": {}}
    reg = REG + [k for k in CHASER_REG if k in ytr]
    for k in reg:
        mask_tr, mask_va = (ytr["chaser_visible"], yva["chaser_visible"]) if k.startswith("chaser") else (np.ones(len(ftr), bool), np.ones(len(fva), bool))
        if np.sum(mask_tr) < 20 or np.sum(mask_va) < 5:
            continue
        p = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(ftr[mask_tr], ytr[k][mask_tr]).predict(fva[mask_va])
        res["r2"][k] = float(r2_score(yva[k][mask_va], p))
    for k in FLAGS + [k for k in CHASER_FLAGS if k in ytr]:
        if len(set(ytr[k])) < 2 or len(set(yva[k])) < 2:
            continue
        p = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(ftr, ytr[k].astype(float)).predict(fva)
        res["auc"][k] = float(roc_auc_score(yva[k], p))
    OUTPUTS_DIR.mkdir(exist_ok=True)
    out = a.out or OUTPUTS_DIR / f"probes_{'+'.join(a.data)}_{a.features}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
