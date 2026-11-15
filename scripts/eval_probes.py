"""Linear probes on frozen per-frame features against ground truth, with baselines.

uv run python scripts/eval_probes.py --data pretrain_100 --features pixels|random|ckpt [--ckpt path]
Writes outputs/probes_<data>_<features>.json. Features: raw pixels (PCA 256), a random-init timm ViT-tiny, or a
trained encoder checkpoint. Always the mean of patch tokens for ViTs."""

import argparse
import json

import numpy as np
import timm
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hjepa_agent.config import OUTPUTS_DIR
from hjepa_agent.data import FrameDataset

REG = ["pos_x", "pos_y", "heading_cos", "heading_sin"]
FLAGS = ["room2", "patch_visible"]
CHASER_REG, CHASER_FLAGS = ["chaser_dist", "chaser_bearing_cos", "chaser_bearing_sin"], ["chaser_visible"]


def take(ds, n, rng):
    items = [ds[i] for i in rng.choice(len(ds), min(n, len(ds)), replace=False)]
    return torch.stack([it["image"] for it in items]), {k: np.array([it[k] for it in items]) for k in items[0] if k != "image"}


def vit(ckpt=None, device="cuda"):
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False, img_size=(48, 64), patch_size=(6, 8), num_classes=0).to(device).eval()
    if ckpt:
        m.load_state_dict(torch.load(ckpt, map_location=device))
    return m


@torch.no_grad()
def vit_features(m, x, device="cuda", bs=512):
    out = [m.forward_features(x[i:i + bs].to(device))[:, m.num_prefix_tokens:].mean(1).cpu() for i in range(0, len(x), bs)]
    return torch.cat(out).numpy()


def collapse(f):
    ev = np.linalg.svd(f - f.mean(0), compute_uv=False) ** 2
    ev = ev / ev.sum()
    return {"participation_ratio": float(1 / (ev ** 2).sum()), "dims_for_90pct": int((ev.cumsum() < 0.9).sum()) + 1, "dim": f.shape[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--features", choices=["pixels", "random", "ckpt"], required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--train-n", type=int, default=20000)
    ap.add_argument("--val-n", type=int, default=5000)
    a = ap.parse_args()
    rng = np.random.RandomState(0)
    xtr, ytr = take(FrameDataset(a.data, "train"), a.train_n, rng)
    xva, yva = take(FrameDataset(a.data, "val"), a.val_n, rng)
    if a.features == "pixels":
        pca = PCA(256, random_state=0).fit(xtr.flatten(1).numpy())
        ftr, fva = pca.transform(xtr.flatten(1).numpy()), pca.transform(xva.flatten(1).numpy())
    else:
        m = vit(a.ckpt if a.features == "ckpt" else None)
        ftr, fva = vit_features(m, xtr), vit_features(m, xva)
    res = {"features": a.features, "data": a.data, "collapse": collapse(fva), "r2": {}, "auc": {}}
    reg = REG + [k for k in CHASER_REG if k in ytr]
    for k in reg:
        mask_tr, mask_va = (ytr["chaser_visible"] > 0, yva["chaser_visible"] > 0) if k.startswith("chaser") else (slice(None), slice(None))
        if np.sum(mask_tr) < 20 or np.sum(mask_va) < 5:
            continue
        p = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(ftr[mask_tr], ytr[k][mask_tr]).predict(fva[mask_va])
        res["r2"][k] = float(r2_score(yva[k][mask_va], p))
    for k in FLAGS + [k for k in CHASER_FLAGS if k in ytr]:
        if len(set(ytr[k])) < 2 or len(set(yva[k])) < 2:
            continue
        p = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(ftr, ytr[k]).predict(fva)
        res["auc"][k] = float(roc_auc_score(yva[k] > 0, p))
    OUTPUTS_DIR.mkdir(exist_ok=True)
    out = OUTPUTS_DIR / f"probes_{'+'.join(a.data)}_{a.features}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
