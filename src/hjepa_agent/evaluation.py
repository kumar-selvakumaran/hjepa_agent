"""Offline evaluation of a trained world model (gaps G1, G2, G4, G5 in docs/journal/world-model-gaps.md).

- Rollout error by horizon: the predictor fed its own predictions, against copying the last latent and shuffled actions.
- Layer features: all patch tokens, mean patch token, CLS token, projected latent (where information is kept or lost).
- Identifiable probe targets: cos/sin of 4 x heading, |pos_y|, pos_x, room; plus absolute heading for reference.
- History features: three consecutive projected latents plus their actions (can a short history resolve absolute heading?).
Labels are privileged and used only for evaluation (docs/journal/simulation-privileges.md)."""

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hjepa_agent.data import Shard, episode_split, frame_labels
from hjepa_agent.model import LeWMModule
from hjepa_agent.recipe import Recipe


def load_module(ckpt, device="cuda", recipe=Recipe()):
    m = LeWMModule(recipe, total_steps=1)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"])
    return m.to(device).eval()


def episode_sequences(names, split, n, length, frameskip=2, stats=None, seed=0):
    """n random sequences of `length` frames spaced `frameskip` steps apart, from episodes of `split`.
    Returns pixels float32 [n, length, 3, 48, 64], actions float32 [n, length, 2 * frameskip] (entry k covers the steps
    from frame k to frame k + 1; the last entry is zero), and index int64 [n, 3] = (shard, episode, first step)."""
    names = [names] if isinstance(names, str) else list(names)
    shards, rng = [Shard(x) for x in names], np.random.RandomState(seed)
    span = (length - 1) * frameskip
    eps = [episode_split(sh.E, split) for sh in shards]
    idx = []
    for _ in range(n):
        si = rng.randint(len(shards))
        idx.append((si, int(rng.choice(eps[si])), rng.randint(0, shards[si].T - span)))
    mean, std = (stats["mean"], stats["std"]) if stats else ([0.0] * 3, [1.0] * 3)
    mean, std = torch.tensor(mean).view(1, 1, 3), torch.tensor(std).view(1, 1, 3)
    pix, act = [], np.zeros((n, length, 2 * frameskip), np.float32)
    for j, (si, e, s) in enumerate(idx):
        sh = shards[si]
        fr = torch.from_numpy(np.array(sh.frames[e, s:s + span + 1:frameskip])).float() / 255
        pix.append(((fr - mean) / std).permute(0, 3, 1, 2))
        a = sh.get("actions")[e]
        for k in range(length - 1):
            act[j, k] = a[s + k * frameskip: s + (k + 1) * frameskip].reshape(-1)
    return torch.stack(pix), torch.from_numpy(act), np.array(idx, np.int64)


@torch.no_grad()
def encode(m, pixels, actions, bs=64):
    """Projected latents [n, L, D] and action embeddings [n, L, A] for sequences."""
    dev = next(m.parameters()).device
    z, a = [], []
    for i in range(0, len(pixels), bs):
        o = m.model.encode({"pixels": pixels[i:i + bs].to(dev), "action": actions[i:i + bs].to(dev)})
        z.append(o["emb"].float()), a.append(o["act_emb"])
    return torch.cat(z), torch.cat(a)


@torch.no_grad()
def rollout_errors(m, pixels, actions, history=3, seed=0):
    """Autoregressive rollout from `history` true latents, for every remaining frame of the sequences.
    Returns per horizon h = 1..L-history: mean squared error of the rollout, of copying the last true context latent,
    and of the rollout with actions shuffled across sequences; plus their ratios."""
    z, a = encode(m, pixels, actions)
    n, L, _ = z.shape
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).to(z.device)
    out = {"horizon": [], "model": [], "copy": [], "shuffled": []}
    for name, act in (("model", a), ("shuffled", a[perm])):
        ctx = z[:, :history]
        for t in range(L - history):
            pred = m.model.predict(ctx, act[:, t:t + history]).float()[:, -1]
            err = (pred - z[:, history + t]).pow(2).mean().item()
            out[name].append(err)
            ctx = torch.cat([ctx[:, 1:], pred[:, None]], 1)
    for t in range(L - history):
        out["horizon"].append(t + 1)
        out["copy"].append((z[:, history - 1] - z[:, history + t]).pow(2).mean().item())
    out["model_over_copy"] = [x / max(y, 1e-12) for x, y in zip(out["model"], out["copy"])]
    out["shuffled_over_model"] = [x / max(y, 1e-12) for x, y in zip(out["shuffled"], out["model"])]
    return out


@torch.no_grad()
def layer_features(m, frames, bs=256):
    """Per-frame features at four points of the encoder path, for frames float32 [N, 3, 48, 64]."""
    dev, out = next(m.parameters()).device, {"patch_all": [], "patch_mean": [], "cls": [], "projected": []}
    for i in range(0, len(frames), bs):
        h = m.model.encoder(frames[i:i + bs].to(dev), interpolate_pos_encoding=True).last_hidden_state.float()
        out["patch_all"].append(h[:, 1:].flatten(1).half().cpu())
        out["patch_mean"].append(h[:, 1:].mean(1).cpu())
        out["cls"].append(h[:, 0].cpu())
        out["projected"].append(m.model.projector(h[:, 0]).float().cpu())
    return {k: torch.cat(v).float().numpy() for k, v in out.items()}


def identifiable_targets(labels):
    """Targets readable from one frame in this world (iteration 2), plus absolute heading for reference."""
    th = np.arctan2(labels["heading_sin"], labels["heading_cos"])
    return {"cos4h": np.cos(4 * th), "sin4h": np.sin(4 * th), "abs_pos_y": np.abs(labels["pos_y"]),
            "pos_x": labels["pos_x"], "heading_cos": labels["heading_cos"], "heading_sin": labels["heading_sin"]}


def probe(ftr, ytr, fva, yva, kind="ridge", seed=0):
    """R2 of a ridge or MLP probe trained on (ftr, ytr) and scored on (fva, yva)."""
    model = Ridge(1.0) if kind == "ridge" else MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=200,
                                                            early_stopping=True, random_state=seed)
    return float(r2_score(yva, make_pipeline(StandardScaler(), model).fit(ftr, ytr).predict(fva)))


def room_auc(ftr, ytr, fva, yva):
    p = make_pipeline(StandardScaler(), Ridge(1.0)).fit(ftr, ytr.astype(float)).predict(fva)
    return float(roc_auc_score(yva, p))


def reduce(ftr, fva, dims=256, seed=0):
    """PCA to `dims` dimensions, fitted on the training features."""
    pca = PCA(dims, random_state=seed).fit(ftr)
    return pca.transform(ftr), pca.transform(fva)


def history_features(m, names, split, n, stats, seed=0, history=3):
    """Concatenated projected latents and actions of `history` consecutive frames, with labels of the last frame."""
    pixels, actions, idx = episode_sequences(names, split, n, history, stats=stats, seed=seed)
    z, _ = encode(m, pixels, actions)
    feats = np.concatenate([z.flatten(1).cpu().numpy(), actions[:, :history - 1].flatten(1).numpy()], 1)
    last = idx.copy()
    last[:, 2] += (history - 1) * 2
    return feats, frame_labels(names, split, last)
