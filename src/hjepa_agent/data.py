"""Window dataset over data/<name>/{frames.npy, truth.npz} (shard contract in docs/journal/stage1-spec.md section 11).

Training items hold frames and actions only. Ground-truth labels (frame_labels) are privileged and for evaluation
only: they must never be an input to the model (docs/journal/simulation-privileges.md)."""

import json
from pathlib import Path

import numpy as np
import torch

from hjepa_agent.config import DATA_DIR


class Shard:
    """One data/<name> directory. Frames are memory-mapped lazily so the object stays cheap to pickle for workers."""

    def __init__(self, name):
        self.name, self.dir = name, DATA_DIR / name
        m = np.load(self.dir / "frames.npy", mmap_mode="r")
        self.E, self.T1 = m.shape[:2]
        self.T = self.T1 - 1
        self._frames, self._truth, self._keys = m, {}, None

    @property
    def frames(self):
        if self._frames is None:
            self._frames = np.load(self.dir / "frames.npy", mmap_mode="r")
        return self._frames

    def has(self, key):
        if self._keys is None:
            self._keys = set(np.load(self.dir / "truth.npz").files)
        return key in self._keys

    def get(self, key):
        if key not in self._truth:
            self._truth[key] = np.load(self.dir / "truth.npz")[key]
        return self._truth[key]

    def __getstate__(self):
        return {**self.__dict__, "_frames": None, "_truth": {}}


def episode_split(n_episodes, split, val_fraction=0.1, seed=0):
    """Sorted episode ids of a split; the same permutation logic for every consumer."""
    perm = np.random.RandomState(seed).permutation(n_episodes)
    n_val = max(1, int(n_episodes * val_fraction))
    return np.sort(perm[:n_val] if split == "val" else perm[n_val:])


class WindowDataset(torch.utils.data.Dataset):
    """Windows of 4 frames spaced `frameskip` steps apart (a window spans 3 x frameskip steps).

    Item: {"pixels": float32 [4,3,48,64], normalised per channel with `stats` (plain x/255 when None);
           "action": float32 [4, 2*frameskip], entry i = actions[s+i*fs : s+(i+1)*fs] flattened, zero-padded past the
           end of the episode (only entry 3 can be padded, and it is unused in training);
           "index": int64 [3] = (shard, episode, start step)}."""

    def __init__(self, names, split="train", frameskip=2, stats=None, val_fraction=0.1, seed=0):
        self.names = [names] if isinstance(names, str) else list(names)
        self.split, self.frameskip, self.stats = split, frameskip, stats
        self.val_fraction, self.seed = val_fraction, seed
        self.span = 3 * frameskip
        self.shards = [Shard(n) for n in self.names]
        rows = []
        for si, sh in enumerate(self.shards):
            n_starts = sh.T - self.span + 1
            if n_starts <= 0:
                continue
            eps = episode_split(sh.E, split, val_fraction, seed)
            e, s = np.repeat(eps, n_starts), np.tile(np.arange(n_starts), len(eps))
            rows.append(np.stack([np.full_like(e, si), e, s], 1))
        self.index = np.concatenate(rows).astype(np.int64) if rows else np.zeros((0, 3), np.int64)
        mean, std = (stats["mean"], stats["std"]) if stats else ([0.0] * 3, [1.0] * 3)
        self.mean, self.std = torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)

    def other_split(self, split):
        return WindowDataset(self.names, split, self.frameskip, self.stats, self.val_fraction, self.seed)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, e, s = (int(v) for v in self.index[i])
        sh, fs = self.shards[si], self.frameskip
        fr = torch.from_numpy(np.array(sh.frames[e, s:s + self.span + 1:fs])).float() / 255
        pixels = ((fr - self.mean) / self.std).permute(0, 3, 1, 2).contiguous()
        acts, action = sh.get("actions"), np.zeros((4, 2 * fs), np.float32)
        for k in range(4):
            seg = acts[e, s + k * fs: min(s + (k + 1) * fs, sh.T)]
            action[k, :seg.size] = seg.reshape(-1)
        return {"pixels": pixels, "action": torch.from_numpy(action), "index": torch.tensor([si, e, s])}


def stats_path(names):
    names = [names] if isinstance(names, str) else list(names)
    return DATA_DIR / names[0] / "stats.json"


def compute_stats(names, split="train", n_frames=20000, seed=0, path=None):
    """Per-channel mean and std of random training frames, in [0, 1] units. Saved as JSON (default: next to shard 0)."""
    names = [names] if isinstance(names, str) else list(names)
    shards, rng = [Shard(n) for n in names], np.random.RandomState(seed)
    eps = [episode_split(sh.E, split) for sh in shards]
    sizes = np.array([len(e) * sh.T1 for e, sh in zip(eps, shards)], dtype=np.float64)
    pick = rng.choice(len(shards), n_frames, p=sizes / sizes.sum())
    total, sq, count = np.zeros(3), np.zeros(3), 0
    for si, sh in enumerate(shards):
        n = int((pick == si).sum())
        if not n:
            continue
        e, t = rng.choice(eps[si], n), rng.randint(0, sh.T1, n)
        x = np.stack([sh.frames[a, b] for a, b in zip(e, t)]).astype(np.float64) / 255
        total, sq, count = total + x.sum((0, 1, 2)), sq + (x ** 2).sum((0, 1, 2)), count + x.shape[0] * x.shape[1] * x.shape[2]
    mean = total / count
    out = {"mean": mean.tolist(), "std": np.sqrt(np.maximum(sq / count - mean ** 2, 0)).tolist()}
    path = Path(path) if path else stats_path(names)
    path.write_text(json.dumps(out))
    return out


def load_stats(source):
    """Load stats from a .json path, or from the default location of a list of shard names."""
    p = Path(source) if isinstance(source, (str, Path)) and str(source).endswith(".json") else stats_path(source)
    return json.loads(p.read_text())


def sample_frame_indices(names, split, n, seed=0, val_fraction=0.1):
    """n random (shard, episode, frame) triples from the episodes of a split, as int64 [n, 3]."""
    names = [names] if isinstance(names, str) else list(names)
    shards, rng = [Shard(x) for x in names], np.random.RandomState(seed)
    eps = [episode_split(sh.E, split, val_fraction) for sh in shards]
    sizes = np.array([len(e) * sh.T1 for e, sh in zip(eps, shards)], dtype=np.float64)
    si = rng.choice(len(shards), n, p=sizes / sizes.sum())
    e = np.array([rng.choice(eps[s]) for s in si])
    t = np.array([rng.randint(0, shards[s].T1) for s in si])
    return np.stack([si, e, t], 1).astype(np.int64)


def load_frames(names, indices, stats=None):
    """Frames for (shard, episode, frame) triples as float32 [N, 3, 48, 64]: in [0, 1], or normalised with `stats`."""
    names = [names] if isinstance(names, str) else list(names)
    shards = [Shard(x) for x in names]
    x = np.stack([shards[s].frames[e, t] for s, e, t in indices])
    x = torch.from_numpy(x).float() / 255
    if stats:
        x = (x - torch.tensor(stats["mean"])) / torch.tensor(stats["std"])
    return x.permute(0, 3, 1, 2).contiguous()


def colour_flags(frames_u8):
    """Pixel-colour presence flags (at least 3 pixels) for chaser (red), patch (pink) and food (ochre).
    frames_u8: [N, 48, 64, 3] uint8. Privileged: the renderer gives each entity a unique colour."""
    f = np.asarray(frames_u8).astype(np.float32) / 255
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    count = lambda m: m.reshape(len(f), -1).sum(1) >= 3
    return (count((r > .5) & (g < .25) & (b < .25)),
            count((r > .85) & (g > .45) & (g < .65) & (b > .6) & (b < .8)),
            count((r > .7) & (g > .5) & (g < .65) & (b < .2)))


def frame_labels(shard_names, split, indices, val_fraction=0.1):
    """Ground-truth labels for probes (evaluation only) for (shard, episode, frame) triples.

    Raises ValueError if an episode is not in `split` (pass split=None to skip the check). Chaser distance and bearing
    (relative to the nearest chaser) are present only if every involved shard has chasers."""
    names = [shard_names] if isinstance(shard_names, str) else list(shard_names)
    shards, idx = [Shard(x) for x in names], np.asarray(indices)
    n = len(idx)
    pos, rot, frames = np.zeros((n, 2), np.float32), np.zeros(n, np.float32), np.zeros((n, 48, 64, 3), np.uint8)
    used = sorted(set(idx[:, 0].tolist()))
    with_chasers = all(shards[s].has("chaser_pos") and shards[s].get("chaser_pos").shape[2] > 0 for s in used)
    near = np.zeros((n, 2), np.float32)
    for s in used:
        sel, sh = idx[:, 0] == s, shards[s]
        e, t = idx[sel, 1], idx[sel, 2]
        if split is not None and not np.isin(e, episode_split(sh.E, split, val_fraction)).all():
            raise ValueError(f"episodes outside split {split!r} in shard {sh.name}")
        pos[sel], rot[sel] = sh.get("agent_pos")[e, t], sh.get("agent_rot")[e, t]
        frames[sel] = np.stack([sh.frames[a, b] for a, b in zip(e, t)])
        if with_chasers:
            ch = sh.get("chaser_pos")[e, t]
            near[sel] = ch[np.arange(len(e)), np.linalg.norm(ch - pos[sel][:, None], axis=-1).argmin(1)]
    chaser, patch, food = colour_flags(frames)
    out = {"pos_x": pos[:, 0], "pos_y": pos[:, 1], "heading_cos": np.cos(rot), "heading_sin": np.sin(rot),
           "room2": pos[:, 0] > 0, "patch_visible": patch, "food_visible": food, "chaser_visible": chaser}
    if with_chasers:
        rel = near - pos
        bearing = np.arctan2(rel[:, 1], rel[:, 0]) - rot
        out.update({"chaser_dist": np.linalg.norm(rel, axis=-1).astype(np.float32),
                    "chaser_bearing_cos": np.cos(bearing).astype(np.float32), "chaser_bearing_sin": np.sin(bearing).astype(np.float32)})
    return out
