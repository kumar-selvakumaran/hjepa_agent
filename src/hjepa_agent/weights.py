"""Situation classes, coverage reports and importance weights from ground truth.

Privileged (docs/journal/simulation-privileges.md SP-01, SP-03, SP-12): the classes need true positions of the agent,
obstacles, blocks and food. They shape which windows the dataloader draws; they are never a model input."""

import numpy as np
import torch

# World geometry, as built in agent_world: two 8x8 rooms side by side, a 1.0-unit doorway in the middle wall.
ARENA_X, ARENA_Y = 8.0, 4.0
DOORWAY_GAP_HALF = 0.5              # the middle wall (x = 0) is absent for |y| < 0.5
WALL_MARGIN = 0.6                   # near_wall: closer than this to any wall
OBSTACLE_CLEARANCE = 0.6            # near_obstacle_or_block: centre closer than this
FOOD_CLEARANCE = 0.5                # near_food: a present food disc closer than this
DOORWAY_X, DOORWAY_Y = 1.0, 0.7     # doorway region: |x| < 1.0 and |y| < 0.7
CORNER_DISTANCE = 1.0               # corner: within this of two walls
CELL = 1.0                          # position cell side; 16 x 8 cells
N_CELLS_X, N_CELLS_Y, N_HEADING = 16, 8, 8
WEIGHT_CAP = 10.0

CLASSES = ["open_floor", "near_wall", "near_obstacle_or_block", "doorway", "near_food", "corner"]
PRIORITY = ["corner", "doorway", "near_food", "near_obstacle_or_block", "near_wall", "open_floor"]  # highest first


def wall_distances(pos):
    """Distance from [N, 2] positions to the left, right, bottom, top and middle walls, as [N, 5]."""
    x, y = pos[:, 0], pos[:, 1]
    gap = np.maximum(DOORWAY_GAP_HALF - np.abs(y), 0.0)   # inside the doorway gap the nearest wall is a segment end
    return np.stack([x + ARENA_X, ARENA_X - x, y + ARENA_Y, ARENA_Y - y, np.hypot(x, gap)], 1)


def situation_classes(pos, objects=None, food_pos=None, food_present=None):
    """Class id (index into CLASSES) per position. pos [N,2]; objects [N,M,2] obstacle and block centres (or None);
    food_pos [N,F,2] with food_present [N,F] bool (or None). The highest-priority matching class wins."""
    d = wall_distances(pos)
    masks = {
        "corner": (d < CORNER_DISTANCE).sum(1) >= 2,
        "doorway": (np.abs(pos[:, 0]) < DOORWAY_X) & (np.abs(pos[:, 1]) < DOORWAY_Y),
        "near_wall": d.min(1) < WALL_MARGIN,
        "open_floor": np.ones(len(pos), bool),
        "near_obstacle_or_block": np.zeros(len(pos), bool),
        "near_food": np.zeros(len(pos), bool),
    }
    if objects is not None and objects.shape[1]:
        masks["near_obstacle_or_block"] = (np.linalg.norm(objects - pos[:, None], axis=-1) < OBSTACLE_CLEARANCE).any(1)
    if food_pos is not None and food_pos.shape[1]:
        masks["near_food"] = ((np.linalg.norm(food_pos - pos[:, None], axis=-1) < FOOD_CLEARANCE) & food_present).any(1)
    cls = np.zeros(len(pos), np.int64)
    for name in reversed(PRIORITY):   # lowest priority first, so higher priorities overwrite
        cls[masks[name]] = CLASSES.index(name)
    return cls


def window_features(dataset, chunk=100_000):
    """Position cell, heading bin and situation class of every window's first frame, in dataset index order."""
    cells, heads, sits = [], [], []
    for si, sh in enumerate(dataset.shards):
        sel = dataset.index[:, 0] == si
        e_all, s_all = dataset.index[sel, 1], dataset.index[sel, 2]
        apos, arot = sh.get("agent_pos"), sh.get("agent_rot")
        obst = [sh.get(k) for k in ("obstacle_pos", "block_pos") if sh.has(k)]
        objects = np.concatenate(obst, 1) if obst else None
        has_food = sh.has("food_pos") and sh.has("food_present")
        fpos, fpres = (sh.get("food_pos"), sh.get("food_present")) if has_food else (None, None)
        for lo in range(0, len(e_all), chunk):
            e, s = e_all[lo:lo + chunk], s_all[lo:lo + chunk]
            pos, rot = apos[e, s], arot[e, s]
            sits.append(situation_classes(pos, objects[e] if objects is not None else None,
                                          fpos[e, s] if has_food else None, fpres[e, s] if has_food else None))
            ix = np.clip(np.floor((pos[:, 0] + ARENA_X) / CELL), 0, N_CELLS_X - 1)
            iy = np.clip(np.floor((pos[:, 1] + ARENA_Y) / CELL), 0, N_CELLS_Y - 1)
            cells.append((ix * N_CELLS_Y + iy).astype(np.int64))
            heads.append(np.floor((np.mod(rot + np.pi, 2 * np.pi)) / (2 * np.pi) * N_HEADING).astype(np.int64) % N_HEADING)
    cat = lambda a: np.concatenate(a) if a else np.zeros(0, np.int64)
    return cat(cells), cat(heads), cat(sits)


def normalise_cap(w, cap=WEIGHT_CAP):
    """Scale w so the mean is 1 and the maximum is at most `cap` (water-filling: cap, then rescale until both hold)."""
    w = np.asarray(w, np.float64)
    w = w / w.mean()
    if w.max() <= cap:
        return w
    lo, hi = 1.0, cap / w.min()
    for _ in range(100):
        a = 0.5 * (lo + hi)
        lo, hi = (a, hi) if np.minimum(a * w, cap).mean() < 1.0 else (lo, a)
    return np.minimum(0.5 * (lo + hi) * w, cap)


def window_weights(dataset, cap=WEIGHT_CAP):
    """Importance weight per window from its first frame: product of inverse frequencies of the position cell, heading
    bin and situation class, each to the power 1/3, estimated on the TRAIN split; mean 1, capped at `cap`."""
    feats = window_features(dataset)
    train = window_features(dataset.other_split("train")) if dataset.split != "train" else feats
    w = np.ones(len(feats[0]))
    for f, ft, n in zip(feats, train, (N_CELLS_X * N_CELLS_Y, N_HEADING, len(CLASSES))):
        p = np.maximum(np.bincount(ft, minlength=n), 1) / max(len(ft), 1)
        w = w * (1.0 / p[f]) ** (1.0 / 3.0)
    return normalise_cap(w, cap)


def coverage_report(dataset, weights=None, path=None):
    """Raw and weighted histograms of position cells, heading bins, situation classes and the first transition's actions,
    plus the effective sample size. Returns a dict (with a readable "text"); writes the text to `path` if given."""
    cells, heads, sits = window_features(dataset)
    n = len(cells)
    w = np.ones(n) if weights is None else np.asarray(weights, np.float64)
    fs, acts = dataset.frameskip, []
    for si, sh in enumerate(dataset.shards):
        sel = dataset.index[:, 0] == si
        e, s = dataset.index[sel, 1], dataset.index[sel, 2]
        a = sh.get("actions")
        acts.append(np.stack([a[e, s + k] for k in range(fs)], 1))   # [n, fs, 2]
    acts = np.concatenate(acts) if acts else np.zeros((0, fs, 2))
    edges = np.linspace(-1, 1, 11)

    def both(values, bins, weight):
        raw = np.bincount(values, minlength=bins).astype(float)
        return raw.tolist(), np.bincount(values, weights=weight, minlength=bins).tolist()

    def hist(v, wt):
        return np.histogram(v, bins=edges)[0].tolist(), np.histogram(v, bins=edges, weights=wt)[0].tolist()

    wt_a = np.repeat(w, fs)
    pc_raw, pc_w = both(cells, N_CELLS_X * N_CELLS_Y, w)
    hd_raw, hd_w = both(heads, N_HEADING, w)
    sc_raw, sc_w = both(sits, len(CLASSES), w)
    fw_raw, fw_w = hist(acts[..., 0].ravel(), wt_a)
    tu_raw, tu_w = hist(acts[..., 1].ravel(), wt_a)
    ess = float(w.sum() ** 2 / (w ** 2).sum()) if n else 0.0
    rep = {"n_windows": n, "effective_sample_size": ess,
           "position_cells": {"shape": [N_CELLS_X, N_CELLS_Y], "raw": pc_raw, "weighted": pc_w},
           "heading_bins": {"raw": hd_raw, "weighted": hd_w},
           "situation_classes": {"names": CLASSES, "raw": sc_raw, "weighted": sc_w},
           "action_forward": {"bin_edges": edges.tolist(), "raw": fw_raw, "weighted": fw_w},
           "action_turn": {"bin_edges": edges.tolist(), "raw": tu_raw, "weighted": tu_w}}
    lines = [f"windows {n}; effective sample size {ess:.0f} ({ess / max(n, 1) * 100:.0f}% of windows)",
             "situation class: raw count, weighted share"]
    lines += [f"  {c:24s} {r:10.0f}  {x / max(sum(sc_w), 1e-12) * 100:5.1f}%" for c, r, x in zip(CLASSES, sc_raw, sc_w)]
    lines += ["heading bin: raw share, weighted share"]
    lines += [f"  bin {i}: {r / max(n, 1) * 100:5.1f}%  {x / max(sum(hd_w), 1e-12) * 100:5.1f}%" for i, (r, x) in enumerate(zip(hd_raw, hd_w))]
    lines += ["position cells (x by y), raw share in percent:"]
    grid = np.array(pc_raw).reshape(N_CELLS_X, N_CELLS_Y) / max(n, 1) * 100
    lines += ["  " + " ".join(f"{v:4.1f}" for v in row) for row in grid]
    rep["text"] = "\n".join(lines)
    if path is not None:
        open(path, "w", encoding="utf-8").write(rep["text"] + "\n")
    return rep


def make_sampler(dataset, weights, num_samples=None, seed=0):
    """A WeightedRandomSampler drawing windows with probability proportional to `weights`."""
    return torch.utils.data.WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                                  num_samples or len(dataset), replacement=True,
                                                  generator=torch.Generator().manual_seed(seed))
