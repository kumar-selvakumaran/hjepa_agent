"""Frame dataset over data/<name>/{frames.npy, truth.npz} for stable-pretraining. Splits are by episode.

Each item is {"image": float [3,48,64] in [0,1], <ground-truth labels>}. Labels are for probes only, never for training."""

import numpy as np
import torch

from hjepa_agent.config import DATA_DIR


def visible(img):
    """Pixel-based presence flags for the pink patch and red chaser. img: float [48,64,3] in [0,1]."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    chaser = ((r > .5) & (g < .25) & (b < .25)).sum() >= 3
    patch = ((r > .85) & (g > .45) & (g < .65) & (b > .6) & (b < .8)).sum() >= 3
    return float(chaser), float(patch)


class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, names, split="train", val_fraction=0.1, seed=0):
        if isinstance(names, str):
            names = [names]
        self.sets, self.index = [], []
        for si, name in enumerate(names):
            d = DATA_DIR / name
            frames = np.load(d / "frames.npy", mmap_mode="r")
            truth = np.load(d / "truth.npz")
            perm = np.random.RandomState(seed).permutation(frames.shape[0])
            n_val = max(1, int(len(perm) * val_fraction))
            eps = np.sort(perm[:n_val] if split == "val" else perm[n_val:])
            self.sets.append((frames, {k: truth[k] for k in truth.files}))
            self.index += [(si, int(e), t) for e in eps for t in range(frames.shape[1])]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, e, t = self.index[i]
        frames, tr = self.sets[si]
        img = torch.from_numpy(np.array(frames[e, t])).float() / 255
        pos, rot = tr["agent_pos"][e, t], float(tr["agent_rot"][e, t])
        chaser_vis, patch_vis = visible(img)
        out = {"image": img.permute(2, 0, 1), "pos_x": float(pos[0]), "pos_y": float(pos[1]),
               "heading_cos": float(np.cos(rot)), "heading_sin": float(np.sin(rot)),
               "room2": float(pos[0] > 0), "patch_visible": patch_vis, "chaser_visible": chaser_vis}
        ch = tr["chaser_pos"][e, t]
        if len(ch):
            rel = ch - pos
            r = rel[np.linalg.norm(rel, axis=-1).argmin()]
            bearing = float(np.arctan2(r[1], r[0]) - rot)
            out.update({"chaser_dist": float(np.linalg.norm(r)), "chaser_bearing_cos": float(np.cos(bearing)),
                        "chaser_bearing_sin": float(np.sin(bearing))})
        return out
