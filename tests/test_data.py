import json

import numpy as np
import pytest
import torch

from hjepa_agent import data


def make_shard(root, name="toy", E=10, T=20, C=2, K=3, B=2, F=4, seed=0, constant=None):
    d = root / name
    d.mkdir(parents=True)
    rng = np.random.RandomState(seed)
    frames = np.full((E, T + 1, 48, 64, 3), constant, np.uint8) if constant is not None else rng.randint(0, 255, (E, T + 1, 48, 64, 3), dtype=np.uint8)
    np.save(d / "frames.npy", frames)
    actions = np.zeros((E, T, 2), np.float32)
    actions[..., 0] = np.arange(E)[:, None] * 100 + np.arange(T)[None]   # unique, recognisable values
    actions[..., 1] = -actions[..., 0]
    np.savez(d / "truth.npz", agent_pos=rng.uniform(-3, 3, (E, T + 1, 2)).astype(np.float32), agent_rot=rng.uniform(-3, 3, (E, T + 1)).astype(np.float32),
             actions=actions, bump=np.zeros((E, T), bool), chaser_pos=rng.uniform(-3, 3, (E, T + 1, C, 2)).astype(np.float32),
             obstacle_pos=rng.uniform(-3, 3, (E, K, 2)).astype(np.float32), obstacle_height=np.ones((E, K), np.float32),
             block_pos=rng.uniform(-3, 3, (E, B, 2)).astype(np.float32), food_pos=rng.uniform(-3, 3, (E, T + 1, F, 2)).astype(np.float32),
             food_present=np.ones((E, T + 1, F), bool), walk_scale=np.ones(E, np.float32), walk_p=np.ones(E, np.float32))
    (d / "config.json").write_text(json.dumps({"seed": seed}))
    return frames, actions


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    return tmp_path


def test_item_shapes_dtypes_and_normalisation(root):
    frames, _ = make_shard(root)
    stats = {"mean": [0.5, 0.4, 0.3], "std": [0.25, 0.2, 0.1]}
    ds = data.WindowDataset("toy", "train", frameskip=2, stats=stats)
    it = ds[7]
    si, e, s = it["index"].tolist()
    assert it["pixels"].shape == (4, 3, 48, 64) and it["pixels"].dtype == torch.float32
    assert it["action"].shape == (4, 4) and it["action"].dtype == torch.float32 and it["index"].dtype == torch.int64
    for k in range(4):
        want = (frames[e, s + 2 * k].astype(np.float32) / 255 - np.array(stats["mean"])) / np.array(stats["std"])
        assert np.allclose(it["pixels"][k].permute(1, 2, 0).numpy(), want, atol=1e-5)
    plain = data.WindowDataset("toy", "train", frameskip=2)[7]["pixels"]
    assert np.allclose(plain[0].permute(1, 2, 0).numpy(), frames[e, s].astype(np.float32) / 255, atol=1e-6)


def test_split_is_by_episode_and_window_counts(root):
    make_shard(root, E=10, T=20)
    tr, va = data.WindowDataset("toy", "train", 2), data.WindowDataset("toy", "val", 2)
    eps = lambda ds: set(ds.index[:, 1].tolist())
    assert eps(tr).isdisjoint(eps(va)) and len(eps(tr)) == 9 and len(eps(va)) == 1
    per_episode = 20 - 3 * 2 + 1
    assert len(tr) == 9 * per_episode and len(va) == 1 * per_episode
    assert tr.index[:, 2].max() == 20 - 3 * 2 and len(data.WindowDataset("toy", "train", 5)) == 9 * (20 - 15 + 1)


def test_action_concatenation_order_and_zero_padding(root):
    _, actions = make_shard(root, T=20)
    ds = data.WindowDataset("toy", "train", frameskip=2)
    last = np.where((ds.index[:, 2] == 14))[0][0]            # s = T - 3*fs: entry 3 starts exactly at T
    mid = np.where((ds.index[:, 2] == 13))[0][0]             # entry 3 has one real action then padding
    first = np.where((ds.index[:, 2] == 3))[0][0]
    e = int(ds.index[first, 1])
    a = ds[first]["action"].numpy()
    for k in range(4):                                       # entry k = actions s+2k, s+2k+1, each (forward, turn)
        want = np.concatenate([actions[e, 3 + 2 * k], actions[e, 3 + 2 * k + 1]])
        assert np.allclose(a[k], want)
    assert np.allclose(ds[last]["action"].numpy()[3], 0)
    e_mid = int(ds.index[mid, 1])
    assert np.allclose(ds[mid]["action"].numpy()[3], [actions[e_mid, 19, 0], actions[e_mid, 19, 1], 0, 0])
    assert np.allclose(ds[last]["action"].numpy()[:3], ds[last]["action"].numpy()[:3])


def test_dataloader_batches(root):
    make_shard(root)
    dl = torch.utils.data.DataLoader(data.WindowDataset("toy", "train", 2), batch_size=5, shuffle=True)
    b = next(iter(dl))
    assert b["pixels"].shape == (5, 4, 3, 48, 64) and b["action"].shape == (5, 4, 4) and b["index"].shape == (5, 3)


def test_compute_and_load_stats(root):
    make_shard(root, constant=51)                            # every pixel 51/255 = 0.2
    st = data.compute_stats("toy", n_frames=50)
    assert np.allclose(st["mean"], 0.2, atol=1e-6) and np.allclose(st["std"], 0, atol=1e-6)
    assert data.load_stats(["toy"]) == st and data.load_stats(root / "toy" / "stats.json") == st
    out = root / "custom.json"
    data.compute_stats("toy", n_frames=10, path=out)
    assert out.exists()


def test_frame_labels_and_flags(root):
    frames, _ = make_shard(root, E=10, T=20, C=1, constant=0)
    tr = np.load(root / "toy" / "truth.npz")
    truth = {k: tr[k] for k in tr.files}
    e = int(data.episode_split(10, "train")[0])
    frames = np.load(root / "toy" / "frames.npy")
    frames[e, 0, :3, :3] = [230, 20, 20]                     # chaser red
    frames[e, 1, :3, :3] = [245, 140, 180]                   # patch pink
    frames[e, 2, :3, :3] = [204, 148, 30]                    # food ochre
    np.save(root / "toy" / "frames.npy", frames)
    truth["agent_pos"][e, 0], truth["agent_rot"][e, 0] = [1.0, 2.0], 0.0
    truth["chaser_pos"][e, 0, 0] = [4.0, 2.0]                # 3 units straight ahead
    np.savez(root / "toy" / "truth.npz", **truth)
    idx = np.array([[0, e, 0], [0, e, 1], [0, e, 2], [0, e, 3]])
    lab = data.frame_labels("toy", "train", idx)
    assert lab["chaser_visible"].tolist() == [True, False, False, False]
    assert lab["patch_visible"].tolist() == [False, True, False, False]
    assert lab["food_visible"].tolist() == [False, False, True, False]
    assert np.isclose(lab["pos_x"][0], 1.0) and np.isclose(lab["pos_y"][0], 2.0) and lab["room2"][0]
    assert np.isclose(lab["heading_cos"][0], 1.0) and np.isclose(lab["heading_sin"][0], 0.0)
    assert np.isclose(lab["chaser_dist"][0], 3.0) and np.isclose(lab["chaser_bearing_cos"][0], 1.0) and np.isclose(lab["chaser_bearing_sin"][0], 0.0)
    with pytest.raises(ValueError):
        data.frame_labels("toy", "val", idx)                 # episode e is in the train split


def test_no_chaser_labels_without_chasers(root):
    make_shard(root, C=0, constant=0)
    idx = data.sample_frame_indices("toy", "train", 6, seed=1)
    lab = data.frame_labels("toy", "train", idx)
    assert "chaser_dist" not in lab and not lab["chaser_visible"].any() and idx.shape == (6, 3)
    x = data.load_frames("toy", idx)
    assert x.shape == (6, 3, 48, 64) and 0 <= x.min() and x.max() <= 1
