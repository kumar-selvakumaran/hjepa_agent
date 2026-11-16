import json

import numpy as np
import pytest
import torch

from hjepa_agent import data, weights


def make_shard(root, name="toy", E=10, T=30, seed=0, pos=None):
    d = root / name
    d.mkdir(parents=True)
    rng = np.random.RandomState(seed)
    pos = rng.uniform([-7.5, -3.5], [7.5, 3.5], (E, T + 1, 2)).astype(np.float32) if pos is None else pos
    np.save(d / "frames.npy", np.zeros((E, T + 1, 48, 64, 3), np.uint8))
    np.savez(d / "truth.npz", agent_pos=pos, agent_rot=rng.uniform(-3.1, 3.1, (E, T + 1)).astype(np.float32),
             actions=rng.uniform(-1, 1, (E, T, 2)).astype(np.float32), bump=np.zeros((E, T), bool),
             chaser_pos=np.zeros((E, T + 1, 0, 2), np.float32), obstacle_pos=rng.uniform(-6, 6, (E, 3, 2)).astype(np.float32),
             obstacle_height=np.ones((E, 3), np.float32), block_pos=rng.uniform(-6, 6, (E, 2, 2)).astype(np.float32),
             food_pos=rng.uniform(-6, 6, (E, T + 1, 4, 2)).astype(np.float32), food_present=np.ones((E, T + 1, 4), bool),
             walk_scale=np.ones(E, np.float32), walk_p=np.ones(E, np.float32))
    (d / "config.json").write_text(json.dumps({}))


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    return tmp_path


def cls(pos, objects=None, food=None, present=None):
    c = lambda a: None if a is None else np.asarray(a, np.float32)[None]
    return weights.CLASSES[weights.situation_classes(np.array([pos], np.float32), c(objects), c(food), None if present is None else np.asarray(present)[None])[0]]


def test_situation_class_priority():
    assert cls([-3.0, 1.5]) == "open_floor"
    assert cls([-7.6, 0.0]) == "near_wall"
    assert cls([7.8, 3.8]) == "corner"                                    # within 1.0 of two walls
    assert cls([0.2, 0.1]) == "doorway"
    assert cls([0.2, 3.8]) == "corner"                                    # middle wall plus top wall
    assert cls([-6.0, 3.7], objects=[[-6.0, 3.5]]) == "near_obstacle_or_block"   # beats near_wall
    assert cls([3.0, 0.0], objects=[[3.0, 0.1]], food=[[3.2, 0.0]], present=[True]) == "near_food"   # beats obstacle
    assert cls([3.0, 0.0], food=[[3.2, 0.0]], present=[False]) == "open_floor"                        # absent food ignored
    assert cls([0.3, 0.2], food=[[0.4, 0.2]], present=[True]) == "doorway"                           # doorway beats food
    assert cls([7.8, 3.8], objects=[[7.8, 3.7]]) == "corner"                                          # corner beats all


def test_normalise_cap_mean_one_and_capped():
    w = np.ones(1000)
    w[0] = 1e6
    out = weights.normalise_cap(w, cap=10.0)
    assert np.isclose(out.mean(), 1.0, atol=1e-6) and out.max() <= 10.0 + 1e-9
    plain = weights.normalise_cap(np.array([1.0, 2.0, 3.0]), 10.0)
    assert np.isclose(plain.mean(), 1.0)


def test_window_weights_upweight_rare_cells(root):
    pos = np.full((10, 31, 2), -4.5, np.float32)
    pos[:, :, 1] = 1.5                                                    # almost everything in one cell
    pos[0, :, :] = [6.5, -2.5]                                            # one episode elsewhere
    make_shard(root, pos=pos)
    ds = data.WindowDataset("toy", "train", 2)
    w = weights.window_weights(ds)
    assert np.isclose(w.mean(), 1.0, atol=1e-6) and w.max() <= weights.WEIGHT_CAP + 1e-9 and len(w) == len(ds)
    rare = ds.index[:, 1] == 0
    if rare.any():
        assert w[rare].mean() > w[~rare].mean()
    val = data.WindowDataset("toy", "val", 2)
    wv = weights.window_weights(val)                                      # val windows use train-split frequencies
    assert len(wv) == len(val) and np.isclose(wv.mean(), 1.0, atol=1e-6)


def test_coverage_report_and_sampler(root, tmp_path):
    make_shard(root)
    ds = data.WindowDataset("toy", "train", 2)
    w = weights.window_weights(ds)
    out = tmp_path / "coverage.txt"
    rep = weights.coverage_report(ds, w, path=out)
    assert rep["n_windows"] == len(ds) and 0 < rep["effective_sample_size"] <= len(ds)
    assert out.exists() and "situation class" in out.read_text()
    assert abs(sum(rep["situation_classes"]["weighted"]) - w.sum()) < 1e-6 and sum(rep["situation_classes"]["raw"]) == len(ds)
    assert np.isclose(sum(rep["action_forward"]["raw"]), len(ds) * 2)
    flat = weights.coverage_report(ds)
    assert np.isclose(flat["effective_sample_size"], len(ds))
    sampler = weights.make_sampler(ds, w, num_samples=50, seed=3)
    drawn = list(sampler)
    assert len(drawn) == 50 and all(0 <= i < len(ds) for i in drawn)
    assert torch.is_tensor(next(iter(torch.utils.data.DataLoader(ds, batch_size=4, sampler=sampler)))["pixels"])
