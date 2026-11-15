import numpy as np

from hjepa_agent import data


def _make(tmp_path, chasers):
    d = tmp_path / "toy"
    d.mkdir()
    E, T1 = 5, 4
    rng = np.random.RandomState(0)
    np.save(d / "frames.npy", rng.randint(0, 255, (E, T1, 48, 64, 3), dtype=np.uint8))
    np.savez(d / "truth.npz", agent_pos=rng.randn(E, T1, 2), agent_rot=rng.randn(E, T1),
             chaser_pos=rng.randn(E, T1, chasers, 2), actions=np.zeros((E, T1 - 1, 2)), bump=np.zeros((E, T1 - 1), bool))


def test_split_by_episode_and_item_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    _make(tmp_path, chasers=2)
    tr, va = data.FrameDataset("toy", "train"), data.FrameDataset("toy", "val")
    eps = lambda ds: {e for _, e, _ in ds.index}
    assert eps(tr).isdisjoint(eps(va)) and len(tr) + len(va) == 5 * 4
    item = tr[0]
    assert item["image"].shape == (3, 48, 64) and 0 <= item["image"].min() <= item["image"].max() <= 1
    assert {"pos_x", "heading_cos", "patch_visible", "chaser_dist"} <= set(item)


def test_no_chaser_labels_when_world_has_none(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    _make(tmp_path, chasers=0)
    assert "chaser_dist" not in data.FrameDataset("toy", "train")[0]
