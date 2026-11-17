import numpy as np
import torch

from hjepa_agent import evaluation as ev
from hjepa_agent.model import LeWMModule
from hjepa_agent.recipe import Recipe

SMALL = Recipe(pred_depth=1, pred_heads=2, pred_mlp_dim=64, pred_dim_head=16, proj_hidden=64, sigreg_proj=32)


def test_rollout_errors_cover_every_horizon_and_baselines():
    m = LeWMModule(SMALL, total_steps=1).eval()
    px, ac = torch.rand(6, 7, 3, 48, 64), torch.randn(6, 7, 4)
    r = ev.rollout_errors(m, px, ac)
    assert r["horizon"] == [1, 2, 3, 4]
    assert all(len(r[k]) == 4 for k in ("model", "copy", "shuffled", "model_over_copy", "shuffled_over_model"))
    assert all(np.isfinite(r[k]).all() for k in ("model", "copy", "shuffled"))


def test_layer_features_shapes():
    m = LeWMModule(SMALL, total_steps=1).eval()
    f = ev.layer_features(m, torch.rand(5, 3, 48, 64))
    assert f["cls"].shape == (5, 192) and f["projected"].shape == (5, 192) and f["patch_mean"].shape == (5, 192)
    assert f["patch_all"].shape == (5, 48 * 192)


def test_identifiable_targets_and_probe():
    th = np.random.RandomState(0).uniform(-np.pi, np.pi, 400)
    t = ev.identifiable_targets({"heading_cos": np.cos(th), "heading_sin": np.sin(th), "pos_x": th, "pos_y": -th})
    assert np.allclose(t["cos4h"], np.cos(4 * th)) and np.allclose(t["abs_pos_y"], np.abs(th))
    x = np.stack([np.cos(4 * th), np.sin(4 * th)], 1)
    assert ev.probe(x[:300], t["cos4h"][:300], x[300:], t["cos4h"][300:], "ridge") > 0.99
