import math
from types import SimpleNamespace

import torch

from hjepa_agent.callbacks import CollapseGuard
from hjepa_agent.model import LeWMModule, effective_rank
from hjepa_agent.recipe import Recipe

SMALL = Recipe(pred_depth=1, pred_heads=2, pred_mlp_dim=64, pred_dim_head=16, proj_hidden=64, sigreg_proj=32)


def batch(b=4, fs=2):
    return {"pixels": torch.rand(b, 4, 3, 48, 64), "action": torch.randn(b, 4, 2 * fs)}


def test_training_and_validation_steps_are_finite_and_log_monitors():
    m = LeWMModule(SMALL, total_steps=10)
    logged = {}
    m.log_dict = lambda d, **k: logged.update({a: float(b) for a, b in d.items()})
    loss = m.training_step(batch(), 0)
    assert torch.isfinite(loss) and {"train/pred_loss", "train/sigreg"} <= set(logged)
    loss.backward()
    m.eval()
    m.validation_step(batch(), 0)
    assert {"val/copy_ratio", "val/action_sensitivity", "val/eff_rank", "val/sigreg"} <= set(logged)
    assert all(math.isfinite(v) for v in logged.values())


def test_effective_rank_distinguishes_collapse():
    assert effective_rank(torch.randn(512, 32)) > 25
    assert effective_rank(torch.ones(512, 32) + 1e-4 * torch.randn(512, 1) * torch.ones(1, 32)) < 2


def test_collapse_guard_saves_when_healthy_and_stops_after_patience(tmp_path):
    g = CollapseGuard(tmp_path, width=192, sigreg_max=5.0, rank_min_frac=0.25, patience=3, grace_steps=0)
    saved = []
    tr = SimpleNamespace(sanity_checking=False, global_step=10, should_stop=False, callback_metrics={}, save_checkpoint=lambda p: saved.append(p))
    for sig, rank, expect_stop in [(1.2, 100.0, False), (9.0, 100.0, False), (9.0, 100.0, False), (9.0, 100.0, True)]:
        tr.callback_metrics = {"val/sigreg": torch.tensor(sig), "val/eff_rank": torch.tensor(rank)}
        g.on_validation_end(tr, None)
        assert tr.should_stop == expect_stop
    assert len(saved) == 1 and g.stopped_for.startswith("collapse")
    assert not g.healthy(float("nan"), 100.0) and not g.healthy(1.0, 10.0)
