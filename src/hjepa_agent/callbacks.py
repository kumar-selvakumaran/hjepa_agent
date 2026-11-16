"""Collapse guard: stop early on collapse and keep the last good checkpoint (INT-63, INT-83)."""

import math
from pathlib import Path

import lightning.pytorch as pl


class CollapseGuard(pl.Callback):
    """After each validation, a check is healthy if the SIGReg statistic is finite and below `sigreg_max`
    and the effective rank is at least `rank_min_frac` of the width. Healthy checks save `last_good.ckpt`.
    `patience` consecutive unhealthy checks stop training."""

    def __init__(self, dirpath, width, sigreg_max, rank_min_frac, patience, grace_steps):
        self.dir, self.width, self.sigreg_max = Path(dirpath), width, sigreg_max
        self.rank_min, self.patience, self.grace = rank_min_frac, patience, grace_steps
        self.bad, self.stopped_for = 0, None

    def healthy(self, sig, rank):
        return math.isfinite(sig) and math.isfinite(rank) and sig <= self.sigreg_max and rank >= self.rank_min * self.width

    def on_validation_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        if trainer.sanity_checking or "val/sigreg" not in m or trainer.global_step < self.grace:
            return
        sig, rank = float(m["val/sigreg"]), float(m["val/eff_rank"])
        if self.healthy(sig, rank):
            self.bad = 0
            self.dir.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(self.dir / "last_good.ckpt")
        else:
            self.bad += 1
            if self.bad >= self.patience:
                self.stopped_for = f"collapse: sigreg {sig:.2f} (max {self.sigreg_max}), effective rank {rank:.1f} (min {self.rank_min * self.width:.1f})"
                trainer.should_stop = True
