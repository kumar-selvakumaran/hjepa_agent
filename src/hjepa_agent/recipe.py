"""The training recipe: LeWM's values (INT-77) with our frameskip (INT-76), patch (INT-71) and monitors (INT-83, INT-86)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Recipe:
    # windows
    frameskip: int = 2
    history: int = 3  # frames the predictor reads; one more frame is predicted
    # model (LeWM)
    embed_dim: int = 192
    patch: int = 8
    vit_image_size: int = 64  # square native grid; 48x64 frames use interpolated position embeddings
    pred_depth: int = 6
    pred_heads: int = 16
    pred_mlp_dim: int = 2048
    pred_dim_head: int = 64
    pred_dropout: float = 0.1
    proj_hidden: int = 2048
    # loss (LeWM)
    sigreg_weight: float = 0.09
    sigreg_proj: int = 1024
    sigreg_knots: int = 17
    # optimisation (LeWM)
    lr: float = 5e-5
    weight_decay: float = 1e-3
    batch_size: int = 128
    epochs: int = 10
    grad_clip: float = 1.0
    precision: str = "bf16-mixed"
    seed: int = 42
    warmup_frac: float = 0.02  # fraction of steps; LeWM does not state its warmup
    # safety, monitoring (proposed thresholds, INT-83)
    ckpt_every_steps: int = 1000
    val_every_steps: int = 1000
    val_batches: int = 50
    time_cap_minutes: int = 90
    collapse_grace_steps: int = 2000
    collapse_sigreg_max: float = 5.0  # healthy floor is about 1.0 to 1.4; untrained models read 6 to 8
    collapse_rank_min_frac: float = 0.25
    collapse_patience: int = 3

    @property
    def action_dim(self) -> int:
        return 2 * self.frameskip
