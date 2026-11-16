"""LeWM from stable-worldmodel, wrapped as a Lightning module. Only encode and predict are used: no goal cost, no planning (INT-11)."""

import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.utilities import grad_norm
from stable_pretraining.backbone.utils import vit_hf
from stable_pretraining.optim.lr_scheduler import LinearWarmupCosineAnnealingLR
from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.lewm.module import Embedder, MLP, Predictor
from stable_worldmodel.wm.loss import SIGReg

from hjepa_agent.recipe import Recipe


def build_lewm(r: Recipe) -> LeWM:
    enc = vit_hf("tiny", patch_size=r.patch, image_size=r.vit_image_size, use_mask_token=False)
    proj = lambda: MLP(r.embed_dim, r.proj_hidden, r.embed_dim, norm_fn=nn.BatchNorm1d)
    predictor = Predictor(num_frames=r.history, depth=r.pred_depth, heads=r.pred_heads, mlp_dim=r.pred_mlp_dim,
                          input_dim=r.embed_dim, hidden_dim=r.embed_dim, dim_head=r.pred_dim_head, dropout=r.pred_dropout)
    return LeWM(enc, predictor, Embedder(input_dim=r.action_dim, emb_dim=r.embed_dim), projector=proj(), pred_proj=proj())


def effective_rank(x: torch.Tensor) -> torch.Tensor:
    """exp of the entropy of the normalised singular values of a centred [N, D] matrix."""
    s = torch.linalg.svdvals(x - x.mean(0, keepdim=True))
    p = s / s.sum().clamp_min(1e-12)
    return torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum())


class LeWMModule(pl.LightningModule):
    def __init__(self, recipe: Recipe, total_steps: int):
        super().__init__()
        self.r, self.total_steps = recipe, total_steps
        self.model = build_lewm(recipe)
        self.sigreg = SIGReg(knots=recipe.sigreg_knots, num_proj=recipe.sigreg_proj)

    def _forward(self, batch, act=None):
        o = self.model.encode({"pixels": batch["pixels"], "action": batch["action"]})
        emb, act_emb = o["emb"], o["act_emb"] if act is None else act
        h = self.r.history
        pred = self.model.predict(emb[:, :h], act_emb[:, :h])
        return emb, act_emb, pred, emb[:, 1:h + 1]

    def training_step(self, batch, batch_idx):
        emb, _, pred, tgt = self._forward(batch)
        pred_loss = (pred - tgt).pow(2).mean()
        sig = self.sigreg(emb.float().transpose(0, 1))
        loss = pred_loss + self.r.sigreg_weight * sig
        self.log_dict({"train/loss": loss, "train/pred_loss": pred_loss, "train/sigreg": sig, "step_idx": float(self.global_step)}, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        emb, act_emb, pred, tgt = self._forward(batch)
        h = self.r.history
        pred_loss = (pred - tgt).pow(2).mean()
        copy_loss = (emb[:, :h] - tgt).pow(2).mean()  # predict "the next latent equals the current one"
        shuffled = self.model.predict(emb[:, :h], act_emb[:, :h][torch.randperm(emb.size(0), device=emb.device)])
        flat = emb.float().flatten(0, 1)
        self.log_dict({
            "val/pred_loss": pred_loss, "val/sigreg": self.sigreg(emb.float().transpose(0, 1)),
            "val/copy_ratio": pred_loss / copy_loss.clamp_min(1e-12),  # below 1 means better than copying
            "val/action_sensitivity": (shuffled - tgt).pow(2).mean() / pred_loss.clamp_min(1e-12),  # above 1: actions matter
            "val/eff_rank": effective_rank(flat), "val/dim_std": flat.std(0).mean(),
        }, on_epoch=True, sync_dist=False)

    def on_before_optimizer_step(self, optimizer):
        self.log("train/grad_norm", grad_norm(self, norm_type=2)["grad_2.0_norm_total"], on_step=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.r.lr, weight_decay=self.r.weight_decay)
        sched = LinearWarmupCosineAnnealingLR(opt, warmup_steps=int(self.r.warmup_frac * self.total_steps), max_steps=self.total_steps)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
