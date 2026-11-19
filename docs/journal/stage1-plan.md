# Stage 1: perception pre-training (plan)

Prepared 2026-11-15. Labels: see [README.md](README.md). Strategy B from [training-strategies.md](training-strategies.md): pre-train the encoder, then freeze it for the world-model stage.

## 1. Principle applied: reuse before writing

- **[Kumar]** Keep from-scratch code to a minimum; use out-of-the-box functionality when it gives a pipeline with little effort; be critical about adding code.
- **[Agreed]** This replaces the earlier default "port the LeWM core ourselves" (Q1 of [agent-proposal.md](agent-proposal.md)).
- Installed and checked: `stable-pretraining` 0.1.8 (MIT; Python 3.13 and torch 2.11 work), `stable-worldmodel` 0.1.1 (its PyPI release includes `wm/lewm`, `wm/pldm`, `wm/prejepa`, a planning module and solvers).

## 2. Reuse map

| Need | Reuse | Check | Our code |
| --- | --- | --- | --- |
| Masked latent prediction (I-JEPA) | `stable_pretraining.methods.IJEPA`, EMA teacher, `TeacherStudentCallback` | **[Verified]** runs on 48x64 frames with a timm ViT-tiny and 6x8-pixel patches (8x8 grid, 64 tokens); 3x4 patches give 256 tokens | none |
| Training loop, logging, checkpoints | Lightning `Trainer` and `spt.Manager` | **[Verified]** imports | a short script |
| Collapse monitor | `spt.callbacks.RankMe` | not yet run | none |
| Online probe against ground truth | `spt.callbacks.OnlineProbe` (any loss, so MSE regression works) | not yet run | label construction from the world's truth |
| Offline probes (R2, AUC) | scikit-learn `Ridge`, `roc_auc_score` | installed with `spt` | about 30 lines |
| World model, stage 2 | `stable_worldmodel.wm.lewm.LeWM`, `wm/loss.py` (SIGReg), `wm/lewm/module.py` (autoregressive predictor) | read; needs an adapter for our encoder (it expects a HuggingFace ViT output) | an adapter, about 10 lines |
| Planning, later | `stable_worldmodel.planning` solvers (CEM and others) | needs a Gymnasium-style environment; our world is VMAS | an adapter, later |
| World and data | `agent_world` as a git dependency (pinned in `uv.lock`) | **[Verified]** | `policies.py`, `generate_data.py` (domain-specific; nothing off the shelf) |

Removed on this principle: a hand-written ridge/AUC/collapse module with tests, and a frame-dataset class (both uncommitted drafts).

## 3. Design decisions for stage 1

- **[Proposed]** Method: I-JEPA as shipped. It has an EMA teacher, which settles my earlier concern C1 (inventing masked prediction plus SIGReg) by using the established recipe. LeJEPA (SIGReg, multi-view) is also in the package but needs multi-view augmentation; flips would reverse the patch's spin direction and crops break the fixed camera geometry, so it fits less well.
- **[Proposed]** Encoder: timm ViT-tiny at 48x64, patch 6x8 (64 tokens) first; patch 3x4 (256 tokens) if small entities are not recoverable.
- **[Proposed]** No augmentation. The view is deterministic and geometry-bearing.
- **[Proposed]** Stage-2 input: the per-frame embedding is the mean of patch tokens, as in `spt`'s own example; LeWM expects a vector per frame.
- **[Proposed]** Exit gate: linear probes on frozen features recover heading, position, room, chaser visibility, chaser distance and bearing, and patch visibility, compared with a raw-pixel baseline and a random-init encoder; embedding rank is not collapsed.

## 4. Pilot data

Waypoint policy (targets: a random point in the room, through the doorway, the nearest chaser, anywhere), 512 episodes x 600 steps = 307,200 transitions, 2.84 GB raw, generated in 443 s at 700 world-steps/s on the local GPU.

| Measure | Random walk (200 steps) | Waypoint policy (600 steps) |
| --- | --- | --- |
| Episodes reaching room 2 | 1.6% | **40%** |
| Doorway crossings | not counted | 463 (0.90 per episode) |
| Bumps | 0.07% of steps | **0.50% of steps**; 82% of episodes have at least one |
| Nearest chaser within 1 unit | 5% | 33% |
| Revisits to a 0.5-unit cell | not measured | 1.4% of steps |
| Mean distance moved per episode | 10.9 | 27.3 of 60 possible |

- Batch size matters: world-steps/s were 143 (64 worlds), 315 (256), 541 (512), and 700 (512 worlds, waypoint policy, 600 steps); 1M transitions now takes about 24 minutes.
- **[Open]** Chaser proximity at 33% may be heavier than wanted; the mix of target types is a knob.

## 5. What remains to build for stage 1

- **Dataset adapter** for `spt`: a torch Dataset over `frames.npy` returning `{"image": ...}` plus ground-truth labels. About 40 lines.
- **Training script:** `IJEPA` + `Trainer` + callbacks. About 50 lines.
- **Eval script:** sklearn probes on frozen features, with baselines. About 30 lines.

## 6. Questions, concerns, blockers

| ID | Item | Default |
| --- | --- | --- |
| Q1 | Dataset size for stage 1: how many frames? | 1M transitions (about 24 min to generate), 10% of episodes held out |
| Q2 | Patch 6x8 (64 tokens) or 3x4 (256 tokens) first? | 6x8 |
| Q3 | Keep the chaser-heavy waypoint mix, or reduce it? | Keep for now |
| Q4 | Stage 2 reads our data how: `swm`'s HDF5 format, or our own loader? | Our own thin loader first; consider `swm` HDF5 if it buys the dataloader |
| C1 | `stable-pretraining` pulls a large dependency set (wandb, transformers, datasets, lancedb and others) | Accept; it is the price of reuse |
| C2 | `spt`'s I-JEPA predictor needs a square patch grid, so 48x64 needs non-square patches | Use 6x8 or 3x4 patches |
| C3 | The LeWM class expects a HuggingFace ViT output and a CLS token; I-JEPA does not train a CLS token | Adapter returning the mean-pooled patch tokens |
| C4 | Pre-training on single frames sees no time or actions (concern C2 of the strategies doc) | Gate stage 2 on the probes; add an action-aware term only if they fail |

## 7. Decisions, 2026-11-15 (appended; earlier text kept)

- **[Kumar]** Q1: 1M transitions for stage 1, 10% of episodes held out. **[Agreed]**
- **[Kumar]** Q2: patch 6x8 pixels (64 tokens) first. **[Agreed]**
- **[Kumar]** Q3: no bumps in pretraining. Principle: use data that would be available in real life; the adversarial bump is meant to be a critical, unexpected obstacle. **[Agreed]**
- **[Kumar]** Chasers are **off** in pretraining data, so chasers first appear at a later stage as a novel obstacle. **[Agreed]**
- **[Kumar]** Q4: thin loader first for stage 2. **[Agreed]**
- **[Built]** The waypoint policy no longer seeks chasers: targets are the current room (0.5), through the doorway (0.3), anywhere (0.2).

### Consequences

- **[Proposed]** Pretraining data: `pretrain_100` to `pretrain_103`, each 512 episodes x 500 steps, chasers off, 1,024,000 transitions over 2,048 randomised layouts.
- **[Proposed]** A separate `transfer_chasers` set (512 x 300, chasers on) is never used for pretraining. It tests whether the frozen encoder, which has never seen a chaser, still represents one: linear probes for chaser visibility, distance and bearing are trained on it.
- **[Proposed]** The stage-1 exit gate drops chaser probes. It keeps heading, position, room and patch visibility, a raw-pixel and a random-encoder baseline, and the rank check. Chaser probes become a transfer measurement, not a gate.
- **[Open]** Whether to also hold out some layout factor for transfer.

## 8. Amendment, 2026-11-15 (appended; earlier text kept)

- **[Kumar]** Planning is kept out of the world model entirely, and random walks are an explicit design decision: the world model only models the world. See INT-11 and INT-22 in [intentions.md](intentions.md).
- **[Superseded]** The waypoint policy of sections 4 and 7 (a goal-seeking, ground-truth-steered data generator) is removed. The pilot results in section 4 describe that policy and no longer apply to the data plan.
- **[Kumar]** Further data decisions: the turning patch is removed entirely in pretraining; random walks are dispersed throughout the world (uniform free-space start poses in both rooms, random heading) and include stopping, reversing and varied speeds; 2,048 episodes x 500 steps; world content at defaults; per-channel normalisation, no augmentation; all frames kept.
- **[Open]** The exact random-walk distribution (speed range, stop probability, segment length) and the method, encoder, latent, recipe, gate and platform questions are not yet agreed.
