# Stage 1 specification (draft for consensus)

Drafted 2026-11-15. Labels: see [README.md](README.md). Intention numbers (INT-xx) refer to [intentions.md](intentions.md). Nothing here is run until the **[Open]** items are settled. Older plans ([stage1-plan.md](stage1-plan.md), [training-strategies.md](training-strategies.md)) are history; where they differ, this document governs.

## 1. Scope

- **[Kumar]** Stage 1 produces a frozen perception module (an encoder, plus a projector if adopted) that stage 2 uses to train a LeWM-style action-conditioned world model (INT-10).
- **[Kumar]** No notion of goals, costs or planning exists in this module or in the world model; planning arrives with the actor (INT-11).
- **[Kumar]** Chaser bumps and the turning patch are absent from pretraining and appear later at runtime (INT-12, INT-21).

## 2. Data

| Item | Decision | Status |
| --- | --- | --- |
| World | `agent_world` at its pinned revision; one agent; two rooms with an open doorway; defaults for obstacles (12), blue blocks (3 per room), food (24 per room, respawn 5 s); **no chasers, no rotating patch** | **[Agreed]** (INT-21, INT-24) |
| Source of experience | Random walk only; no goals, no pose-based steering | **[Agreed]** (INT-11, INT-22) |
| Speed | Per episode, a speed scale uniform in [0.3, 1.0] | **[Agreed]** (INT-62) |
| Actions | Per segment: forward uniform in [-0.5, 1.0] x speed scale; with probability 0.15 a pause (forward 0); turn uniform in [-1, 1] | **[Agreed]** (INT-62) |
| Hold length | Per episode, how many steps a command is held before a new random one is drawn, from 1 step up to about 25 steps (2.5 s). Exact distribution (proposed: log-uniform over the resample probability in [0.04, 1.0]) | **[Agreed]** in kind; exact law **[Proposed]** |
| Start poses | Uniform over collision-free space in both rooms, random heading; implemented by placing the agent after each reset | **[Agreed]** (INT-23); not yet implemented |
| Size and layout | 2,048 episodes x 500 steps (50 s at 0.1 s per step) = 1,024,000 transitions, generated as 4 shards of 512 worlds with different seeds | **[Agreed]** (INT-24) |
| Held out | 10% of episodes, by episode (so layouts are unseen) | **[Agreed]** |
| Curation | Keep all frames; no filtering | **[Agreed]** (INT-24) |
| Uniformity | After generation, measure coverage from ground truth (position grid, action marginals, situation classes: open floor, near wall, near obstacle or block, in the doorway, near food, in a corner) and give the dataloader importance weights to approach uniform | **[Agreed]** (INT-43a, INT-56); the weighting scheme is **[Proposed]** |
| Storage | `frames.npy` uint8 [episodes, steps+1, 48, 64, 3] (9.4 GB total) and `truth.npz` (poses, actions) per shard; ground truth is for evaluation and weighting only, never an input to training | **[Agreed]** (INT-42a) |
| Transfer sets (never used for pretraining) | One with the patch on and chasers off (already generated, `transfer_patch`); one with chasers on and the patch off (to generate); used to test whether the frozen module represents what it has never seen | **[Proposed]** |

## 3. Preprocessing

- **[Agreed]** Divide by 255, then subtract the per-channel mean and divide by the per-channel standard deviation measured on the training split (first shard measurement: mean 0.576, 0.538, 0.572; std 0.266, 0.217, 0.230); the statistics are stored with the checkpoint (INT-25).
- **[Agreed]** No augmentation (INT-25): flips would reverse the spin direction later, crops break the fixed camera geometry, and colour jitter has no counterpart in a single deterministic renderer.

## 4. Model

| Part | Decision | Status |
| --- | --- | --- |
| Encoder | timm ViT-tiny: 12 layers, width 192, 3 heads, 5.5M parameters, input 48x64, patch 6x8 so an 8x8 grid of 64 tokens | **[Agreed]** (INT-50) |
| Projector | A small MLP with BatchNorm after the encoder, applied to every token; SIGReg and the prediction act on its output; it belongs to the frozen perception module. Reason: the ViT's final LayerNorm fixes each token's mean and norm, which SIGReg's Gaussian target cannot satisfy (LeWM reports this) | **[Open]** (on hold, INT-65) |
| Objective | Predict the latents of masked regions from the visible ones, with SIGReg preventing collapse (no EMA teacher; the targets are a stop-gradient copy of the same network) | **[Open]**: chosen in principle (INT-54), under review because of INT-65 |
| Predictor (training only) | A narrow transformer, width 192, depth 4, about 1.8M parameters, discarded after stage 1 | **[Proposed]** |
| Mask | I-JEPA-style: 4 target blocks covering 15 to 20% of the grid each (aspect 0.75 to 1.5), context covering 85 to 100% of the rest | **[Open]**: depends on the predictability study |
| Frame latent for stage 2 | The mean of the patch tokens (one 192-dimensional vector per frame), as LeWM does (INT-51); attentive pooling is a potential configurator role (INT-52, see [component-roles.md](component-roles.md)) | **[Agreed]** |

## 5. Training

| Item | Decision | Status |
| --- | --- | --- |
| Pilot | Three runs of 5 epochs with SIGReg weights 0.03, 0.09 and 0.3, then choose | **[Agreed]** (INT-59) |
| Main run | 40 epochs over 1.02M frames, batch 512, peak learning rate 5e-4, 3 warmup epochs then cosine to zero, AdamW with betas (0.9, 0.95), weight decay 0.05, bf16, gradient clip 1.0, seed 42, SIGReg with 1,024 projections | **[Agreed]** in kind (INT-59); numbers **[Proposed]** |
| Monitors | Every few hundred steps on a held-out batch: effective rank (RankMe), per-dimension spread, the SIGReg value; online linear probes for monitoring only | **[Proposed]** |
| Early exit | If the rank falls below a threshold or the spread reaches zero on several consecutive checks, stop and keep the last good checkpoint (INT-63) | **[Agreed]**; thresholds **[Proposed]** |
| Tooling | `stable-pretraining` (Lightning, `Manager`, probe and rank callbacks) with our own dataset adapter and the few lines for the method's loss | **[Agreed]** (INT-30) |

## 6. Evaluation and gate

- **Probe, defined:** a linear regression (ridge) trained on frozen features to predict a known ground-truth quantity. Features are the per-frame vector. Training frames come from training episodes; scoring uses held-out episodes. It never feeds back into the encoder. If a linear map can read a quantity, the representation holds it explicitly.
- **Targets:** agent x and y, heading (cosine and sine), and whether the agent is in room 2.
- **Metrics:** R squared for continuous targets (1 is perfect, 0 is no better than predicting the mean) and area under the ROC curve for yes-or-no targets (0.5 is chance).
- **Baselines:** the same probe on raw pixels (PCA to 256 dimensions) and on a randomly initialised encoder.
- **Gate (proposed thresholds):** R squared at least 0.9 for position and heading, area under the curve at least 0.95 for room, both baselines beaten, rank not collapsed.
- **Transfer (reported, not gated):** probes for chaser visibility, distance and bearing, and patch visibility, on the transfer sets.
- **If the gate fails:** report and offer options; no automatic second run (INT-63). **[Agreed]**

## 7. Compute

- **[Agreed]** Platform: Lambda (INT-64), a single A100 or H100 virtual machine over SSH, targeting 30 to 60 minutes for the main run, with the later path to 1-Click Clusters.
- **[Agreed]** Costly or critical commands follow the `hitl-dev` skill: a step file that you run, with outputs serialised for me (INT-33).
- **[Proposed]** Data on the machine: regenerate there with the same seeds (about 10 to 15 minutes) or upload the 9.4 GB; to be decided.
- **[Unverified]** Time and cost per run: about $1.5 to $4 per hour of use by provider prices; the 30 to 60 minute estimate comes from arithmetic, not a measurement.

## 8. Open items

1. Whether masked latent prediction suits our images at all (INT-65); a predictability study is running.
2. The projector, which depends on item 1.
3. The exact hold-length law, and the probe-gate and early-exit thresholds.
4. How data reaches the machine.
5. Stage 2 and stage 3 specifications, deferred until stage 1's gate result.

## 9. Pivot, 2026-11-16 (appended; earlier text kept)

**[Kumar]** The encoder and the action-conditioned predictor are trained jointly from scratch, LeWM style (INT-70). Sections 4 and 5 above (masked objective, projector, pilot, frozen encoder) are **[Superseded]**.

### What stands, what is superseded

| Item | Status |
| --- | --- |
| Data: random walks, no chasers, no patch, dispersed starts, 2,048 x 500, 10% held out, weighting for uniformity, keep all frames (section 2) | **[Agreed]**, stands |
| Preprocessing: per-channel mean and standard deviation, no augmentation (section 3) | **[Agreed]**, stands |
| No notion of goals or planning in the model (INT-11); chasers and patch as runtime novelties (INT-12) | stands |
| Staged training with a frozen encoder and later joint fine-tuning (INT-10) | **[Superseded]** |
| Masked latent prediction, SIGReg-masked, I-JEPA pilots, 6x8 patches for I-JEPA (INT-54, INT-59, part of INT-26) | **[Superseded]** |
| Frame latent: the mean of patch tokens (INT-51) | **[Open]**: LeWM itself uses the CLS token plus a projector |
| Exit rules: early stop on collapse, probe gate, stop-and-ask on failure (INT-63) | stands; the probes read the jointly trained latent |
| Platform: Lambda (INT-64), `hitl-dev` for costly steps | stands |

### Joint training as LeWM defines it

- Encoder: ViT-tiny (12 layers, width 192, 3 heads, 5.4M parameters) on each frame; the CLS token goes through an MLP with BatchNorm to give a 192-dimensional frame latent.
- Action encoder: a small embedder over the actions of one transition (frameskip x 2 numbers).
- Predictor: a causal transformer (6 layers, 16 heads, width 192, 10.8M parameters) with action conditioning through adaptive LayerNorm; it sees 3 past latents and actions and predicts the next latent.
- Loss: mean squared error between predicted and actual next latents, plus 0.09 x SIGReg on the latents (1,024 projections); no stop-gradient, no EMA.
- Training windows: 4 frames spaced by the frameskip, with the actions between them.
- LeWM's goal-matching functions (`criterion`, `get_cost`) are not used (INT-11).

### Measured locally (RTX 3050, 4 GB, bf16, random data; scratch probe `lewm_smoke.py`)

| Patch | Tokens | Batch | ms per step | Samples per second | Peak GPU memory |
| --- | --- | --- | --- | --- | --- |
| 8 | 48 | 32 / 64 / 128 | 201 / 229 / 346 | 159 / 280 / 370 | 0.98 / 1.62 / 2.94 GB |
| 4 | 192 | 32 / 64 / 128 | 369 / 4,122 / 22,372 | 87 / 16 / 6 | 2.71 / 5.06 / 9.83 GB |

- 10 epochs over about 1M windows at patch 8 and batch 128 would take about 7.5 hours locally (arithmetic from the measurement); a rented A100 or H100 should be far faster (**[Unverified]**, not measured).

### Amendment, 2026-11-16 (appended)

- **[Kumar]** Frameskip is fixed at 2 (INT-76), replacing the earlier choice of 1; patch 8 by 8 (INT-71) and the CLS token with the projector (INT-72) stand. A training window is 4 frames spaced 2 steps apart, spanning 7 steps, with a four-number action per transition (the first three of the four loaded entries are used).
- **[Open]** The SIGReg weight and training length (see [sigreg-weight.md](sigreg-weight.md)); dispersed start poses are not yet implemented; the LeWM training script and the Lambda step file are not yet written.

## 10. Consensus snapshot for the first run, 2026-11-16 (appended)

Everything below is what we would run. **[Agreed]** means stated by Kumar; **[Proposed]** means a default of mine that Kumar has not yet explicitly confirmed.

| Area | Decision | Status |
| --- | --- | --- |
| Method | Encoder and action-conditioned predictor trained jointly from scratch, LeWM style; no goals, costs or planning (INT-70, INT-11) | **[Agreed]** |
| Encoder | timm-style ViT-tiny (12 layers, width 192, 3 heads, 5.4M parameters), patches of 8 by 8 on 48 by 64 frames (6 by 8 grid, 48 tokens), built at a square 64 by 64 native size with interpolated position embeddings | **[Agreed]** (INT-50, INT-71); the 64 by 64 construction is forced by the library |
| Frame latent | CLS token through an MLP with BatchNorm, 192 numbers (INT-72) | **[Agreed]** |
| Predictor | LeWM's: causal transformer, 6 layers, 16 heads, width 192, dropout 0.1, action conditioning by adaptive LayerNorm, 10.8M parameters | **[Proposed]**, covered by "use what LeWM uses" (INT-77) |
| Windows | 4 frames spaced 2 steps apart (7 steps); history 3, predict 1; one four-number action per transition (INT-76) | **[Agreed]** |
| Loss | Mean squared error between predicted and actual next latents plus 0.09 x SIGReg (1,024 projections, 17 knots); no EMA, no stop-gradient (INT-77) | **[Agreed]** |
| Optimiser and length | AdamW, learning rate 5e-5, weight decay 1e-3, batch 128, bf16, gradient clip 1.0, warmup then cosine, 10 epochs over every valid window (about 1.01 million: 2,048 episodes x 494 starts) (INT-77) | **[Agreed]** in kind; interpretation of "LeWM's values" to confirm |
| Data | Random walks per INT-62 (speed scale, hold length, 15% pauses, reversing), no chasers, no patch, defaults otherwise, starts uniform over free space in both rooms with random heading (not yet implemented), 2,048 x 500, 10% of episodes held out, all frames kept (INT-21 to INT-25) | **[Agreed]** |
| Preprocessing | Divide by 255, per-channel mean and standard deviation from the training split, no augmentation (INT-25) | **[Agreed]** |
| Uniformity | Measure coverage from ground truth (position grid, action marginals, situation classes) and apply importance weights in the dataloader (INT-56); the exact weighting scheme | **[Agreed]** in kind; scheme **[Proposed]** |
| Early exit | Stop and keep the last good checkpoint when collapse is detected; thresholds proposed: SIGReg statistic far above its healthy floor (about 1.0 to 1.4) or effective rank under 25% of the width on several consecutive checks (INT-63) | **[Agreed]** in kind; thresholds **[Proposed]** |
| Exit gate | Probes on held-out episodes (position and heading R squared at least 0.9, room area under the curve at least 0.95, food visibility added), both baselines beaten, one-step and two-step prediction error below the copy-last-latent baseline, action sensitivity checked; if it fails, stop and ask (INT-63) | **[Agreed]** in kind; thresholds **[Proposed]** |
| Transfer sets | Never used in training: patch on and chasers off (generated earlier, with the old policy), chasers on and patch off (to generate) | **[Proposed]** |
| Platform | Lambda, a single GPU over SSH, costly steps through `hitl-dev` step files (INT-64, INT-33) | **[Agreed]** |
| Data on the machine | Regenerate there with the same seeds, or upload about 9.4 GB | **[Open]** |

## 11. Data format contract and confirmations, 2026-11-16 (appended)

- **[Agreed]** All decisions in section 10 marked **[Proposed]** are now confirmed (INT-83 to INT-85, INT-77), except that the "Data on the machine" item is settled as: generate locally, upload compressed (INT-78). Simulation-only capabilities used are listed in [simulation-privileges.md](simulation-privileges.md).
- **Shard contract** (`data/<name>/`), written by the generator and read by the loader:
  - `frames.npy`: uint8 [E, T+1, 48, 64, 3].
  - `truth.npz`: `agent_pos` [E, T+1, 2]; `agent_rot` [E, T+1] (radians); `actions` [E, T, 2] (forward, turn; action t takes step t to step t+1); `bump` [E, T] bool; `chaser_pos` [E, T+1, C, 2] (C is 0 when chasers are off); `obstacle_pos` [E, K, 2] and `obstacle_height` [E, K]; `block_pos` [E, B, 2]; `food_pos` [E, T+1, F, 2] and `food_present` [E, T+1, F] bool; `walk_scale` [E]; `walk_p` [E].
  - `config.json`: the world configuration, policy parameters, seed and the git revisions of this repository and of `agent_world`.

## 12. Training pipeline built, 2026-11-16 (appended)

**[Built]** and tested locally (19 unit tests; a 60-step smoke run; an injected failure and a resume). Nothing has been run on real data or in the cloud.

| Piece | What it is | Reuse |
| --- | --- | --- |
| `scripts/generate_data.py`, `world.py` | Random-walk generator with dispersed starts and the recording of section 11 | VMAS calls found by a forked agent: `set_pos`, `set_rot`, then `get_from_scenario` to re-render |
| `data.py`, `weights.py` | Window loader (frameskip 2, split by episode, per-channel normalisation), privileged labels for probes, situation classes, importance weights capped at 10, coverage report, weighted sampler | custom (domain-specific) |
| `recipe.py`, `model.py` | LeWM's recipe as a frozen dataclass; LeWM built from `stable-worldmodel` with the ViT from `stable-pretraining`, wrapped as a Lightning module; only encode and predict are used | `stable-worldmodel`, `stable-pretraining`, Lightning |
| `callbacks.py` | `CollapseGuard`: stop on collapse and keep `last_good.ckpt` | custom, 30 lines |
| `scripts/train.py` | Preflight, training, checkpointing, summary; `--resume`, `--preflight-only`, `--fail-at-step` | Lightning `Trainer`, `ModelCheckpoint`, `OnExceptionCheckpoint`, `Timer`, `LearningRateMonitor`, loggers; `stable-pretraining` `HardwareMonitor`, scheduler |
| `scripts/eval_probes.py` | Linear probes on pixels, random-init latents, or a checkpoint's latents | scikit-learn |

**Monitoring** (CSV and TensorBoard, INT-86, INT-87): per step, total, prediction and SIGReg loss, learning rate, gradient norm; per validation, prediction error, SIGReg statistic, copy-the-last-latent ratio, action sensitivity (error with shuffled actions over error with true actions), effective rank, per-dimension spread; hardware (GPU utilisation, memory, power, temperature; CPU, RAM, disk, network); an environment dump and a frozen requirements list.

**Checkpointing** (INT-86, INT-88): the last 3 step checkpoints and `last.ckpt` every 1,000 steps; `last_good.ckpt` on every healthy validation; `on_exception.ckpt` on any exception; `final.ckpt` always, in a `finally`; `summary.json` with the stop reason; a 90-minute `Timer`. `--resume` continues from the newest checkpoint of any kind, `--resume last_good` from the last healthy one.

**Preflight** (INT-81): GPU present and large enough, disk space, shard shapes, six real forward and backward steps with a finiteness check, a checkpoint save and load roundtrip, measured samples per second and an estimated total time that must fit within 1.2 x the time cap.

**Findings while building:**
- `HardwareMonitor` does nothing without `psutil` and `pynvml`; both are now dependencies.
- `stable-worldmodel` needs `imageio`, `tensorboard` was added for the dashboard.
- On Windows set `PYTHONUTF8=1`; otherwise logging errors appear but training is unaffected.
- A first failed preflight caught a wrong attribute name: the preflight earns its place.
- Local measured speed in the smoke run: 83 to 125 samples per second at batch 32 (a few-step estimate, not the batch-128 figure of 370).
