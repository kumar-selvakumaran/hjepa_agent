# Evaluation methods and hypothesis validation

How the world model is evaluated, and how each hypothesis in the cloud iterations was tested. Written 2026-11-17 during the first auto-research campaign (`docs/journal/cloud-iterations.md`; artefacts in `scratch/auto-research/2026-11-17-lewm-stage1/`).
Evidence marks: **[measured]**, **[verified]** (checked directly in code or data), **[documented]** (a source says so), **[assumed]**.

---

## 1. What is evaluated, and why

| Question | Evaluated by | Section |
| --- | --- | --- |
| Is training healthy right now? | Validation monitors every 1,000 steps; the collapse guard | 2 |
| Does the latent carry information about the world? | Linear and MLP probes against privileged ground truth, with baselines | 3 |
| Is the information identifiable at all from one frame? | Identifiability checks (symmetry harmonics, sign-free targets) | 4 |
| Where in the network is information kept or lost? | Layer probes (patch tokens, CLS, projected latent) | 5 |
| Has training converged? | Probe trend over checkpoints, seed agreement, plateau rule | 6 |
| Is a suspected failure real? | Hypothesis tests designed so each candidate predicts a different outcome | 7 |
| Is the run efficient? | Capacity measurements and profiling | 8 |

Privileged labels (true position, heading, room, visibility flags) are used **only** for evaluation, never as model input (simulation privileges, `docs/journal/simulation-privileges.md`).

## 2. Training-time monitors

### 2.1 Validation set
- **Episodes:** 10 percent of episodes per shard are held out (`episode_split`), so validation windows come from episodes never trained on **[verified]**.
- **Windows:** a fixed random subset of 6,400 validation windows (50 batches of 128), drawn once with the recipe seed (commit `19e315e`) **[verified]**.
- **Why fixed and random:** fixed so every check sees the same windows (comparable over time); random so each batch mixes about 47 episodes (measured on `pretrain_100`). Before this fix each batch held 128 consecutive windows of one episode (iteration 1).

### 2.2 Monitors (logged as `val/*` every 1,000 steps)

| Monitor | Definition | Healthy | Reads |
| --- | --- | --- | --- |
| `val/pred_loss` | Mean squared error between the predicted and the actual next latent | Falling | Prediction quality |
| `val/sigreg` | SIGReg statistic of the batch's latents (Epps-Pulley on 1,024 random projections, 17 knots) | Near its floor (about 1.0 to 1.5); at most 5 for the guard | Distance from an isotropic Gaussian; high means collapse or a non-representative batch |
| `val/eff_rank` | exp of the entropy of the normalised singular values of the centred latents [N, D] | At least 25 percent of 192 (48) for the guard | How many directions the latents use |
| `val/dim_std` | Mean per-dimension standard deviation of the latents | Near 1 | Scale; near 0 means collapse |
| `val/copy_ratio` | Prediction error divided by the error of copying the last latent | Below 1 | Whether the predictor beats the trivial "nothing changes" guess |
| `val/action_sensitivity` | Error with actions shuffled across the batch, divided by error with true actions | Above 1 | Whether the predictor uses the action |

Training also logs `train/loss`, `train/pred_loss`, `train/sigreg` and `train/grad_norm` on the steps the logger writes (every 10 steps; commit `1809669`), plus hardware monitors (GPU use, memory, power, CPU, RAM).

### 2.3 Collapse guard (INT-83)
- After a 2,000-step grace period, a validation check is unhealthy if `val/sigreg` > 5 or `val/eff_rank` < 48.
- Three unhealthy checks in a row stop the run; each healthy check saves `last_good.ckpt`.
- **Lesson (iteration 1):** the guard is only as good as its measurement. Batch statistics must come from batches that represent the distribution.

## 3. Probes

### 3.1 Protocol (`scripts/eval_probes.py`, `tools/probe_trend.py`)

| Item | Setting |
| --- | --- |
| Frames | 20,000 training frames and 5,000 validation frames, sampled at random from the train and held-out episodes of the four pretraining shards (seeds 0 and 1) |
| Normalisation | Per-channel mean and standard deviation of the training shards |
| Features | One frame at a time; the frame latent, or the layer under study (section 5) |
| Linear probe | `StandardScaler` then ridge regression (alpha 1.0) |
| Non-linear probe | `StandardScaler` then an MLP (two hidden layers of 256, early stopping, seed 0) |
| Regression score | R2 on the validation frames (1 is perfect, 0 is no better than the mean, negative is worse) |
| Classification score | ROC AUC of the ridge output (room, food visible) |

### 3.2 Baselines (every probe score is read against both)

| Baseline | Features | Purpose |
| --- | --- | --- |
| Pixels | The frame flattened, reduced to 256 dimensions with PCA | What a probe can read with no learning at all |
| Random init | The same network with random weights | What the architecture gives without training |

A trained latent should beat both. Random-init ridge scores can be strongly negative (for example -10): the probe overfits a poorly conditioned random feature space.

### 3.3 Targets

| Target | Identifiable from one frame? | Status |
| --- | --- | --- |
| `pos_x`, `pos_y` (absolute position) | Partly: mirror symmetry hides the sign of y; x mainly through which room | Original gate target (INT-83) |
| `heading_cos`, `heading_sin` (absolute heading) | No: the world looks alike every 90 degrees | Original gate target; ill-posed (iteration 2) |
| `cos(4 x heading)`, `sin(4 x heading)` | Yes | Added in iteration 2 |
| `abs(pos_y)` | Yes (distance from the centre line) | Added in iteration 2 |
| `room2` (which room) | Mostly | Original gate target (AUC) |
| `food_visible` | Yes | Original target (beats baselines only) |

## 4. Identifiability checks

A probe score of 0 can mean the model failed or the target is not visible in the input. These checks separate the two.

| Check | How | Result |
| --- | --- | --- |
| Pixel ceiling | Probe the same target from pixels with an MLP | If pixels+MLP also score 0, the target is not in single frames |
| Symmetry harmonics | Probe cos and sin of k x heading for k = 1, 2, 4 | k = 4 decodable, k = 1 not: 90-degree aliasing (iteration 2) |
| Sign-free targets | Probe abs(pos_y) next to pos_y | abs readable (0.54), signed not (-0.01): mirror symmetry (iteration 2) |
| Label timing | Correlate the recorded turn action with the next heading change, at lags -1, 0, +1 | 0.95 at lag 0: labels aligned with frames (iteration 2) |

## 5. Layer probes (where information is kept or lost)

The encoder path is: frame, ViT patch tokens (48), CLS token (192), projector (MLP with batch norm, to 192), frame latent. The predictor and SIGReg act on the projected latent.

| Layer probed | Feature |
| --- | --- |
| All patch tokens | 48 x 192 values, reduced with PCA to 256 |
| Mean patch token | Average over the 48 patch tokens |
| CLS token | Before the projector |
| Projected latent | After the projector (what the predictor sees) |
| Pixels | Reference ceiling |

Result on run1 final (MLP probe, R2, cos and sin of 4 x heading) **[measured]**: all patch tokens 0.88 / 0.87; mean patch 0.66 / 0.64; CLS 0.70 / 0.68; projected 0.50 / 0.49; pixels about 0.90.
Reading: the encoder keeps the heading-mod-90 signal; the CLS summary keeps most of it; the projector discards about a third. This matches the documented effect that probing a projector's input beats its output, by up to about 30 percent in self-supervised methods (Guillotine Regularization: Bordes and Balestriero, arXiv 2206.13378; Chen et al. 2020; Xue et al., ICLR 2024) **[documented]**.
Consequence for evaluation: downstream readout should be judged on the backbone (CLS or patch tokens); the projected latent is the space the predictor and SIGReg shape.

## 6. Convergence (INT-96)

A run counts as converged when all of the following hold.

| Criterion | How it is measured |
| --- | --- |
| Finished without collapse | `summary.json` status `complete`; guard never stopped the run |
| Validation prediction loss plateaus | `val/pred_loss` over the last checkpoints (read from the run's CSV) |
| Probe scores plateau | `tools/probe_trend.py` on checkpoints every 10,000 steps plus `final` |
| Two seeds agree within their spread | The same probes and losses on run1 (seed 42) and run2 (seed 43) at the same steps |
| Probes beat pixel and random baselines | Section 3.2 |

The seed spread measured so far (about 0.03 R2 on probes, 0.4 percent on final prediction loss) is the yardstick for "no further improvement": a later change must beat it to count.

## 7. How hypotheses were validated

### 7.1 Method
1. **List competing hypotheses**, each with what it predicts (2 to 4).
2. **Ground** each in run evidence (logs, metrics, code) and ranked outside sources (tier 1: official docs or source; 2: papers; 3: maintainers; 4: community; 5: blogs, never alone).
3. **Design a discriminating test**: the cheapest measurement whose outcome differs between the hypotheses. Offline tests on saved checkpoints come before new training jobs.
4. **Run, read, decide**: keep the hypotheses whose predictions held; record the numbers and the decision in `reasoning.md` and the iteration journal.

### 7.2 Record of hypotheses

| Iteration | Id | Hypothesis | Discriminating test | Outcome |
| --- | --- | --- | --- | --- |
| 1 | H1 | Batch norm in evaluation mode distorts validation | D01: SIGReg and rank on mixed batches with the network in eval mode, batch-norm-only batch mode, and full train mode | Small real effect: eval mode adds about 1.1 to SIGReg, still under 5 |
| 1 | H2 | Each validation batch holds one episode | Local count of episodes per batch; D01 on mixed batches; live run after the fix | **Confirmed**: 1 episode per batch; mixed batches healthy in all modes; run1's next check healthy (SIGReg 2.5, rank 61) |
| 1 | H3 | Real collapse | D01 rank on mixed batches; probes against baselines | Rejected: rank about 56; probes above baselines |
| 2 | H4 | Information present but not linear | MLP vs ridge probes | Partly: MLP helps for heading mod 90, not for absolute heading |
| 2 | H5 | World symmetry aliases absolute pose | Harmonics (k = 1, 2, 4) and sign-free targets, on latents and pixels | **Confirmed** for heading (90 degrees) and position y (mirror) |
| 2 | H6 | Labels misaligned with frames | Turn action vs heading change correlation by lag | Rejected: 0.95 at lag 0 |
| 3 | (perf) | The data loader limits throughput | Process CPU use per worker; py-spy on the main process | Rejected: workers at 4 percent; the main process was saturated by per-step logging (31 percent in gradient norms, 15 percent in `log_dict`) |
| 4 | H7 | The projector discards heading information | Layer probes (section 5) | **Confirmed**: CLS 0.70, projected 0.50, patch tokens 0.88 |
| 4 | H8 | The encoder never learns it | Layer probes | Rejected: patch tokens match pixels |
| 4 | H9 | Single-frame probes undersell a temporal model | Not yet tested | Open |

### 7.3 Diagnostics used (all serialised under `scratch/auto-research/2026-11-17-lewm-stage1/iterations/`)

| Diagnostic | What it measures | Where |
| --- | --- | --- |
| Validation batch composition | Episodes per batch, old vs new loader | `iter-01/evidence/val_batch_check.*` |
| D01 | SIGReg, rank, prediction loss by network mode and split, then probes | `iter-01/evidence/D01/` |
| Non-linear probe | Ridge vs MLP on latents and pixels | `iter-02/probes/probe_nonlinear.*` |
| Symmetry probe | Heading harmonics; label timing | `iter-02/probes/probe_symmetry.*` |
| Position probe | Signed vs unsigned position | `iter-02/probes/probe_position.*` |
| py-spy | Where the main training process spends time | `iter-03/evidence/D02/` |
| Layer probe | Information by encoder layer | `iter-04/probes/probe_layers.*` |
| Probe trend | Probe scores over checkpoints, both seeds | `runs/<run>/probe_trend.jsonl` |

## 8. Efficiency measurements

| Measurement | Method | Result |
| --- | --- | --- |
| Throughput | Steps per second from the log; 300-step test (C09) | 10.4 alone; 7.3 each with two runs; 8.3 and 8.0 after the logging fix |
| Capacity per job | GPU memory, GPU busy, CPU per process | 3.4 to 3.9 GB; about 50 percent busy alone; one core for the main process |
| Concurrent jobs | floor(capacity / use) over each resource, smallest binds | GPU compute binds: 2 jobs |
| Profile | py-spy, 30 s, 50 Hz, on the running main process | Section 7.2, iteration 3 |

## 9. Limits

- Probes are single-frame; the world model sees three frames. Temporal information (H9) is not probed yet.
- The probe set is 25,000 frames from the pretraining distribution; transfer sets (chasers, patch) are evaluated separately with the training normalisation.
- MLP probe scores depend on the MLP size and early stopping; differences under about 0.03 are within the measured seed spread.
- The original gate (INT-83) asks for absolute heading R2 of at least 0.9, which section 4 shows is not identifiable from one frame in this world. It is reported, not changed; a revision needs Kumar's decision.

## 10. Added 2026-11-18: frame of reference
- Labels for heading and position are in agent_world's fixed world frame: origin at the centre between the rooms, +x from room 1 into room 2, heading counter-clockwise from +x, unwrapped (it accumulates turns). Checked on `pretrain_100`: forward motion follows (cos, sin) of the heading with a median error of 0.018 rad.
- A probe learns the frame only from its labels; the encoder never sees it. A world-frame target is readable only to the extent the image determines it.
- In this world the floor's axis-aligned checkerboard makes heading modulo 90 degrees readable from any frame, a simulation shortcut (journey section 31, gap G10). Future perception targets: egocentric quantities (distance to the wall ahead, bearing and distance to visible objects, free space ahead) and pose relative to the episode start (INT-100).
