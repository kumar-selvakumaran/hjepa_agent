# World exploration: data and variance

Explored 2026-11-15 by generating data from `agent_world` (commit f09f406, cloned into `scratch/agent_world`) and measuring what a learner would see. Labels: see [README.md](README.md). All numbers were **[Verified]** by probes (`scratch/gen_data.py`, `scratch/analyze_variance.py`, outputs `scratch/*.txt`) unless marked.

## 1. What was generated

| Dataset | Policy | Chasers | Size | Note |
| --- | --- | --- | --- | --- |
| `smooth` | forward 0.2 to 1.0, turn resampled with prob 0.08 per step | 2 per room | 64 episodes x 200 steps = 12,800 steps | Persistent random walk |
| `uniform` | uniform random in [-1, 1]^2 each step | 2 per room | same | Jittery, stays near start |
| `smooth_nochasers` | as `smooth` | off | same | Isolates the chasers' share |

- 1 agent, default `WorldConfig`, 64x48 RGB view, `dt` = 0.1 s, so an episode is 20 s. Saved with actions, agent pose, chaser positions and bump flags (about 5 MB each, in gitignored `scratch/data/`).
- **Throughput on the local 4 GB GPU, 64 worlds:** 143 world-steps/s with chasers, 319 without; peak GPU memory 0.28 GB. At that rate 1M transitions take about 1.9 h with chasers and 52 min without. Larger batches were not tried.

## 2. Figures

- [frame_pairs.png](../media/frame_pairs.png): frame t, t+1, their difference (x8), frame t+5 and its difference, for four cases: a random step, a chaser in view, just after a bump, the largest one-step change.
- [variance_maps.png](../media/variance_maps.png): per-pixel std within an episode, std of the one-step change, and std between episodes.
- [exploration_smooth.gif](../media/exploration_smooth.gif): 120 steps of one smooth-policy episode.

## 3. Variance (smooth policy unless stated)

| Measure | Value | Reading |
| --- | --- | --- |
| Pixel variance, overall | 0.031 (std 0.18) | Plenty of signal; not a constant image |
| Share within an episode / between episodes | 87% / 14% (uniform: 68% / 32%) | Motion dominates; layout differences between resets are a minority |
| Pixels with std < 0.01 | 3.4% | These are the agent's own body, fixed in view |
| Lag 1: pixels changing by > 0.02 | 19% | A third of the view changes every step |
| Lag 1: copy-last-frame MSE / variance | 0.39 | Predicting "same as now" is not trivially good |
| Lag 5 (LeWM's frameskip): same ratio | 0.98 | At five steps the view is nearly decorrelated pixel-wise |
| Frames: participation ratio of PCA spectrum | 10 (of 9,216) | Variance sits in about ten directions |
| Frames: PCs for 90% / 99% of variance | 222 / 1,460 | Low-dimensional core, long tail |
| Lag-1 differences: participation ratio | 259 | Changes are much higher-dimensional than frames |
| corr(change size, abs forward) / corr(change size, abs turn) | 0.04 / 0.48 | **Turning moves the picture; forward motion barely changes it** |

Without chasers the figures are nearly the same (variance 0.031, copy ratio 0.42), so chasers add almost nothing to pixel variance.

## 4. What the data covers

| Item | Smooth | Uniform | Reading |
| --- | --- | --- | --- |
| Episodes that reach room 2 in 20 s | 2% | 0% | The doorway and the patch are almost never visited |
| Mean distance moved per episode | 10.9 units | 7.0 | Out of about 20 possible |
| Frames with a chaser in view (3+ red pixels) | 18% | 15% | Visible but small |
| Frames with the pink patch in view | 33% | 26% | Seen from afar |
| Frames with food in view | 73% | 75% | Everywhere |
| Frames with a blue block in view (own body masked) | 30% | 37% | |
| Nearest chaser within 1.0 unit | 5% of steps | 5% | |
| **Bump events** | **9 in 12,800 steps (0.07%)** | 12 (0.09%) | **Rare** |

- The first blue-block detector matched the agent's own blue body and read 100%; it was corrected by masking the 104 fixed own-body pixels.

## 5. Is the variance enough to avoid collapse?

- **[Verified]** For trivial collapse (constant output, or "next latent = current latent"): **yes, apparently**. Variance is high and consecutive frames differ.
- **[Unverified]** Collapse is a property of training, not data; nothing was trained, so this only rules out the easy failure.
- **[Proposed]** The real risk is dropping what matters, not collapsing outright:
  - Pixel variance is dominated by egocentric motion (turning, horizon, floor shading).
  - The things a survival agent must keep, a chaser, a bump, the patch, are small in pixels, and bumps are 0.07% of steps.
  - The effective dimension is low (about 10 directions carry most variance), which is the regime where LeWM reported SIGReg being weaker. Treat as a risk to monitor.
  - A forward command barely changes the picture, so a predictor may learn to ignore it. Track the predictor's action sensitivity (H-JEPA does).

## 6. Consequences for the data strategy

- **[Proposed]** Random walks are not enough. Add: a doorway-seeking policy (to reach room 2 and cross the patch), chaser-approach episodes (to get bumps and pain), and longer episodes.
- **[Proposed]** Rare events need oversampling or reweighting at training time, and a probe that checks they are recoverable.
- **[Proposed]** Generation time can exceed training time: 1M transitions is about 2 h locally with chasers. Generate on the rented GPU, or test larger batches first.
- **[Open]** Whether the dataset should vary the world (rotate wall, obstacle layouts, patch rate) per episode, as the papers' environments vary scene factors.
- **[Open]** The probe used one agent and the default configuration only.
