# Codebase study: H-JEPA, LeWM, PLDM, HWM

Studied 2026-11-15. Clones are in `scratch/` (gitignored); papers are in `scratch/papers/*.pdf` with text extracts.
Labels: see [README.md](README.md). Nothing was installed or run; every claim is **[Verified]** by reading unless marked.

## 1. The four, at a glance

| | PLDM | LeWM | HWM | H-JEPA |
| --- | --- | --- | --- | --- |
| Paper | Sobal et al., NeurIPS 2025 (arXiv 2502.14819) | Maes et al. 2026 (2603.19312) | Zhang et al., CoRL 2026 (2604.03208) | Zhang, Terver et al. 2026 (2610.06805) |
| Repo | `vladisai/PLDM` | `lucas-maes/le-wm` | `kevinghst/HWM_PLDM` | `kevinghst/H-JEPA` |
| Role | Flat JEPA world model + planner | Simplest flat JEPA | Hierarchy over a **shared** latent space | Hierarchy with a **learned latent per level** |
| Anti-collapse | VICReg-style var + cov, plus time-smoothness and IDM (5 loss terms) | SIGReg only (2 loss terms) | Inherits the base model's | SIGReg per level, plus IDM when scenes vary |
| Encoder | Impala CNN | ViT-tiny, CLS token, then MLP + BatchNorm | Base model's | LeWM's L1; 2-layer MLP above |
| Predictor | GRU or MLP, ensemble of 5 | 6-layer causal transformer, AdaLN-zero action conditioning | Transformer over waypoints | Causal transformer per level |
| Planner | MPPI | CEM | CEM / MPPI at both levels | Gradient descent (AdamW) at every level |
| Licence | MIT | MIT | **No licence file** | MIT |
| Size | 16.8k lines | **0.8k lines** | 17.4k lines (PLDM fork) | 19.6k lines (includes vendored `stable_worldmodel`) |
| Stack | plain PyTorch | Lightning, Hydra, wandb, stable-pretraining, stable-worldmodel | plain PyTorch | same as LeWM, pinned to Python 3.11 and torch 2.7.1 |

## 2. What each one contributes (declarative)

### PLDM
- Reward-free offline data, goal reaching by latent planning. Beats model-free RL on unseen layouts, short trajectories, random data.
- Reuses the model for a new task by flipping the cost sign: **avoid a chasing agent** (Fig. 5). Directly relevant to our chasers.
- Cost = goal distance + `beta` x ensemble variance (an uncertainty penalty that keeps plans in-distribution). Eq. 5 to 7.
- Loss = `L_sim + a L_var + b L_cov + d L_time-sim + w L_IDM`. Ablation: dropping var or cov collapses it (13% and 29% success on Two-Rooms).
- **Limitations stated:** navigation only, **no partial observability**, about 4x slower at inference than model-free.

### LeWM
- `L = ||z_hat - z||^2 + 0.09 x SIGReg(Z)`. SIGReg projects embeddings on 1024 random directions and runs an Epps-Pulley normality test on each.
- No stop-gradient, no EMA, no pretrained encoder. About 15M parameters; the paper says a few hours on one GPU.
- `jepa.py` (about 140 lines) is the whole model: `encode`, `predict`, `rollout`, `criterion`. `module.py` holds SIGReg and the transformer blocks.
- Planning cost = squared distance from the **last predicted latent to the goal latent**. Horizon 5, action block 5 (Push-T config).
- Probing and violation-of-expectation (teleport raises surprise; colour change barely does). Latent paths straighten without being asked.
- **Limitations stated:** short planning horizon; needs diverse offline data; **SIGReg is weaker in simple, low-dimensional worlds** (Two-Room: LeWM 87% vs PLDM and DINO-WM 97 to 100%).

### HWM
- Two world models in one latent space: low level predicts 1 step; high level predicts between waypoints, conditioned on a **learned macro-action** (encoded from a chunk of primitive actions).
- The first high-level prediction becomes the subgoal for the low-level planner. Closed-loop, both levels replan.
- Fixes two failures of flat planning: non-greedy tasks (move away to get closer) and long horizons. Real-robot Franka pick-and-place: 70% vs 0% flat.
- **Limitations seen:** failures come from coarse subgoals losing detail (the paper's own words). Evaluated goal-image only.
- **Code caveat:** `HWM_PLDM` is a PLDM fork on **Diverse Maze only**, with no licence file. In its config the level-2 model has a latent variable `z_dim: 8` with a KL term and a posterior built from actions. **[Unverified]** how this maps to the paper's "action encoder".

### H-JEPA
- Each level has its own encoder E, action encoder A and predictor F, trained **end to end** with `L = L_pred + lambda SIGReg` per level. Gradients from upper levels reach the lower encoders.
- Level `l` state = encoder over a window of level `l-1` latents, stride `s` (all runs: stride 2, window 1). Level `l` action = pooled `s` lower actions.
- Planning, top-down: the top level minimises distance to the goal in **its** latent space. Its first predicted state is the subgoal for the level below, measured in the upper space. Eq. 4 to 6.
- Result: Visual AntMaze 18% (flat LeWM) to 73% (3 levels, **fewer** planner FLOPs). 4 levels drop to 63%. Mean over 3 seeds.
- Selective abstraction: probes show fast, unpredictable factors (ant leg pose, a distractor) fade at higher levels; position stays. Appears only when the fast/slow **frequency gap is above about 2x**.
- Part of the gain is a **better cost geometry**: the upper-level latent gives smoother distance-to-goal than level 1 (Fig. 7, Table 1), even for flat planning.
- DROID real video needs the IDM term; without it the encoder keeps the scene and drops the robot ("slow-feature collapse", Appendix L).

## 3. How the code works (procedural)

| Step | LeWM | H-JEPA |
| --- | --- | --- |
| Data | HDF5, windows of `history + preds` frames, `frameskip` actions grouped per step | Same, plus per-level `frameskip` and clips that must lie inside one episode |
| Encode | `emb = projector(ViT(pixels).cls)`; `act_emb = MLP(frameskip x action)` | Level 1 as LeWM, optional proprio fusion; higher levels via `encode_hierarchical`, with random temporal crops per level |
| Train step | `pred = predictor(emb[:, :ctx], act_emb[:, :ctx])`; MSE to `emb[:, 1:]`; add `0.09 x SIGReg` | The same per level (`hjepa_forward`), summed over levels |
| Plan | CEM over `num_samples` action sequences; cost = last-step MSE to goal latent; MPC every K steps | `HierarchicalSolver.solve`: loop from top level down; each level runs `GradientSolver` (e.g. 90 AdamW steps x 100 to 400 samples) |
| Subgoal hand-off | none | Upper level's first predicted latent becomes the lower level's goal; on the last step the real goal replaces it |
| Evaluate | success when the env reaches the goal state | Same; plus probes and post-hoc decoders for diagnosis |

Typical settings read from config: 64x64 pixels (ant), patch 4, embed 384, predictor 4 to 5 layers, history 3, batch 128, bf16, **5 epochs** (ant) or 100 (LeWM Push-T), lr 5e-5, AdamW.

## 4. Critical read

### What is solid
- **[Verified]** The LeWM objective is genuinely small and stable on paper; the code matches the paper. A port is feasible in about 300 lines.
- **[Verified]** H-JEPA's ablations separate the temporal and representational benefits, with the cost-only experiment. That is good practice.
- **[Verified]** Compute is reported as planner FLOPs, not just success. We should copy that habit.

### Weak points and open questions
- **Goal images are the only objective.** All four plan to match a goal *observation*. No reward, no cost over bodily state, no value estimate. LeCun's Cost module is absent.
- **No memory.** Predictor context is `history_size = 3` frames. That is a short window, not Short Term Memory, and none of the four tests partial observability (PLDM says so explicitly).
- **Deterministic predictors.** No latent variable for unpredictability. PLDM uses an ensemble only as an uncertainty penalty. A chaser that sometimes sees you and sometimes does not will be averaged.
- **No reactive actor.** Everything is planning at test time (LeCun's Mode 2). There is no amortised policy (Mode 1), so no habit formation, and no Configurator choosing between them.
- **Fixed camera, fixed goal.** Environments use top-down or fixed back cameras, so a goal image is a well-defined target. An egocentric rotating view makes goal images ambiguous.
- **Hierarchy keeps what is predictable, not what matters.** H-JEPA's abstraction criterion is predictability over the level's horizon. A rare, decisive event (a chaser hit) is unpredictable, so a higher level may discard it.
- **Small evaluation budgets.** 50 tasks, 3 seeds, mean +/- SE. Planner settings are picked as "best mean success under 100 TFLOPs". The paper says depth comparisons use a separate task set; **[Unverified]** whether selection and reporting sets are fully disjoint (appendix E not read).
- **Non-monotone depth.** 4 levels are worse than 3 on AntMaze; Push-T collapses at 3 levels (short episodes starve the upper levels of clips, 58% of transitions kept).
- **Small discrepancies, minor:** LeWM paper text says lambda = 0.1; the config uses 0.09. LeWM paper plans with CEM; H-JEPA plans with gradient descent. H-JEPA requires Python 3.11 only.
- **Reproducibility burden.** 3 of 4 repos lean on `stable-pretraining` or `stable-worldmodel`, plus Lightning, Hydra and wandb. Hard to extend; conflicts with our "simplicity first" and "uv only" rules.
- **HWM_PLDM has no licence.** We may read it but should not copy code from it.
EOF
## 5. Correction, 2026-11-15 (appended; earlier text kept)

Section 4 says LeWM's weaker Two-Room result is a limit that "describes" simple worlds such as `agent_world`. That overstates it:

- It is the authors' **untested hypothesis** (low diversity and low intrinsic dimensionality make matching a high-dimensional isotropic Gaussian harder), from one environment.
- On the visually richer 3D OGBench-Cube, DINO-WM beat LeWM (86% vs 74%), so complexity does not predict how LeWM compares.
- `agent_world` has a richer view than Two-Room, and its data diversity is ours to control.
- Treat it as a **risk to monitor** (embedding spread and rank, latent probes against ground truth), not a prediction.
