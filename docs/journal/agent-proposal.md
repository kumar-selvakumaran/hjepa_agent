# Proposal: building the HJepa-Agent

Drafted 2026-11-15 from [codebase-study.md](codebase-study.md) and the diagram in `../diagrams/`. Everything here is **[Proposed]** until Kumar agrees.
Labels: see [README.md](README.md).

## 1. Target

- The diagram's agent: Configurator, Perception, World Model, Short Term Memory, Actor, Cost (intrinsic cost + Critic), acting in `agent_world`.
- The world gives a 64x48 RGB third-person view and a 2D action `(forward speed, turn rate)`. Physics is cheap; rendering dominates (about 350 world-steps/s at 256 worlds).
- The world has no reward, energy or pain yet. Food is eaten with no effect. `chasers.touched` is the hook for pain.

## 2. Module map

| Module (diagram) | Proposal | Source to reuse | Status in the 4 repos |
| --- | --- | --- | --- |
| Perception | ViT-tiny or small CNN on the view, CLS then projection with BatchNorm. A second small MLP encodes the **body channel** (pain, energy, own speed and turn) and fuses with it | LeWM `jepa.py`; H-JEPA proprio fusion encoder | Pixels: **yes**. Body channel: only as "proprio" |
| World Model | Action-conditioned causal transformer predictor (AdaLN-zero), trained with next-latent MSE + SIGReg, end to end | LeWM `module.py`, `jepa.py` | **Yes** (flat); hierarchical in H-JEPA |
| Short Term Memory | Rolling buffer of `(latent, action, intrinsic cost)` per level. Predictor context = this buffer. Start with a window of about 16 steps, not 3 | none (LeWM and H-JEPA use `history_size = 3`) | **No** |
| Cost: intrinsic | Fixed, hand-written function of the **predicted body channel** (pain up, energy down = cost). Immutable, as in LeCun | none | **No** (they use goal distance) |
| Cost: Critic | Small MLP on a latent, predicting discounted future intrinsic cost; trained on real and imagined rollouts | none | **No** |
| Cost: goal term | Optional latent distance to a goal embedding, kept as one switchable term | LeWM `criterion`; H-JEPA Eq. 4 to 6 | **Yes** |
| Actor, mode 2 | Plan in latent space: sample action sequences, roll out, score cost + critic, MPC replan | LeWM CEM; H-JEPA `GradientSolver` | **Yes** (CEM, MPPI, gradient) |
| Actor, mode 1 | Policy network distilled from the planner's first actions (amortised habit) | none | **No** |
| Hierarchy | Stack of JEPA levels, stride 2, each in its own latent; top-down subgoals | H-JEPA `HJEPA`, `HierarchicalSolver` | **Yes** (goal-image only) |
| Uncertainty | Predictor ensemble (cheap), later a latent variable `z` for unpredictable events | PLDM Eq. 6; HWM_PLDM `z_dim` (**[Unverified]**) | Partial |
| Configurator | Static settings object first: cost weights, horizon, which levels, planner budget. Later rule-based (plan harder when uncertain), only then learned | none | **No** |

## 3. Control loop (one step)

1. Perception encodes the current view and body channel to a latent; push into Short Term Memory.
2. Configurator picks cost weights, horizon and levels.
3. Actor plans: roll the World Model forward under candidate actions; score each by intrinsic cost (+ critic at the end, + goal term if set).
4. Execute the first K actions; observe; replan. Later: Mode 1 answers instantly when the Configurator trusts it.
5. Training runs offline on logged data, later also on fresh rollouts.

## 4. Build order

| Stage | What | Done when (a test that fails without it) |
| --- | --- | --- |
| S0 | Data: scripted and random exploration from `agent_world`, saved with ground truth (pose, chaser, food, patch) | Dataset loads; ground truth lines up with frames |
| S1 | Perception + flat World Model, **our own minimal port of LeWM** | Latent probes recover pose and chaser position; no collapse (embedding std, rank) |
| S2 | Short Term Memory: window 3 vs 16, and occlusion probes | A chaser hidden behind a block is predicted better with memory than without |
| S3 | Body channel, intrinsic cost, Critic. Needs world changes (energy, pain) | Planner avoids a chaser it can see, **with a fixed cost** and no goal image |
| S4 | Actor mode 2 end to end (CEM first), then mode 1 distilled | Mode 1 matches planner on routine states at a fraction of the cost |
| S5 | Hierarchy (H-JEPA levels), uncertainty ensemble | Long-horizon task (cross the patch into room 2 and eat) succeeds at lower planner FLOPs than flat |
| S6 | Configurator: static, then rule-based | Gating by uncertainty beats a fixed budget at equal compute |

Why this order: the papers' gains (hierarchy) show up on long-horizon goal navigation, which the world does not yet have. The missing LeCun modules (memory, cost, critic) are what make `agent_world` demands testable.

## 5. Evaluation (reuse the papers' protocols)

- **[Proposed]** Probing against ground truth with variance-normalised MSE, per entity (H-JEPA §3). Fits the "ground truth is always available" principle.
- **[Proposed]** Violation of expectation: teleport a chaser or block, expect a surprise spike; recolour, expect little (LeWM §5.2).
- **[Proposed]** Planner FLOPs next to success (H-JEPA §4.1), not success alone.
- **[Proposed]** Generalisation to the avoid-a-chaser task by changing the cost only, as in PLDM Fig. 5.

## 6. Questions, concerns, blockers (each with a default)

### Questions

| ID | Question | Default |
| --- | --- | --- |
| Q1 | Depend on the H-JEPA and LeWM repos, or port the core ourselves? | **Port** about 300 lines of LeWM (MIT, with attribution). Plain PyTorch, frozen-dataclass configs like `agent_world`; no Lightning, Hydra, wandb or stable-pretraining. Port H-JEPA's level stacking at S5 |
| Q2 | Where does the first learning signal come from, given no reward? | Prediction error and probes at S1 to S2; hand-written intrinsic cost at S3 |
| Q3 | Memory form? | A longer transformer context first; a recurrent belief state only if S2 shows it is needed |
| Q4 | Data: offline only, or online too? | **Offline** from scripted exploration first; online later (rendering cost is the limit) |
| Q5 | Observation: picture (built) or entity list? | Picture; keep the encoder swappable. It is the world journal's open question |
| Q6 | Name the stage-one deliverable. | S1 plus S2: a world model with memory and probes, on `agent_world` |

### Concerns

| ID | Concern | Default |
| --- | --- | --- |
| C1 | The world has no energy, pain or reward. Cost and critic cannot start without them | Ask for them as a world change; see B1 |
| C2 | LeWM reports weaker results in simple, low-diversity worlds (Two-Room), which describes `agent_world` | Diversify data, monitor collapse from day one, keep an IDM term (PLDM, H-JEPA) as a fallback |
| C3 | H-JEPA's levels keep what is predictable, not what matters. A rare hit by a chaser may vanish upward | Add the body channel at every level; probe for chaser and pain at each level |
| C4 | Goal images are ambiguous in an egocentric, rotating view | Do not rely on them; the intrinsic cost is the main objective, the goal term is optional |
| C5 | Compute on a small GPU (4 GB in the world journal) | Small models (about 15M or fewer), 64x48 input, measure before scaling |
| C6 | `HWM_PLDM` has no licence | Read it, do not copy code from it |

### Blockers

| ID | Blocker | Default |
| --- | --- | --- |
| B1 | S0 and S3 need `agent_world` to expose ground truth and a body channel (pain, energy). That repo is not this one | Raise it with Kumar; until then record ground truth from env internals (**[Unverified]** that they are reachable) |
| B2 | Repos are not yet installable here (Python and torch pins differ: H-JEPA wants 3.11 and torch 2.7.1) | Not needed if Q1 default holds; `uv init` with the world's torch index |

## 7. Amendments, 2026-11-15 (appended; earlier text kept)

- **C2 (amended):** the SIGReg weakness in simple worlds is an untested hypothesis from one environment, not a property of `agent_world`. Default unchanged: diversify data, monitor collapse from day one (embedding spread and rank, probes), keep an IDM term as a fallback. See [codebase-study.md](codebase-study.md) section 5.
- **C5 (amended):** compute is now analysed in [compute-options.md](compute-options.md). Target: the full training finishes in 1 to 2 hours; hardware is sized from a measured job size, with a probe first.

## 8. Amendment, 2026-11-15 (appended; earlier text kept)

- **Q1 (amended):** the default "port the LeWM core ourselves" is replaced by reuse: `stable-pretraining` and `stable-worldmodel` (which ships LeWM, SIGReg and an autoregressive predictor). See [stage1-plan.md](stage1-plan.md).
