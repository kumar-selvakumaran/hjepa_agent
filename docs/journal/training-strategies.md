# Training strategies

Defined 2026-11-15. Labels: see [README.md](README.md). Evidence is from the four reference papers (see [codebase-study.md](codebase-study.md)) and the data probes in [world-exploration.md](world-exploration.md).

## 1. The two strategies

| | A: joint (LeWM style) | B: staged (Kumar's inclination) |
| --- | --- | --- |
| Status | **[Kumar]** stated as one option | **[Kumar]** stated as the current inclination |
| Phase 1 | none | Pre-train the encoder by masked latent prediction on single frames |
| Phase 2 | Encoder and action-conditioned predictor trained together on trajectories | Freeze the encoder; train the action-conditioned predictor alone |
| Phase 3 | none | When the predictor has converged, fine-tune everything together |
| Anti-collapse | SIGReg on the latents (LeWM), one weight | Phase 1 needs its own mechanism; phase 3 needs one too |
| Costly part | Encoder runs on every frame, every step | Encoder runs once per frame to cache latents in phase 2; full cost again in phase 3 |

## 2. Assessment of B

### In its favour

- **Cheap predictor training. [Proposed]** With a frozen encoder, latents can be cached once (1M frames). The predictor is about 10M parameters on 3 to 4 tokens per sample, so phase 2 is small next to the encoder. This helps the 1 to 2 hour target.
- **Stability. [Proposed]** Phases are separable; each has a clear check. Frozen encoders are how DINO-WM avoids collapse.
- **Precedent. [Verified]** H-JEPA has a stagewise variant (train level 1, freeze, train level 2) but used it only for analysis; headline results are end to end. DINO-WM (frozen encoder) vs LeWM (end to end) is the nearest comparison: LeWM matched or beat it on Push-T and Reacher, and lost on Cube. Not a controlled test.

### Concerns

- **C1. Masked latent prediction has its own collapse problem.** I-JEPA and V-JEPA use an EMA target encoder and stop-gradient, which LeWM calls heuristics without a well-defined objective. Alternative: masked prediction plus SIGReg (LeJEPA, as H-JEPA cites it), no EMA. Default: masked prediction + SIGReg.
- **C2. Single-frame pre-training sees no time or actions.** It rewards features predictable from spatial context: floor, walls, layout. Our pixel variance is mostly egocentric motion, and chasers, bumps and the patch are small and rare ([world-exploration.md](world-exploration.md) section 5). H-JEPA found that a purely predictive objective can keep the scene and drop the agent, fixed by an inverse-dynamics term. Default: pre-train over short clips, not single frames (space-time masking), or add an inverse-dynamics head, and gate phase 2 on probes.
- **C3. Fine-tuning can undo a good encoder.** Unfrozen, the predictor's gradient can reshape or collapse the encoder. Default: keep the phase-1 loss and SIGReg as an auxiliary during fine-tuning, with a low encoder learning rate (H-JEPA gives upper levels one fifth of the base rate).
- **C4. Total cost is not obviously lower.** Phase 3 costs about the same per epoch as strategy A. B saves time only if fine-tuning is brief or skipped. **[Unverified]**
- **C5. No head-to-head evidence on our data.** Default: run both on the same data and compare (section 3).

## 3. Proposed comparison

- **[Proposed]** Same dataset, same encoder size, same compute budget, for A and for B up to phase 2, plus B after phase 3.
- **Measures, all from the ground truth that the world exposes:**
  - Linear and MLP probe error for agent pose, heading, chaser position, food and the patch, per entity.
  - Embedding spread and rank (collapse check).
  - Open-loop prediction error at 1, 5 and 20 steps against a copy-last-latent baseline.
  - Predictor action sensitivity, and the surprise spike when a chaser teleports.
  - Training time and planner compute.
- **[Proposed]** A gate between phases: do not start phase 2 until the phase-1 latents recover chaser position, heading and patch presence above a set threshold.

## 4. Data strategy (from the probes)

| Need | Why | Proposal |
| --- | --- | --- |
| Reach room 2 and cross the patch | 2% of random episodes reach it | A doorway-seeking policy and longer episodes |
| Bumps and pain events | 0.07% of steps | Chaser-approach episodes; oversample or reweight |
| Diverse layouts | 14% of variance is between episodes | Keep per-reset randomisation; consider more factors |
| Speed of generation | 1M transitions about 2 h locally with chasers | Test larger batches; generate on the rented GPU |
| Both rooms, both policies | Coverage, not volume, drives learning in PLDM | Mix policies, as PLDM mixes random and expert trajectories |

## 5. Questions, concerns, blockers

| ID | Item | Default |
| --- | --- | --- |
| Q1 | Phase-1 anti-collapse: SIGReg, EMA with stop-gradient, or both? | SIGReg, no EMA |
| Q2 | Phase-1 input: single frames, or short clips with space-time masking? | Short clips |
| Q3 | Run A as a baseline next to B? | Yes, same data and budget |
| Q4 | Include actions in phase 1 (an inverse-dynamics head)? | Yes, with a small weight |
| Q5 | Which policies make up the dataset, and how long are episodes? | Smooth walk, doorway-seeking, chaser-approach; 500+ steps |
| Q6 | Vary world parameters per episode (patch rate, layout)? | Layout only at first |
| C-new | Latent-caching for phase 2 assumes the encoder is frozen; augmentations cannot be applied on the fly | Accept for now |
| B1 (from before) | Ground truth is available from `scenario.scene()` in the clone; a body channel (pain, energy) still does not exist | Keep as an open request to the world |

## 6. Pivot, 2026-11-16 (appended; earlier text kept)

**[Kumar]** Strategy A (joint, LeWM style) is adopted; strategy B is set aside (INT-70). Section 2's assessment of B and section 3's comparison are history. See [stage1-spec.md](stage1-spec.md) section 9.
