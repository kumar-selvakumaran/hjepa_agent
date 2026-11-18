# World model: remaining gaps after convergence (stage 1)

State on 2026-11-17: run1 (seed 42) and run2 (seed 43) finished 71,310 steps each, healthy; probes plateau from about step 40,000; seeds agree. Kumar accepted the model as converged "unless there is a major functionality or performance gap left". This document lists every known gap, how it was measured, what would close it, and whether that is worth the effort.
Evidence marks: **[measured]**, **[verified]**, **[documented]**, **[assumed]**. Methods: `docs/journal/evaluation-methods.md`. Iterations: `docs/journal/cloud-iterations.md`.

---

## 1. Summary

| Id | Gap | Kind | Severity for stage 1 (world model) | Severity for later stages (actor, planning) | Close it? |
| --- | --- | --- | --- | --- | --- |
| G1 | Only one-step prediction is validated; multi-step rollouts are unmeasured | Functionality, untested | **Major (unknown)** | **Major**: planning rolls the predictor forward | **Yes, now** (cheap, offline) |
| G2 | Behaviour on novelties (chasers, patch) is unmeasured | Functionality, untested | Moderate | **Major**: runtime novelties are a core principle | **Yes, now** (cheap, offline) |
| G3 | The projector discards about a third of the heading-mod-90 signal | Performance | Minor | Moderate to major, if planning needs fine heading | Measure first (part of G1) |
| G4 | Absolute pose is not identifiable from one frame | Property of the world, not the model | None (not a model fault) | **Major**: the agent must localise | Not in the world model; belongs to short-term memory |
| G5 | Room AUC 0.895, below the gate's 0.95 | Performance | Minor | Minor to moderate | Measure on the backbone; likely data coverage |
| G6 | The gate asks for targets that cannot be read from one frame | Evaluation design | Blocking for "pass" | none | **Yes**: revise the gate (Kumar's decision) |
| G7 | Few room-to-room transitions in the data (3 percent of episodes cross the doorway) | Data coverage | Minor | Moderate | Later, with more or targeted data |
| G8 | Each run is limited by its main process issuing small GPU operations | Efficiency | none | none (cost only) | Later, if training is repeated often |
| G9 | Only two seeds; the spread estimate is rough | Statistical | Minor | Minor | Only if a decision hinges on a small difference |

## 2. Gaps in detail

### G1. Multi-step prediction is unmeasured (major, untested)
- **What is known [measured]:** one-step prediction beats copying the last latent by 13 times (copy ratio 0.076) and depends strongly on the action (sensitivity 21.2).
- **What is not:** error after 2, 5, 10 or more steps fed back into the predictor (autoregressive rollout). Planners (model-predictive control, as in LeWM and PLDM) roll the model forward many steps; small one-step errors can compound **[documented: LeWM and PLDM plan by rollout]**.
- **Effort to close:** an offline evaluation on validation episodes: encode the first frames, roll the predictor forward with the true actions for up to 16 steps, and compare with the encoded true frames. Report error against horizon, against a copy baseline, and against a shuffled-action baseline. About half a day of code; minutes on the GPU.
- **Worth it:** **yes**. It is the most important untested function of a world model, and it is cheap.

### G2. Novelty behaviour is unmeasured (major for later stages, untested)
- **What is not known:** how prediction error and probes behave on `transfer_chasers` (chasers present, bumps) and `transfer_patch_new` (turning patch present), which training never saw.
- **Why it matters:** the agent should notice novelty (surprise) and still represent the scene; this is the basis of the curiosity and safety mechanisms planned later.
- **Effort to close:** run the existing probes with the training normalisation (`--stats-from`) and the one-step and multi-step prediction errors on both transfer sets; compare with the pretraining validation set. Hours of code (mostly reuse); minutes on the GPU.
- **Worth it:** **yes**. Cheap, and it decides whether the world model can serve as a novelty detector.

### G3. Projector bottleneck (performance; significance depends on G1)
- **Measured:** heading modulo 90 degrees, MLP probe R2: all patch tokens 0.88, CLS 0.70, projected latent 0.51, pixels 0.90. The projector discards about a third of the signal the CLS token holds.
- **Documented:** probing a projector's input beats its output, by up to about 30 percent in self-supervised methods ("Guillotine Regularization", Bordes and Balestriero, arXiv 2206.13378).
- **Why it may not matter:** the projected space is what the predictor needs for prediction; the predictor already reaches a copy ratio of 0.076.
- **Why it may matter:** planning toward a goal in the projected space needs heading if the goal depends on heading.
- **Effort to close (only if G1 shows heading errors drive rollout errors):** compare a lower SIGReg weight (the documented sensitive knob, `docs/journal/sigreg-weight.md`), a wider or linear projector, or planning on the CLS token. Two concurrent runs, about 2.5 hours, about $5 to $6.
- **Worth it:** **not yet**. Measure G1 first; act only if rollout errors concentrate on heading.

### G4. Absolute pose needs memory (property of the world)
- **Measured:** absolute heading R2 about 0 and signed y about 0 from latents **and from pixels with an MLP**; heading modulo 90 degrees and abs(y) are readable. Labels are aligned (turn action vs heading change, correlation 0.95).
- **Meaning:** a single frame cannot say which of four symmetric directions the agent faces, or which side of the centre line it is on. No frame encoder can fix this.
- **Effort to close (not in the world model):** short-term memory that integrates actions and frames over time (the STM module in LeCun's architecture). A cheap check now: probe absolute heading from the predictor's history (three frames plus actions); if it still fails, memory over longer horizons is required.
- **Worth it:** the cheap check, **yes** (part of the evaluation suite); the memory module belongs to a later stage.

### G5. Room AUC 0.895 (minor)
- **Measured:** room AUC 0.895 on the projected latent; pixels 0.65; random 0.51; gate threshold 0.95.
- **Likely causes [assumed]:** frames facing a wall look alike in both rooms; doorway views are rare (G7); the projector loses some of it (G3).
- **Effort:** measure on the CLS token and patch tokens (part of the evaluation suite); if the backbone reaches 0.95, the gap is the projector, else data.
- **Worth it:** the measurement, yes; a training change, only together with G3 or G7.

### G6. The gate is ill-posed (evaluation design)
- **Fact:** the gate (INT-83) requires absolute heading and position R2 of at least 0.9, which G4 shows no frame encoder can reach in this world.
- **Proposal (for Kumar's decision):** replace the frame-level absolute targets with identifiable ones, and judge readout on the backbone:
  - cos and sin of 4 x heading, MLP R2 on the CLS token, at least 0.6 and above both baselines;
  - abs(pos_y), at least 0.5; room AUC on the backbone, at least 0.9;
  - keep: completion, copy ratio below 1, action sensitivity above 1, rank, SIGReg;
  - add: multi-step rollout error below the copy baseline up to a stated horizon (G1).
- **Worth it:** **yes**; it changes no training, only what "pass" means.

### G7. Few room transitions (data coverage)
- **Measured:** 3 percent of episodes cross the doorway in 500 steps; doorway windows are 2 percent of training windows (4.5 percent after weighting).
- **Effort:** more episodes, longer episodes, or starts near the doorway; regenerate data and retrain (hours plus about $5).
- **Worth it:** later, if G1 or G2 show failures near the doorway.

### G8. Training efficiency (cost only)
- **Measured:** one run alone keeps the GPU about 50 percent busy; its main process is saturated issuing small operations; two runs reach about 88 percent.
- **Effort:** CUDA graphs or `torch.compile` (recipe unchanged); untested on this model.
- **Worth it:** only if training is repeated often (sweeps); each full run costs about $3.70 alone.

### G9. Seed count (statistical)
- **Measured:** two seeds agree within about 0.03 on probes and 0.4 percent on final prediction loss.
- **Worth it:** a third seed only if a later decision depends on differences near 0.03.

## 3. Proposed next effort

One evaluation suite, built to software-engineering standards (a module, a command-line entry point and tests, reviewed through a pull request), covering G1, G2, G4 (the cheap check) and G5, run on the existing checkpoints on the cloud GPU. No retraining. Then a short decision on G3 and G6.

| Step | What | Cost |
| --- | --- | --- |
| 1 | `src/hjepa_agent/evaluation/` module: rollout error by horizon, transfer-set evaluation, layer probes (patch, CLS, projected), history probes | Code and tests (no cloud) |
| 2 | `scripts/evaluate.py` entry point; results as JSON plus plots per run | Code |
| 3 | Run on run1 and run2 final checkpoints and the transfer sets | Minutes of GPU, about $0.30 |
| 4 | Report in `cloud-iterations.md` (iteration 5); update this document with the measured gaps | none |

## 4. Standard practice for each gap, and the fix proposed here

Kumar's rule (2026-11-17): gaps whose fix would need heuristics or fickle tricks are documented, not implemented. Each row says how significant the gap is, the fix proposed, whether that fix is a standard method or a trick, and what the state of the art does. Source tiers as in the auto-research skill (1 docs or source, 2 papers, 4 community).

| Gap | Significance | Standard practice (state of the art) | Proposed fix | Standard or trick? | Implement now? |
| --- | --- | --- | --- | --- | --- |
| G1 multi-step prediction | Major for planning; size unknown until measured | Train the latent dynamics over several rollout steps so errors do not compound: TD-MPC unrolls the model over multiple steps and credits this with reducing compounding error; TD-MPC2 uses a multi-step latent consistency loss; PlaNet's latent overshooting regularises multi-step predictions (tier 2) | Measure first (evaluation suite); if error grows fast, add a multi-step prediction loss (predict k steps ahead with the predictor's own outputs) | Standard | Measurement yes; training change only after measurement and Kumar's decision |
| G2 novelty | Major for later stages | Prediction error as surprise (ICM, Pathak 2017), random network distillation (Burda 2018), ensemble disagreement (Plan2Explore, Sekar 2020); raw prediction error is fooled by irreducible noise (the "noisy TV"), which disagreement and RND avoid (tier 2) | Measure prediction error on the transfer sets against the pretraining set; a disagreement ensemble belongs to the curiosity stage, not here | Standard | Measurement yes |
| G3 projector bottleneck | Minor now; depends on G1 | Evaluate and reuse the backbone, not the projector output (SimCLR; Guillotine Regularization, Bordes and Balestriero 2022; Xue et al. ICLR 2024) (tier 2); LeWM and PLDM plan in the projected space | Use the CLS token for readout; keep the projected space for prediction; retrain only if G1 shows heading errors dominate | Standard (evaluation choice) | No training change |
| G4 absolute pose | Major for the agent; not a world-model fault | Recurrent or memory-based state estimation: the recurrent state-space model in PlaNet and Dreamer carries a belief over time (tier 2); in LeCun's architecture, the short-term memory module | Build short-term memory in a later stage; now only measure whether three frames already help (history probe) | Standard | Measurement only |
| G5 room AUC | Minor | Probe the backbone; improve data coverage | Measure by layer (evaluation suite); revisit with G7 | Standard | Measurement only |
| G6 gate | Blocks "pass"; no effect on the model | Probe only quantities identifiable from the input; report rollout error and planning success for world models | Revise the gate to identifiable targets and add rollout error (section 2, G6); Kumar decides | Standard | Proposal only |
| G7 data coverage | Minor to moderate | Exploration-driven data collection (Plan2Explore) or uniform state coverage | Targeted starts near the doorway would be a heuristic; prefer exploration-driven data in the curiosity stage | Targeted starts: trick; exploration: standard | No |
| G8 efficiency | Cost only | `torch.compile` and CUDA graphs reduce per-step launch overhead (tier 1, PyTorch documentation) | Try only if many runs are planned | Standard | No |
| G9 seeds | Minor | Three or more seeds with intervals for claims near the noise level | Add a seed only when a decision depends on it | Standard | No |

Sources: [TD-MPC, arXiv 2203.04955](https://arxiv.org/pdf/2203.04955.pdf); TD-MPC2 (Hansen et al. 2023), described in [DC-MPC, arXiv 2503.00653](https://arxiv.org/pdf/2503.00653); PlaNet latent overshooting, described in [World Models survey, arXiv 2606.00133](https://arxiv.org/pdf/2606.00133); [Plan2Explore, arXiv 2005.05960](https://arxiv.org/pdf/2005.05960v2.pdf); [Guillotine Regularization, arXiv 2206.13378](https://arxiv.org/pdf/2206.13378); [Xue et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/a5554e55d7a21e62d0d7a028ec0ea1c7-Abstract-Conference.html). ICM and RND are summarised from secondary search results and not read in full **[unverified details]**.

**Status of the measurement (section 3):** the evaluation suite is in PR #2 and running on the cloud GPU for run1 and run2; results will be appended here.

## 5. Measured (evaluation suite, PR #2, run on the cloud GPU 2026-11-17)

Final checkpoints of run1 (seed 42) and run2 (seed 43); 512 validation sequences for rollouts, 128 per transfer set; 20,000 / 5,000 probe frames. Raw results: `scratch/auto-research/2026-11-17-lewm-stage1/runs/<run>/evaluation.json` and `rollout.png`.

**G1, rollout error relative to copying the last latent (below 1 is better than copying) [measured]**

| Rollout step (2 env steps each) | 1 | 4 | 7 | 10 | 13 | 16 |
| --- | --- | --- | --- | --- | --- | --- |
| run1 | 0.069 | 0.145 | 0.245 | 0.335 | 0.408 | 0.488 |
| run2 | 0.069 | 0.148 | 0.246 | 0.326 | 0.407 | 0.477 |

- The model beats copying at every horizon up to 32 environment steps; its error grows about 7 times from step 1 to 16, roughly linearly.
- Actions still matter at long range: shuffled-action error over true-action error is 22.7 at step 1 and 2.2 at step 16 (run1; run2 22.8 and 2.3).
- **Verdict: moderate, not blocking.** Compounding is present but bounded; LeWM-style planners plan a few steps ahead. A multi-step loss (standard, section 4) is the fix if planning in a later stage shows long-horizon errors matter.

**G2, novelty [measured]:** one-step latent error on transfer sets vs pretraining validation: chasers 0.031 vs 0.012 (2.6 times), patch 0.018 vs 0.012 (1.5 times); relative to copying, 0.143 and 0.106 vs 0.069 (run1; run2 alike). **Verdict: the world model registers both novelties as higher prediction error**, the basis for a surprise signal. Standard caveat: raw error is fooled by irreducible noise; a disagreement ensemble is the later-stage fix.

**G3 and G5, readout by layer (run1 / run2) [measured]**

| Layer | cos 4 x heading, MLP | abs(pos_y), MLP | pos_x, MLP | Room AUC |
| --- | --- | --- | --- | --- |
| All patch tokens | 0.88 / 0.78 | 0.50 / 0.49 | 0.29 / 0.35 | 0.888 / 0.910 |
| Mean patch token | 0.68 / 0.53 | 0.57 / 0.55 | 0.37 / 0.45 | 0.896 / 0.915 |
| CLS | 0.70 / 0.66 | 0.58 / 0.59 | 0.41 / 0.42 | 0.905 / 0.913 |
| Projected latent | 0.52 / 0.48 | 0.57 / 0.57 | 0.35 / 0.37 | 0.895 / 0.900 |
| Pixels | 0.90 | 0.43 | 0.08 | 0.649 |

- G3 confirmed in both seeds: the projector loses about a quarter to a third of the heading-mod-90 signal; the CLS token keeps more. **Verdict: minor**; use the CLS token for readout.
- G5: room AUC is 0.89 to 0.915 at every layer, so the shortfall against 0.95 is not the projector; it is the data (rare doorway views, G7) or the frames themselves (walls look alike in both rooms). **Verdict: minor.**
- Latents beat pixels on abs(pos_y), pos_x and room; pixels beat latents only on heading mod 90 with an MLP (patch tokens come close in run1).

**G4, history of three frames plus actions [measured]:** absolute heading R2 -0.13 / -0.15, signed y -0.11 / -0.12. **Verdict: confirmed property**: three frames (six environment steps) cannot localise; memory over longer horizons is required (short-term memory, later stage).

## 6. Conclusion

No gap blocks stage 1. The world model is converged by INT-96 and functional for its role: one-step prediction 13 times better than copying, bounded multi-step compounding, strong action dependence, and higher prediction error on novelties. The significant gaps (G1 long horizons, G4 localisation, G2 robust novelty) belong to the planning, memory and curiosity stages, each with a standard fix listed in section 4. The gate (G6) needs Kumar's decision.

## 7. Added 2026-11-18: G10, a world shortcut behind the heading readout

| Gap | Kind | Significance | Decision |
| --- | --- | --- | --- |
| G10 | The checkerboard floor exposes heading modulo 90 degrees and position modulo the checker period from any frame (a global compass the real world lacks) | **Major for interpretation**: part of what the encoder learned (heading mod 90, R2 0.88 in patch tokens) comes from this shortcut, not from transferable perception | Handed to agent_world (INT-99): a floor with no global frame, reseedable, checkerboard kept as an option; hjepa_agent keeps its checkerboard-trained models for later comparison |

- Consequence for G3 and G6: heading modulo 90 degrees may stop being a valid perception target once the floor changes; the gate waits.
- Consequence for evaluation (INT-100): probe egocentric quantities and pose relative to the episode start.
