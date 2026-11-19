# Intentions register

Key intentions and principles, kept append-only. Backfilled 2026-11-15 from this session's record. Labels: see [README.md](README.md). Add to it whenever an intention is stated or noticed; never delete (mark a superseded entry as **[Superseded by INT-xx]** and add the new one).

| Label | Meaning |
| --- | --- |
| **[Kumar]** | Stated by Kumar |
| **[Noticed]** | Inferred by Claude from behaviour or answers; not yet confirmed |

## 1. Project and architecture

| ID | Date | Intention | Label |
| --- | --- | --- | --- |
| INT-01 | 11-15 | Build an extendable agent following LeCun's *A Path Towards Autonomous Machine Intelligence*, acting in `agent_world`. Modules as in the HJepa-Agent diagram: Configurator, Perception, World Model, Short Term Memory, Actor, Cost (intrinsic cost + Critic) | **[Kumar]** |
| INT-02 | 11-15 | Stage one was building the world (`agent_world`); this repo is stage two, the agent | **[Kumar]** |
| INT-03 | 11-15 | Same development principles and behavioural expectations as the `agent_world` master session | **[Kumar]** |
| INT-04 | 11-15 | Study the reference papers and repos (H-JEPA, LeWM, PLDM, HWM) critically and resourcefully, and keep notes | **[Kumar]** |
| INT-05 | 11-15 | The SLAM-style online alignment idea is **tabled**, not dropped | **[Kumar]** |
| INT-06 | 11-15 | Inherited from the world: it is a survival game with intrinsic costs (hunger, pain avoidance, curiosity, discomfort with uncertainty); general, simple, extendable; north-star light-and-mirror puzzle | **[Kumar]** (in `agent_world` journal) |

## 2. Training strategy

| ID | Date | Intention | Label |
| --- | --- | --- | --- |
| INT-10 | 11-15 | Strategy B: pre-train the perception encoder by masked latent prediction; freeze it; train the action-conditioned predictor (LeWM style) on the frozen encoder; when the predictor converges, fine-tune everything jointly. Strategy A (joint, LeWM style) stays as a comparison | **[Kumar]** |
| INT-11 | 11-15 | **Planning is kept out of the world model.** The world model does world modelling only; no notion of goals, costs or planning in it. Planning enters later, when the actor is trained | **[Kumar]** |
| INT-12 | 11-15 | Novel nuances are introduced to the agent at runtime, online, not in pretraining: chaser bumps and the turning patch | **[Kumar]** |
| INT-13 | 11-15 | Explicit consensus before generating data or training: architecture, dataset curation, preprocessing, strategy, hyperparameters | **[Kumar]** |
| INT-14 | 11-15 | Run the stage-1 training on a decent compute platform in about 30 to 60 minutes, to get used to the platform for larger jobs later | **[Kumar]** |

## 3. Data

| ID | Date | Intention | Label |
| --- | --- | --- | --- |
| INT-20 | 11-15 | Use data that would be available in real life; do not engineer critical events into pretraining data | **[Kumar]** |
| INT-21 | 11-15 | Pretraining world: chasers off, rotating patch off entirely (no disc, no forced actuation), so no bumps and no spin | **[Kumar]** |
| INT-22 | 11-15 | Exploration is a **random walk**, an explicit design decision, dispersed throughout the world; it includes stopping and reversing, with varied speeds | **[Kumar]** |
| INT-23 | 11-15 | Start poses uniform over free space in both rooms, random heading | **[Kumar]** |
| INT-24 | 11-15 | 1M transitions as 2,048 episodes x 500 steps, 10% of episodes held out; all other world content stays at defaults; keep all frames, no filtering | **[Kumar]** |
| INT-25 | 11-15 | Per-channel dataset mean/std normalisation; no augmentation | **[Kumar]** |
| INT-26 | 11-15 | Encoder uses patches of 6x8 pixels (64 tokens) first; stage 2 reads data with a thin loader first | **[Kumar]** |

## 4. Ways of working

| ID | Date | Intention | Label |
| --- | --- | --- | --- |
| INT-30 | 11-15 | Minimise from-scratch code; reuse stable packages; be critical before writing code | **[Kumar]** |
| INT-31 | 11-15 | Run independent work in parallel and delegate to subagents wherever subtask and data independence allow | **[Kumar]** |
| INT-32 | 11-15 | Surface questions through the question tool: default first, other options, room for notes | **[Kumar]** |
| INT-33 | 11-15 | Costly, risky or critical-environment commands follow the `hitl-dev` skill: Claude does groundwork and writes a step file; Kumar runs the steps | **[Kumar]** |
| INT-34 | 11-15 | Log key intentions as they appear, in this register | **[Kumar]** |
| INT-35 | 11-15 | One branch per effort off `main`, small atomic commits that pass tests, docs append-only, probes in `scratch/`, indexed Q/C/B, journey updated after each effort, code machine-independent, crisp list-and-table explanations | **[Kumar]** (inherited) |

## 5. Noticed, to confirm

| ID | Date | Intention | Label |
| --- | --- | --- | --- |
| INT-40 | 11-15 | Kumar prefers settling choices explicitly over accepting my defaults; defaults are proposals, not decisions. I stopped data generation and building when consensus was missing | **[Noticed]** |
| INT-41 | 11-15 | Prefers answering by selecting options with room for notes, but rejected a four-question batch that mixed method, size, latent and recipe; smaller, clearer batches may suit | **[Noticed]** |
| INT-42 | 11-15 | Privileged information used only to generate or evaluate data (such as ground-truth pose) is acceptable only if it does not leak notions like goals into the model; to confirm | **[Noticed]** |
| INT-43 | 11-15 | The data should vary in more than layout over time (actions, speed, start pose) to avoid a narrow distribution; to confirm the intent behind D5 | **[Noticed]** |

## 6. Confirmations and updates, 2026-11-15 (appended)

| ID | Date | Intention | Label |
| --- | --- | --- | --- |
| INT-42a | 11-15 | Confirmed: privileged information such as ground truth is acceptable for generating and evaluating data (INT-42) | **[Kumar]** |
| INT-43a | 11-15 | Confirmed and extended: the data should be uniformly distributed over the world's states, and also over affordances, actions and situations, not only layouts | **[Kumar]** |
| INT-41a | 11-15 | Corrects INT-41: the four-question batch was fine; its closure was accidental. Batches of up to four questions are acceptable | **[Kumar]** |

| INT-50 | 11-15 | Encoder: timm ViT-tiny, 12 layers, width 192, 3 heads, 5.5M parameters, patch 6x8 | **[Kumar]** |
| INT-51 | 11-15 | Stage 2 uses LeWM's per-frame latent as the baseline (one vector, mean of patch tokens) | **[Kumar]** |
| INT-52 | 11-15 | Attentive pooling of the frozen tokens is liked but not adopted now: it is a potential **configurator** role, to be documented (not built). Kumar asked for a side-effort document of each component's expected roles and responsibilities (diagram plus LeCun's paper) with a section of precise potential integrations that includes this configurator role | **[Kumar]** |
| INT-53 | 11-15 | Open to SIGReg combined with masked latent prediction as the pretraining objective; asked for my view | **[Kumar]** |
| INT-54 | 11-15 | Pretraining objective: **SIGReg with masked latent prediction only**, chosen over the shipped I-JEPA with EMA, after being told the combination is untested in the sources I found | **[Kumar]** |
| INT-55 | 11-15 | Random walks may be jittery; the aim is good representation of every case, not smooth paths | **[Kumar]** |
| INT-56 | 11-15 | Uniformity approach: uniform start poses, then measure coverage (pose, action, situation classes) and reweight at training time | **[Kumar]** |
| INT-57 | 11-15 | Compute platform: simple, tried and tested, reliable, made for ML training, not overkill; scalable (later, not now) up to about 40B-parameter training across multiple GPUs. The experience gained should carry over to larger jobs | **[Kumar]** |
| INT-58 | 11-15 | Communicate questions and explanations in complete sentences, defining any term I coin | **[Kumar]** |
| INT-59 | 11-15 | Training recipe for the SIGReg-masked method: a pilot first (three runs of 5 epochs with SIGReg weights 0.03, 0.09 and 0.3), then the full 40-epoch run | **[Kumar]** |
| INT-60 | 11-15 | Wants to know the precise mathematical reason, if any, why SIGReg might not work with masked latent prediction, before committing to it; and the reasoning for the projector | **[Kumar]** |
| INT-61 | 11-15 | Early exit with a saved checkpoint if collapse is detected during training | **[Kumar]** (confirmation of my reading pending) |
| INT-62 | 11-15 | Random-walk hold length: each episode draws its own hold length, from a new random action every step up to about 2.5 s; other fixed parts as specified (speed scale 0.3 to 1.0, forward uniform in [-0.5, 1.0] x scale, 15% pauses, turn uniform in [-1, 1]) | **[Kumar]** |
| INT-63 | 11-15 | On collapse, stop early and keep the last good checkpoint; after training apply the probe gate; if the gate fails, tell Kumar and propose options before spending more compute (no automatic second run) | **[Kumar]** |
| INT-64 | 11-15 | Platform: Lambda | **[Kumar]** |
| INT-65 | 11-15 | Hesitation about masked latent prediction itself: the visible tokens may carry no information about the masked ones, because signals may be independent at the masking granularity. The projector decision is on hold until this is resolved; a measurement study was launched | **[Kumar]** |
| INT-66 | 11-15 | Wants the perception pretraining to be a stable latent-prediction type (JEPA style); asked for a rigorous survey of the whole landscape, in simply-explain format | **[Kumar]** |

## 7. Pivot, 2026-11-16 (appended)

| ID | Date | Intention | Label |
| --- | --- | --- | --- |
| INT-70 | 11-16 | **Pivot:** train the encoder and the action-conditioned predictor **jointly from scratch, LeWM style** (strategy A). Reason: among all the pretraining schemes surveyed, each has benefits and drawbacks and none matches exactly what is wanted, so the best course for now is LeWM-style joint training. **Supersedes** INT-10 (staged pretraining, frozen encoder, joint fine-tuning), INT-54 (SIGReg with masked latent prediction), INT-59 (the pilot designed for that objective) and makes INT-65 (doubt about masked prediction) moot | **[Kumar]** |
| INT-71 | 11-16 | Vision transformer patch 8 by 8 pixels (a 6 by 8 grid, 48 tokens) on frames of 48 by 64 | **[Kumar]** |
| INT-72 | 11-16 | Frame latent: the CLS token through the MLP-with-BatchNorm projector, exactly as LeWM does (supersedes INT-51 and the mean-of-patch-tokens note) | **[Kumar]** |
| INT-73 | 11-16 | Frameskip 1 for the training windows (not LeWM's 5): consecutive frames 0.1 s apart, a window of 4 frames spans 4 steps, actions are packed per single step | **[Kumar]** |
| INT-74 | 11-16 | Wants a rigorous investigation of how difficult the SIGReg weight is to choose, what the papers say about its nuances, and how other users choose it, before deciding the recipe | **[Kumar]** |
| INT-75 | 11-16 | Asked whether the number of actions per transition (frameskip) and the number of frames per sample can be dynamic instead of fixed; exploring variable temporal structure for the world model | **[Kumar]** (question; no decision yet) |
| INT-76 | 11-16 | Fixed frameskip of **2** for the training windows (supersedes INT-73's frameskip 1): frames 0.2 s apart, two commands packed into one four-number action, a 4-frame window spans 7 steps. Start low; dynamic frameskip is a later experiment (INT-75) | **[Kumar]** |
| INT-77 | 11-16 | Training recipe, SIGReg weight and length: use exactly what LeWM uses (no sweep). Interpreted as: weight 0.09 (the repository config and the peak of LeWM's own sweep; the paper text says 0.1), 1,024 projections, 17 knots, 10 epochs (the paper; the repo default is 100), batch 128, AdamW with learning rate 5e-5 and weight decay 1e-3, bf16, gradient clip 1.0, warmup then cosine, predictor 6 layers, 16 heads, width 192, dropout 0.1. Supersedes the pilot sweep idea (INT-59) | **[Kumar]** (interpretation to confirm) |
| INT-78 | 11-16 | Data reaches Lambda by generating locally and uploading compressed | **[Kumar]** |
| INT-79 | 11-16 | GPU: whichever single GPU is available, with comfortable capacity (interpreted as at least 24 GB memory and recent compute); spending caps of $10 for the session and 90 minutes for the training run unless Kumar says otherwise | **[Kumar]** (cap interpretation to confirm) |
| INT-80 | 11-16 | Code reaches Lambda as a `git archive` tarball by `scp`; the world is installed from its public GitHub repository; nothing is pushed | **[Kumar]** |
| INT-81 | 11-16 | A short local run first, and fail-fast tests and preflight checks that catch errors early on the cloud machine too | **[Kumar]** |
| INT-82 | 11-16 | Uniformity: inverse-frequency importance weights over position cells, heading bins and situation classes (capped at 10 times), recorded from ground truth. Because this and similar capabilities exist only because we work in simulation, **explicitly and precisely document every such simulation privilege** (see [simulation-privileges.md](simulation-privileges.md)) | **[Kumar]** |
| INT-83 | 11-16 | Early-exit and exit-gate thresholds confirmed as proposed: early exit if the SIGReg statistic stays far above its healthy floor (about 1.0 to 1.4) or the effective rank stays under 25% of the width on several consecutive checks; gate needs R squared at least 0.9 for position and heading, room area under the curve at least 0.95, a food-visibility probe, both baselines beaten, and prediction error below the copy-the-last-latent baseline | **[Kumar]** |
| INT-84 | 11-16 | Two transfer sets of 128 episodes x 500 steps, never used in training: chasers on with the patch off, and the patch on with chasers off | **[Kumar]** |
| INT-85 | 11-16 | Small defaults confirmed: LeWM's predictor as is, seed 42, plain CSV logs under the outputs folder with no wandb, surprise test deferred until after the first run | **[Kumar]** |
| INT-86 | 11-16 | Granular monitoring of the cloud run (out-of-the-box where possible), and checkpointing so that even failed runs leave usable results (never empty-handed) | **[Kumar]** |
| INT-87 | 11-16 | Dashboard: TensorBoard over an SSH port-forward, with CSV logs always written | **[Kumar]** |
| INT-88 | 11-16 | Outputs and checkpoints go to a Lambda persistent filesystem, plus a pull step to Kumar's machine at the end and at any failure | **[Kumar]** |

## 8. Remote and HITL files (2026-11-16, appended)

| ID | Date | Intention | Label |
| --- | --- | --- | --- |
| INT-89 | 11-16 | **Get Kumar's confirmation before pushing any docs** (and, by the same reasoning, before pushing anything outward). Nothing from this project has been pushed; the remote holds only the initial commit | **[Kumar]** |
| INT-90 | 11-16 | HITL files (step files, their helper scripts and outputs) live in `scratch/hitl/`, which is gitignored; they are not committed or pushed. The commit that added them was removed from local history (it was never on the remote, so no force push was needed) | **[Kumar]** |
| INT-91 | 11-16 | Branching and pushing principle (refines INT-89): large significant efforts go into a corresponding branch and are merged into `main` after testing; smaller commits that are not logically additive or significant may go straight to `main` for now; **safe, verified, secret-free code may always be pushed**; **journals must be confirmed by Kumar before they are included in git** | **[Kumar]** |
| INT-92 | 11-16 | Keep all local commits local for now and review the journals together before any push of `main`; from now on, journal and docs changes stay uncommitted until Kumar confirms them (code may still be committed on branches and merged to `main` after testing) | **[Kumar]** |
| INT-93 | 11-17 | The first cloud run may take up to about 170 minutes (the throughput test projected 143 for 10 epochs at 8 data-loader workers, against the earlier 90-minute cap); the session cost cap of $10 stays; the loader is not tuned first | **[Kumar]** |
| INT-94 | 11-17 | Raises INT-93: Kumar accepts a training run of up to 180 minutes; the internal timer is set to 180 minutes (hard timeout 185), the $10 session cap stays | **[Kumar]** |
| INT-95 | 11-17 | While the collapse stop is diagnosed, run floor(capacity / job usage) concurrent jobs on the instance and use them to test the hypotheses; the binding resource is GPU compute (one job used 42 to 54 percent), so two jobs: run1 resumed and run2 fresh with seed 43 | **[Kumar]** |
| INT-96 | 11-17 | Auto-research loop: iterate until the world model converges or Kumar says stop; budget cap $40 in total for the cloud work (unless he stops earlier); converged means a run finishes without collapse, validation prediction loss and probe scores plateau over the last checkpoints, two seeds agree within their spread, and probes beat the pixel and random baselines; the gate's R2 of 0.9 is reported, not required | **[Kumar]** |
| INT-97 | 11-17 | Code reaches GitHub through a code-only branch `code` (docs/ stripped from its history), kept in the sibling clone `hjepa_agent_worktrees/code-only` and updated after each tested code merge by `hjepa_agent_worktrees/sync_code_branch.sh`; first push 2026-11-17 (26 commits, local main up to 6e3067e); local main and the journals stay local until reviewed | **[Kumar]** |
| INT-98 | 11-17 | Pull-request workflow from now on (CI not yet present): the integration branch on GitHub is `develop` (git-flow name; same code-only history as `code`, which stays); each effort is a `feature/<name>` or `fix/<name>` branch in the sibling clone `hjepa_agent_worktrees/code-only`, off `develop` or stacked on the previous open PR's branch, pushed with a PR into `develop` (or into the branch it stacks on) that Kumar reviews and merges; auto-research deploys and tests from the branch tip without waiting for review and cites the PR; after a merge, the change is cherry-picked into the local main (which also holds the journals) | **[Kumar]** |
| INT-99 | 11-18 | World realism principle: the world exposes signals the way the real world does, with less complexity; localisation is done without a compass, from memory, co-existing static objects and self-motion. The checkerboard floor is a shortcut (a compass modulo 90 degrees and a position lattice); a proposal for agent_world is in `scratch/handoff/agent_world-signal-shortcuts.md` (not in git) (walls and box obstacles stay; a fixed floor current becomes a future mode; actuation noise is future scope; floor reseeding as a world parameter; the checkerboard is kept as an option for later comparisons; nothing in hjepa_agent is discarded) | **[Kumar]** |
| INT-100 | 11-18 | hjepa_agent evaluation moves to egocentric quantities and pose relative to the episode start, instead of world-axis labels (agreed C2) | **[Kumar]** |
| INT-101 | 11-18 | Journals may be committed and pushed without Kumar's review, provided they hold no secrets or identifying information; Claude verifies that before every push (refines INT-89, INT-91, INT-92). Journals reach GitHub as `docs/` pull requests into `develop` | **[Kumar]** |
| INT-102 | 11-18 | Retrain stage 1 on agent_world's map floor (its `develop`, commit 860ce30): same LeWM recipe, four new pretraining shards `map_pretrain_200` to `203` (same sizes and policy as before), transfer sets `map_transfer_chasers`, `map_transfer_patch` and `map_transfer_floor1` (a held-out floor map); two seeds concurrently (`map1` seed 42, `map2` seed 43), 180-minute cap; the checkerboard data and runs are kept for comparison | **[Proposed]**, prepared so the job can launch on 2026-11-19 |
| INT-103 | 11-19 | Continuation runs are allowed if more epochs would help (Kumar). Decided by a rule recorded before the runs ended: (a) a late val-loss drop beyond the checkerboard's and beyond twice the seed spread, or (b) a probe trend still rising beyond the seed spread. Both failed, so the map runs were not continued. `--init-from` (warm restart) exists for later use (PR #5) | **[Kumar]** permission; rule by Claude |
