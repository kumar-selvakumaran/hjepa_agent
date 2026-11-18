# Journey

Append-only. Labels: see [README.md](README.md).

## 1. Setup (2026-11-15)

- **[Kumar]** Stage 1 built the world (`agent_world`). Stage 2 is an extendable agent following LeCun (2022).
- **[Kumar]** Same development principles and behaviour as the `agent_world` master session.
- **[Built]** Copied the world's journal and README into `docs/agent_world/` (reference snapshot, see its `SOURCE.md`).
- **[Built]** Copied the Lucid diagram "HJepa-Agent" into `docs/diagrams/` (modules: Configurator, Perception, World Model, Short Term Memory, Actor, Cost = intrinsic cost + Critic).
- **[Built]** `CLAUDE.md`, `.gitignore`, `.dockerignore`, and project memory carried over. `scratch/` is gitignored.

## 2. Study of four reference codebases (2026-11-15)

- **[Kumar]** Cloned into `scratch/`: H-JEPA, LeWM, PLDM, HWM_PLDM. Read the papers and the code, with a critical eye.
- **[Built]** Findings in [codebase-study.md](codebase-study.md). Proposal in [agent-proposal.md](agent-proposal.md), all **[Proposed]**.
- Headline finding: the four codebases give the **World Model + Perception + a goal-image cost + a planner**. They give none of **Short Term Memory, Critic, intrinsic cost, Configurator, or a reactive Actor**; those are the project's real work.
- Nothing was run: no install, no training. Everything is from reading.

## 3. Correction on SIGReg and simple worlds (2026-11-15)

- **[Kumar]** Asked whether LeWM would do better in a complex real world than in a simple one.
- **[Agreed]** My earlier claim, that LeWM's weaker Two-Room result "describes `agent_world`", was overstated. It is the authors' untested hypothesis from one environment; on 3D Cube, DINO-WM beat LeWM (86% vs 74%), so complexity does not predict LeWM's standing. For `agent_world` it is a risk to monitor (embedding spread and rank, latent probes), not a prediction. Appended to [codebase-study.md](codebase-study.md) and [agent-proposal.md](agent-proposal.md).

## 4. Compute and cost (2026-11-15)

- **[Kumar]** Hoped to train locally in about 4 to 5 hours on a 4 GB GPU; asked for training options, reliability and cost; then set the target: **the full training finishes in 1 to 2 hours wall-clock**, whatever compute that needs.
- **[Agreed]** Defaults for Q1 and Q2 of that thread: the 1 to 2 hours covers training only (data staged beforehand), and it means one full run.
- **[Built]** [compute-options.md](compute-options.md): what the reference papers trained on (data in GB, model in MB, GPU, time), what machine a run needs, rental prices, and a sizing rule.
- Finding: the papers give almost no wall-clock times and name no small GPU; the only named cards are a V100 and H200s.
- **[Open]** The real job size is unmeasured. Next: a throughput probe locally, then about 10 minutes on a rented GPU.

## 5. Exploring the world (2026-11-15)

- **[Kumar]** Asked for the world cloned into `scratch/`, data generated from it, the variance within a time step examined, and artifacts saved, including images at time t and t+1.
- **[Built]** Cloned `agent_world` (f09f406) and installed it with its own environment. Generated three small datasets (64 episodes x 200 steps) and measured variance, coverage and throughput. Findings in [world-exploration.md](world-exploration.md); figures in `../media/`.
- Headline: trivial collapse looks unlikely (variance 0.031, 19% of pixels change per step), but the view's variance is mostly egocentric motion; chasers, bumps (0.07% of steps) and the patch are small and rare; random walks reach room 2 in 2% of episodes; data generation runs at about 143 world-steps/s with chasers on this GPU.
- **[Built]** Refreshed the `agent_world` snapshot: the sibling journal gained sections on survival drives and intrinsic costs, a design inclination, contenders and stable-worldmodel notes.

## 6. Training strategies (2026-11-15)

- **[Kumar]** Two strategies: A, train the encoder and the action-conditioned predictor jointly on trajectories (LeWM); B, pre-train the encoder by masked latent prediction, freeze it, train the predictor, then fine-tune everything. Currently inclined to B.
- **[Built]** [training-strategies.md](training-strategies.md): assessment, concerns (masked prediction needs its own anti-collapse mechanism; single frames may miss what matters; fine-tuning can undo the encoder), a proposed A-versus-B comparison with probes, and a data strategy. Questions are open; nothing is decided.

## 7. Online learning as SLAM-style alignment (2026-11-15)

- **[Kumar]** Idea: for online or continual learning, treat the world-model predictor with actions on SO(2) like SLAM: use its predictions and the actions for joint alignment of embeddings, and fine-tune the world model to improve that alignment. Asked whether work exists and what is missing.
- Searched (summaries only; **[Unverified]**, papers not read): **J-LAW** (arXiv 2606.28712) couples pose nodes and latent nodes in one factor graph with loop closure in both spaces; **JEPA-TTT** (2610.00722) and **AdaJEPA** (2606.32026) adapt a latent predictor online; **LatentSLAM** (2105.03265) and **Egocentric Spatial Memory** (1807.11929) are older learned-latent mapping work. None is the exact combination: online fine-tuning of the world model driven by loop-closure alignment.
- Gaps raised: actions are not realised motion (patch, bumps); loop-closure matching under aliasing and dynamic entities; trivial alignment through collapse; drift of stored latents when the encoder changes (a point for a frozen encoder, strategy B); the actions form SE(2), not SO(2).
- **[Open]** Not decided. Cheap first step proposed: measure revisit rates and test cycle consistency on the existing data before any SLAM machinery.

## 8. Stage 1 preparation and a core principle (2026-11-15)

- **[Kumar]** Stage 1 is pre-training the perception module (masked latent prediction), then a LeWM-style world model with the frozen encoder, then joint fine-tuning. SLAM discussion tabled.
- **[Kumar]** Core principle, stated mid-task: keep from-scratch code to a minimum, reuse stable packages, be critical about adding code. Recorded in memory, in CLAUDE.md and in [stage1-plan.md](stage1-plan.md).
- **[Built]** Installed `stable-pretraining` 0.1.8 and `stable-worldmodel` 0.1.1. They cover most of stage 1 (I-JEPA, probes, rank monitor) and stage 2 (a LeWM class, SIGReg, an autoregressive predictor). My earlier plan to port LeWM by hand is dropped, as are a hand-written probe module and dataset class.
- **[Built]** `agent_world` is a pinned git dependency. Added scripted exploration policies and a data generator. A pilot of 307,200 transitions with the waypoint policy reaches room 2 in 40% of episodes (random walk: 1.6%) and has bumps in 82%.
- **[Open]** Questions in stage1-plan.md section 6.

## 9. Stage 1 decisions: realistic pretraining data (2026-11-15)

- **[Kumar]** Answered Q1 to Q4 of the stage-1 plan: 1M transitions, patch 6x8, a thin loader for stage 2, and no chaser bumps in pretraining; chasers are off in pretraining so they appear later as a novel critical obstacle. Core principle: use data that would be available in real life.
- **[Kumar]** Also asked for parallel execution and delegation wherever possible, and for a `hitl-dev` skill (costly or risky commands become a step file the user runs). Built the skill; first use is still pending.
- **[Built]** The waypoint policy stopped seeking chasers. Started generating the 1M pretraining set (chasers off) and a separate chasers-on transfer set; wrote the dataset adapter; delegated the `spt` wiring prototype to a forked agent.

## 10. Consensus before data: random walks, no planning in the world model (2026-11-15)

- **[Kumar]** Stopped data generation until the data, model and training details are agreed explicitly; asked for the details and for confirmations on anything not yet agreed.
- **[Kumar]** Answered the data questions: random walks dispersed throughout the world with stopping, reversing and varied speeds; no turning patch at all; start poses uniform in both rooms; 2,048 x 500; defaults for the rest; per-channel normalisation, no augmentation; keep all frames.
- **[Kumar]** Planning must never exist in the world model: it only models the world, and planning arrives with the actor. Random walks are an explicit design decision. Logged as INT-11 and INT-22 in [intentions.md](intentions.md), which also backfills the other intentions of this session and asks me to log new ones.
- **[Built]** Removed the waypoint policy; the generator is a random walk only. Not yet implemented: stopping, reversing, speed variance and dispersed starts (waiting on the exact distribution).
- **[Open]** The model and recipe questions (M1 to M4) were declined as posed; Kumar asked to clarify them.
- A platform comparison (RunPod, Modal, Lambda, AWS, Vast.ai) came back from a research agent: ranked RunPod Secure first, Modal second, with provider-page prices; to be folded into [compute-options.md](compute-options.md) after the platform is chosen.

## 11. Confirmations (2026-11-15)

- **[Kumar]** Confirmed INT-42 (privileged information is fine for data generation and evaluation) and INT-43, extended to uniform coverage of states, affordances, actions and situations. Corrected INT-41: the question batch was fine and closing it was accidental. Logged in [intentions.md](intentions.md) section 6.

## 12. Method and latent decisions (2026-11-15)

- **[Kumar]** Encoder fixed to ViT-tiny, 12 layers (INT-50). Stage 2 baseline is LeWM's mean-pooled frame latent (INT-51). Attentive pooling is parked as a potential configurator role; a forked agent is writing `component-roles.md` (INT-52). Asked whether SIGReg with masked latent prediction is sensible (INT-53).
- Searched for prior work on SIGReg with masked latent prediction: none found. LeJEPA uses multi-view prediction and LeWM next-frame prediction; the combination is untested in the sources I found.

## 13. Objective, walk and platform decisions (2026-11-15)

- **[Kumar]** Chose SIGReg with masked latent prediction only (INT-54), jittery random walks with good representation of every case (INT-55), uniform starts then coverage measurement and training-time reweighting (INT-56). Platform requirements are broader than my comparison covered (INT-57): research launched for platforms that start simple and scale to about 40B-parameter multi-GPU training.
- A forked agent wrote `component-roles.md`; its primary source (LeCun's paper) could not be retrieved, so its claims are marked [Unverified]; a second agent is trying to verify it.

## 14. Verified roles document, platform research, clarifications (2026-11-15)

- **[Built]** `component-roles.md` now rests on the primary source: the full LeCun paper (v0.9.2) was retrieved from a Wayback copy and about 20 claims were verified; four errors in the first draft were corrected.
- **[Noticed]** LeCun's paper lets the configurator prime the world model for a goal and modulate the intrinsic cost's weights. Our INT-11 (no notion of planning in the world model) is therefore a deliberate deviation from the paper.
- A multi-GPU platform comparison (provider-page prices) ranked Lambda, Nebius and RunPod for this user, with AWS as the strongest multi-node record but the steepest learning curve.
- **[Kumar]** Asked for the precise reason, if any, that SIGReg might fail with masked prediction; asked what "jitter" meant; accepted the projector pending reasoning; asked for complete sentences; accepted the pilot-then-full recipe.

## 15. Hold length, exit rules, platform, and doubt about masked prediction (2026-11-15)

- **[Kumar]** Accepted per-episode hold lengths (INT-62), the stop-and-ask exit rule (INT-63) and Lambda as the platform (INT-64). Hesitated on masked latent prediction itself, fearing the visible tokens may not predict the masked ones (INT-65); the projector decision waits.
- **[Built]** Launched a predictability study: an inpainting network measures how well masked 6x8 patches and I-JEPA-like blocks of our frames can be predicted from the visible context, with and without the previous frame, per pixel class.

## 16. Landscape survey (2026-11-15)

- **[Kumar]** Asked for a rigorous survey of self-supervised latent-prediction schemes for the perception encoder (INT-66). Four parallel research streams reported; most claims are from the agents' memory and are marked as such. Result in [ssl-landscape.md](ssl-landscape.md).
- Headline: no published comparison of masked-then-frozen against jointly trained encoders, no vision precedent for SIGReg with masked prediction, and no test on tiny egocentric frames; the installed packages already ship I-JEPA, SALT, data2vec, VICReg, LeJEPA, LeWM and PLDM.

## 17. Predictability study result (2026-11-15)

- **[Built]** The study of Kumar's concern (INT-65) finished: with spatial context only, floor, wall and own body are well predicted (R squared 0.86 to 0.99 with half the image hidden), while small food discs are predicted worse than a constant colour; the previous frame raises overall block-mask R squared from 0.72 to 0.90. Details in [predictability-study.md](predictability-study.md). The per-class numbers use the overall variance as the denominator.

## 18. Pivot to joint LeWM-style training (2026-11-16)

- **[Kumar]** After the landscape survey and the predictability study, no pretraining scheme matched exactly what is wanted. Pivot (INT-70): train the encoder and the predictor jointly from scratch, LeWM style. The staged strategy B, the masked-prediction objective and the planned pilots are superseded; the data, preprocessing, platform, exit-gate and no-planning decisions stand.
- **[Built]** Groundwork: `stable-worldmodel`'s LeWM builds and trains on our 48x64 frames after adding `imageio` (a missing dependency of that package). The HF ViT needs a square native position grid, so the encoder is built at 64x64 and our 48x64 frames use interpolated position embeddings; 6x8 patches (chosen for I-JEPA) are not usable. Measured locally (RTX 3050, 4 GB, bf16, random data, batch 128): patch 8 (48 tokens) 346 ms per step, 370 samples per second, 2.9 GB peak; patch 4 (192 tokens) does not fit at batch 64 or more (5 to 10 GB, 6 to 16 samples per second). Total parameters 17.9M (encoder 5.39M, predictor 10.79M).

## 19. Pivot decisions and SIGReg weight investigation (2026-11-16)

- **[Kumar]** Patch 8 by 8, the CLS token with the projector, and frameskip 1 (INT-71 to INT-73); asked for a rigorous look at how the SIGReg weight is chosen (INT-74).
- **[Built]** [sigreg-weight.md](sigreg-weight.md): the weight is the only knob that matters, moderately sensitive, environment-dependent (FourRoom 0.72 vs 0.09 elsewhere) and coupled to any inverse-dynamics weight; no schedule found. A wide logarithmic sweep judged without planning scores is the practical procedure.

## 20. Random-walk policy implemented; transition GIFs by gap (2026-11-16)

- **[Built]** `RandomWalkPolicy` implements the agreed walk (INT-62): per-episode speed scale in [0.3, 1.0], per-episode resample probability log-uniform in [0.04, 1.0], forward uniform in [-0.5, 1.0] x scale with 15% pauses, turn uniform in [-1, 1]. Dispersed starts are still not implemented.
- **[Kumar]** Asked for GIFs of the transitions between frames at gaps 1, 3 and 5 (and why not 2 and 4). Made GIFs for every gap 1 to 5 (`docs/media/gap_1.gif` to `gap_5.gif`, plus `gap_strip.png`). Measured on a demo walk: the mean absolute pixel change per frame grows from 0.030 at gap 1 to 0.077 at gap 5; copy-the-last-frame error over variance grows from 0.17 to 0.35 at gap 1 to 0.39 to 0.89 at gap 5 depending on the hold level. The choice of 1, 3 and 5 was an arbitrary spread of examples, not a principled one.

## 21. Frameskip 2 (2026-11-16)

- **[Kumar]** Asked whether the number of actions and frames per sample can be dynamic (INT-75), and why I had suggested gaps 1, 3 and 5 (no principled reason). Chose a fixed frameskip of 2 to start low (INT-76); dynamic frameskip stays a later experiment.

## 22. Recipe: use LeWM's (2026-11-16)

- **[Kumar]** The SIGReg weight and the training length follow LeWM (INT-77); no sweep. Added a consolidated consensus snapshot of the first run to [stage1-spec.md](stage1-spec.md) section 10, marking what is agreed and what is still a proposal of mine.

## 23. Remaining questions, first batch (2026-11-16)

- **[Kumar]** Surfaced all remaining concerns, blockers and decisions (nine concerns, three blockers, eight decisions) with defaults. Answered the first four: local generation then compressed upload, any single GPU with comfortable capacity, tarball code transfer, and a short local run plus fail-fast tests for the cloud (INT-78 to INT-81).

## 24. Remaining questions, second batch (2026-11-16)

- **[Kumar]** Confirmed the importance-weighting approach and asked that every simulation-only privilege be documented precisely (INT-82, [simulation-privileges.md](simulation-privileges.md)); confirmed the early-exit and gate thresholds, the two transfer sets and the small defaults (INT-83 to INT-85). With this, everything needed for the first run is agreed; the data format contract is in [stage1-spec.md](stage1-spec.md) section 11.

## 25. Monitoring and checkpointing (2026-11-16)

- **[Kumar]** Wants granular monitoring of the cloud run and checkpointing that leaves results even after failures (INT-86). Checked the installed packages: Lightning supplies loggers, checkpoint, on-exception checkpoint, timer, throughput, device-stats and spike-detection callbacks; `stable-pretraining` supplies a hardware monitor, rank monitor, online probes and an environment dump. Only a dashboard logger (TensorBoard or Trackio) is not installed; wandb is.
- Forked agents started building the generator changes and the window loader with weights while the training script is written.

## 26. Training pipeline built and tested locally (2026-11-16)

- **[Kumar]** Asked for granular monitoring and checkpointing that never leaves empty hands (INT-86); chose TensorBoard (INT-87) and a Lambda persistent filesystem with a pull step (INT-88).
- **[Built]** With two forked agents (generator, loader and weights) and my own work (model, guard, training script), the whole pipeline exists: see [stage1-spec.md](stage1-spec.md) section 12. Tested: 19 unit tests, a 60-step local smoke run, an injected failure at step 25 (left `on_exception.ckpt`, `final.ckpt` and a failed-status summary), and a resume that completed.
- **[Built]** Found and fixed on the way: the hardware monitor silently needs `psutil` and `pynvml`; an eval-script bug; a wrong attribute in the preflight, caught by the preflight itself.
- **[Open]** Not yet done: generating the real data, the transfer sets, the Lambda step file, a first real run.

## 27. Lambda step file, and moving HITL files out of the repository (2026-11-16)

- **[Built]** A forked agent drafted the Lambda step file (25 steps, 12 with report-back) after reading Lambda's official docs: no documented spending limit (caps enforced by the file), per-minute billing until termination, filesystems at `/lambda/nfs/<name>`, `shutdown` keeps billing. I reviewed it, fixed two mismatches with my scripts (probe file locations for the gate, a `--stats-from` option for transfer sets).
- **[Kumar]** HITL files go in the gitignored `scratch/hitl/`, and nothing is pushed without confirmation (INT-89, INT-90). Removed the commit that added them from local history, moved the files, updated their paths, and fixed a bug found while testing: PowerShell resolved `tar` to Git's GNU tar, which misreads `C:` paths, so the scripts now call Windows' `tar.exe`.

## 28. The pretraining data exists (2026-11-16)

- **[Built]** Generated and validated the data locally: four pretraining shards (512 episodes x 500 steps each, chasers and patch off, seeds 100 to 103; 2.3 GB each, 1,024,000 transitions, 912,780 training windows and 100,980 validation windows at frameskip 2) and two transfer sets of 128 episodes (`transfer_chasers`, chasers on and patch off, 0.45% of steps are bumps; `transfer_patch_new`, patch on and chasers off). Each shard's `config.json` records the world, the policy and the code revisions.
- **Coverage (measured):** position cells are close to uniform (0.5 to 1.1% each against 0.78% uniform); heading bins are uniform (12.4 to 12.7%); actions: 28.3% reverse, 15.1% pause, 56.6% forward, turns 49.6% left and 50.4% right. Situation classes before weighting: open floor 48.2%, near wall 18.4%, near obstacle or block 10.9%, near food 14.7%, corner 5.8%, doorway 2.0%. The importance weights range 0.68 to 2.26 with an effective sample size of 92% and lift the doorway share to 4.5%.
- **Concern:** random walks rarely cross the doorway (3% of episodes cross it in 500 steps), so the model sees few room-to-room transitions; this follows from the random-walk decision (INT-22) and matters later for the patch.
- **Local run is not practical:** the preflight on the real data measured 367 samples per second and estimated 414 minutes for 10 epochs on the 3050; the full run belongs on the cloud.

## 29. First cloud session: setup passed, the throughput test said NO-GO (2026-11-17)

- **[Done]** Lambda account, card, key, filesystem and one A100 40 GB SXM4 instance (Virginia, Lambda Stack 24.04, $1.99 per hour). Code and 0.42 GB of data uploaded in about a minute each; all 19 data files verified; 21 tests pass on Linux.
- **[Failures fixed on the way]** the runner added a carriage return that tripped the cloud scripts; uv picked Python 3.15 (pinned to 3.13); torchvision came from PyPI and clashed with the cu128 torch (now taken from the cu128 index, commit 1d52618); the checksum verifier lived in the ignored scratch folder (now inlined); an ssh call waited on the keyboard in the upload script (-n and timeouts added).
- **[Measured]** 300 real steps at batch 128: 10.4 steps per second, 0.12 s per step with validation, 143 minutes projected for 10 epochs, $6.45 projected session cost; GPU memory used 2.9 of 40 GB; loader workers 8 of 30 CPUs (suspected bottleneck, unproven).
- **[Decision]** the throughput gate said NO-GO against the 90-minute cap. Kumar first chose to terminate (done, about $1 spent), then decided to run the full job and wait up to about 170 minutes (INT-93). `train.py` gained `--time-cap-minutes`.

## 30. run1 stopped by the collapse guard at step 4000; the validation batches were the cause (2026-11-17)

- **[Measured]** run1 at 10 steps per second; training SIGReg 1.9, prediction loss 0.06, copy ratio 0.52, action sensitivity 3.5 (learning), but validation SIGReg 35 to 43 and effective rank 16.7 to 16.9, flat from step 1000; the guard stopped the run at step 4000 as designed. GPU utilisation 42 to 54 percent.
- **[Cause, mechanism verified locally]** the validation loader was unshuffled, so each validation batch held 128 consecutive windows of a single episode (measured: 1 episode per batch; after the fix about 47), and the batch-level SIGReg and rank measured one trajectory, not the distribution. Fixed in commit 19e315e (a fixed random validation subset); `--seed` added.
- **[Open]** a second hypothesis (batch normalisation in evaluation mode) is tested by step D01, which also runs the probes on the step-4000 checkpoint.
- **[Decision, INT-95]** two concurrent jobs: run1 resumed from its checkpoint with the fixed validation, and run2 fresh with seed 43.

## 31. The floor is a compass: a signal the real world does not expose (2026-11-18)

- **[Finding]** Probing what the heading readout relies on showed that the checkerboard floor of agent_world gives every frame the agent's heading modulo 90 degrees (the tile edges run along the world axes) and its position modulo the checker period. Raw pixels read cos/sin of 4 x heading at R2 0.90 and the encoder's patch tokens at 0.88; absolute heading stays at about 0.
- **[Kumar]** World realism principle (INT-99): the world exposes signals the way the real world does, with less complexity; localisation is done without a compass, from memory, co-existing static objects and self-motion. A world-axis lattice on the floor is a simulation shortcut.
- **[Kumar]** Decisions: walls and box obstacles stay as they are; a fixed-direction floor current becomes a future mode; actuation noise is future scope; floor reseeding becomes a world parameter; the checkerboard stays as an option, and nothing in hjepa_agent is discarded, so models trained with and without it can be compared later.
- **[Built]** A handoff proposal for the agent_world session (kept outside git): `scratch/handoff/agent_world-signal-shortcuts.md`.
- **[Agreed]** hjepa_agent evaluation moves to egocentric quantities and pose relative to the episode start (INT-100). The stage-1 gate decision waits until agent_world settles the floor.

## 32. Ready to retrain on the map floor (2026-11-18)

- **[Done, in agent_world]** The checkerboard was replaced by a map floor: one precomputed 62 by 62 map, with a random window and rotation per episode (`floor="map"`, `floor_map_seed`). The checkerboard is still available as `floor="checker"`. agent_world now has a blank default `main`; its code lives on `develop`.
- **[Built]** The world dependency follows agent_world's `develop`. Data generation gained `--floor` and `--floor-seed`. New shard names keep the two floors apart (PR #3, stacked on PR #2; 24 tests pass).
- **[Built, measured]** New data was generated locally on the laptop GPU in 29 minutes, at about 1,000 world-steps per second (4.3 to 4.6 minutes per pretraining shard):
  - four pretraining shards (`map_pretrain_200` to `203`), each 512 episodes of 500 steps;
  - three transfer sets of 128 episodes each. One of them, `map_transfer_floor1`, uses a map never seen in training, to test whether the model memorises the floor map.
- **[Measured]** The map frames compress to about 6.6% of their size with gzip (the checkerboard's compressed to 3.7%), so the upload is about 0.75 GB.
- **[Measured]** A local smoke run (300 steps, batch 32, the four map shards, weighted sampling) completed. Prediction loss fell from 0.23 to 0.049 and SIGReg from 6.5 to 2.3, and checkpoints were saved.
- **[Built]** The cloud helper scripts take the shard and run names from the environment. The defaults are the map shards and the runs `map1` and `map2`.
- **[Synced]** The agent_world snapshot in `docs/agent_world` was refreshed to its commit 0dd10c9.
- **[Kumar]** Journals may be pushed after a safety check, without his review (INT-101).
- **[Expected, unmeasured]** The cos/sin(4 × heading) probe should fall towards the level of absolute heading on the map floor. That fall is the evidence that the shortcut is gone.
