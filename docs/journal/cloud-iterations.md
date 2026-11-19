# Cloud iterations

One section per iteration of hypotheses, fixes and released jobs on the cloud GPU (Kumar's instruction, 2026-11-17). Each section follows the simply-explain format. Evidence marks: **[measured]**, **[verified]** (checked directly), **[assumed]**.

Machine: Lambda 1x A100 40 GB SXM4, Virginia, $1.99 per hour. Data: four pretraining shards (912,780 training windows, 100,980 validation windows) on the persistent filesystem `hjepa-fs`.

---

## Iteration 1 (2026-11-17): the collapse guard stopped run1 at step 4000

**One line:** run1 was learning, but validation measured single trajectories, so the guard read healthy training as collapse and stopped it.

### What it is (declarative)

**The issue**
- run1 stopped at step 4,000 of 71,310, by its own collapse guard **[measured]**.
- The guard stops a run after 3 consecutive validation checks fail either rule: SIGReg statistic at most 5, effective rank at least 48 of 192.
- Validation at steps 1,000, 2,000 and 3,000: SIGReg 43, 38, 35; effective rank 16.9, 16.9, 16.7 **[measured]**.
- Training at step 3,600: SIGReg 1.9, prediction loss 0.06 **[measured]**.
- Validation also showed learning: copy ratio 0.52 (below 1), action sensitivity 3.5 (above 1) **[measured]**.
- Saved: checkpoints at steps 1,999, 2,999, 3,999 and `final.ckpt`; no `last_good.ckpt`.

**The hypotheses**

| Id | Hypothesis | Predicts |
| --- | --- | --- |
| H1 | Batch norm in evaluation mode uses stale running statistics, distorting the validation latent | Validation metrics bad in evaluation mode, good with batch statistics |
| H2 | Each validation batch holds one episode, so batch statistics describe one trajectory | Metrics good on mixed batches, in any mode |
| H3 | The model is genuinely partly collapsed | Metrics bad on mixed batches too; weak probes |

**The fix (for H2)**
- Validation now uses a fixed random subset of 6,400 windows (50 batches of 128), seeded **[verified]**.
- Batches now mix about 47 episodes instead of 1 **[measured locally on `pretrain_100`]**.
- Every check sees the same windows, so checks stay comparable.
- Commit `19e315e`; `train.py` also gained `--seed`. Tests: 21 passed.

**The jobs**

| Job | What | Tests |
| --- | --- | --- |
| D01 | Diagnostic on the step-4000 checkpoint: 3 network modes × validation and training windows, mixed batches; then probes | H1 vs H2 vs H3 |
| run1 resumed | Continues from step 4,000 with the fixed validation | H2 on the real run |
| run2 | Fresh, seed 43, fixed validation, beside run1 | H2 from step 1,000; seed variance |

Two jobs because GPU compute binds: one job used 42 to 54 percent **[measured]**; memory would allow 10, CPUs 3.

### How it works (procedural)

**How the failure arose**
1. The validation loader was unshuffled, and validation is capped at 50 batches.
2. So each batch held 128 consecutive windows of one episode, from the same 13 episodes every time **[verified]**.
3. SIGReg tests whether the batch's latents look like an isotropic Gaussian.
4. One trajectory spans few directions, so SIGReg is high and the effective rank low.
5. Training batches are sampled across all episodes, so training SIGReg stayed near its floor.
6. Three failed checks after the 2,000-step grace period stopped the run.

**How the jobs decide between the hypotheses**
1. D01 measures SIGReg and rank on mixed batches in three modes.
2. Good in all modes means H2 alone; evaluation mode bad means H1 also holds; bad everywhere means H3.
3. D01's probes give position, heading and room scores against pixel and random baselines; weak probes support H3.
4. run1 resumed: its next check (step 5,000) should be healthy if H2 is right.
5. run2: healthy from its first check (step 1,000) if H2 is right; the guard stops it otherwise.

### Limits and unknowns
- H1 and H2 can both hold; D01 separates them.
- Whether the model has learned useful latents waits on the probes (D01) and the gate at the end.
- Two jobs share the GPU: each should run at 8 to 10 steps per second **[assumed]**; each job's 180-minute timer stops it cleanly if slower.
- Cost so far: about $1.30 at the start of this iteration; projected session about $6 to $7 **[assumed]**, cap $10.
- Loader speed (GPU at about half use) is unaddressed in this iteration.

### Sources (reliability tier, per the auto-research skill)
- H1: PyTorch documents that BatchNorm keeps running estimates during training and uses them in evaluation (tier 1): [BatchNorm1d docs](https://docs.pytorch.org/docs/2.3/generated/torch.nn.BatchNorm1d.html).
- H2: SIGReg tests whether a sample of embeddings, projected on random directions, matches an isotropic Gaussian (tier 2): [LeJEPA, arXiv 2511.08544](https://arxiv.org/abs/2511.08544). A sample drawn from one trajectory is not a sample of the distribution, so the test rejects it. No source was found describing this exact symptom.
- Batch order changes results whenever batch statistics are involved (tier 4, community): [PyTorch forum thread](https://discuss.pytorch.org/t/shuffle-true-or-shuffle-false-for-val-and-test-dataloaders/143720).

### Results
**D01 on the step-4000 checkpoint, mixed batches, 20 batches each [measured]:**

| Split | Network mode | SIGReg | Eff. rank (of 192) | Pred. loss |
| --- | --- | --- | --- | --- |
| val | eval | 3.11 | 55.5 | 0.058 |
| val | batch norm in batch mode | 1.96 | 56.3 | 0.058 |
| val | full train mode | 1.95 | 56.3 | 0.061 |
| train | eval | 3.08 | 56.0 | 0.057 |
| train | batch norm in batch mode | 1.93 | 56.8 | 0.058 |
| train | full train mode | 1.90 | 56.8 | 0.060 |

- **H2 confirmed as the cause:** on mixed batches every mode passes both guard rules.
- **H1 is a small real effect:** evaluation mode raises SIGReg by about 1.1 (3.1 vs 1.95), still under 5.
- **H3 rejected:** effective rank about 56, no collapse.
- Validation and training agree, so no overfitting at step 4,000.

**Probes at step 4,000 (linear ridge, 20,000 train and 5,000 val frames) [measured]:**

| Target | Checkpoint | Pixels (PCA 256) | Random init |
| --- | --- | --- | --- |
| R2 pos_x | 0.24 | 0.05 | -3.2 |
| R2 pos_y | -0.01 | -0.01 | -10.3 |
| R2 heading cos / sin | -0.01 / -0.01 | -0.02 / -0.02 | -2.4 / -7.0 |
| AUC room2 | 0.84 | 0.65 | 0.51 |
| AUC food visible | 0.83 | 0.82 | 0.78 |

- The checkpoint beats both baselines on pos_x and room, but is far below the gate (R2 at least 0.9).
- **New open question (iteration 2):** position y and absolute heading are not linearly decodable at all, from latents or pixels. Candidates: too early (step 4,000 of 71,310); absolute pose aliased in a single egocentric frame (similar walls in several directions); a linear probe too weak for cos and sin of heading. To test with later checkpoints and a small non-linear probe.

**Jobs released [measured]:** run1 resumed at step 4,000 (10.6 steps per second before run2 joined); run2 started with seed 43 (7.1 steps per second while sharing). Session cost at release: about $1.10 counted from launch record.

**Monitoring, about 12:45 UTC [measured]:**
- run1, first checks with the fixed validation (step 6,000): SIGReg 2.49, effective rank 61, copy ratio 0.28, action sensitivity 5.5. **H2 confirmed on the live run.**
- run2 at step 1,700: its step-1,000 check reads SIGReg 5.01, rank 39.9; still in the 2,000-step grace period, so not yet judged.
- Two jobs together: 7.7 + 7.3 = 15.0 steps per second, against 10.4 for one job; GPU use 78 to 92 percent.
- Projection [assumed]: run1 ends in about 2.4 hours; run2 may reach its 180-minute timer just short of the last steps, but speeds up when run1 ends.


---

## Iteration 2 (2026-11-17): why can't the probes read heading or position y?

**One line:** absolute heading is not visible in a single frame because the world looks the same every 90 degrees; heading modulo 90 degrees is.

### What it is (declarative)

**The issue:** at step 4,000 and 6,000, probes read pos_y and heading at R2 of about 0, from latents and from pixels **[measured]**.

| Id | Hypothesis | Predicts |
| --- | --- | --- |
| H4 | The information is there but not linear | An MLP probe succeeds where ridge fails |
| H5 | The world's 90-degree symmetry (square rooms, tile grid) aliases absolute heading | cos/sin of 4 x heading decodable; 1 x heading not |
| H6 | Labels misaligned with frames | Heading change does not follow the turn action |

**Fix:** none to the training. The probe targets need cos and sin of 4 x heading (identifiable); the gate's absolute-heading rule is ill-posed for this world. Reported, not changed (the gate is INT-83).

**Jobs:** local probes on run1's step-5,999 checkpoint (no cloud cost). Scripts and results: `scratch/auto-research/2026-11-17-lewm-stage1/iterations/iter-02/probes/`.

### How it works (procedural)
1. A single egocentric frame shows walls, tiles and objects, not a compass.
2. Rotating 90 degrees shows a similar view, so headings 90 degrees apart look alike.
3. Hence cos(4 x heading) is the same for all four lookalikes and can be read; cos(heading) cannot.
4. The timing check correlates the recorded turn action with the next heading change.

### Results [measured]
| Target | Latents ridge | Latents MLP | Pixels ridge | Pixels MLP |
| --- | --- | --- | --- | --- |
| cos / sin (1 x heading) | -0.01 / -0.01 | -0.06 / -0.07 | -0.02 / -0.02 | -0.11 / -0.10 |
| cos / sin (4 x heading) | 0.20 / 0.21 | 0.38 / 0.40 | 0.12 / 0.11 | 0.90 / 0.89 |
| pos_x | 0.24 | 0.30 | 0.05 | 0.08 |

- Turn action vs heading change: correlation 0.95 at lag 0. **H6 rejected.**
- **H5 confirmed.** H4 partly: an MLP helps for 4 x heading, not for absolute heading.
- At step 6,000 the latents carry less heading-mod-90 information than pixels with an MLP (0.39 vs 0.90). Whether training closes this gap is a convergence question for later checkpoints.

### Limits and unknowns
- Position y is still unexplained (likely the same mirror symmetry; untested).
- One checkpoint only; the trend over steps comes when the runs finish.

---

## Iteration 3 (2026-11-17): logging overhead, checkpoint retention, resume length

**One line:** py-spy showed per-step logging took about half of each run's main process; fixed, plus two traceability and length bugs, and both runs resumed.

### What it is (declarative)

**The issues [measured]**
- py-spy, 30 s on run1: 31 percent of the main process in per-parameter gradient norms, 15 percent in `log_dict`; the logger only writes every 10 steps.
- `save_top_k=3` deleted older step checkpoints on the cloud (run1 step 9,999 lost).
- A resumed run starts a new epoch count, so an epoch-bound run would stop about 3,100 steps short of 71,310.

**The fixes**
| Commit | Branch | Change | Training math |
| --- | --- | --- | --- |
| `1809669` | perf-log-every-n | Log metrics and gradient norm only on steps the logger writes | Unchanged |
| `6e3067e` | keep-all-checkpoints | Keep every step checkpoint | Unchanged |
| (cloud script) | none | Resumes pass `--max-steps 71310` | Unchanged; schedule length kept |

**The jobs:** run1 resumed at step 18,000; run2 resumed at step 13,000; both with the fixes. Still two jobs: GPU compute binds (88 percent busy).

### How it works (procedural)
1. Each run's main process issues every GPU operation; it was saturated at one core.
2. Removing per-step logging work frees that process to issue training operations sooner.
3. With two runs, the GPU then becomes the limit, which caps the gain.

### Results [measured]
- Speed per run: 7.3 to 8.3 (run1) and 8.0 (run2) steps per second; GPU 83 to 88 percent busy. Gain about 12 percent.
- Sources: py-spy samples the stack of a running Python process without restarting it (tier 1, [py-spy README](https://github.com/benfred/py-spy)).

### Limits and unknowns
- The progress line now prints "0.0 it/s" (cosmetic).
- Lost on the cloud before the fix: run1 step checkpoints below 15,999 except those already pulled; run2 below 10,999.
- Cost so far: about $3.50 total (instance 1 about $1.30, instance 2 about $2.20); budget $40.

---

## Iteration 4 (2026-11-17): both runs complete; where is heading information lost?

**One line:** run1 and run2 finished healthy and agree; the encoder keeps heading-modulo-90 information, and the projector discards about a third of it, a documented self-supervised effect.

### What it is (declarative)

**Run outcomes [measured]**

| Run | Seed | Steps | Final val pred loss | Val SIGReg | Val rank | Status |
| --- | --- | --- | --- | --- | --- | --- |
| run1 | 42 | 71,310 | 0.01363 | 1.49 | 92.7 | complete (111.8 min of training in 3 segments) |
| run2 | 43 | 71,310 | 0.01358 | 1.49 | 92.3 | complete |

**The issue:** at the probe plateau, latents read heading modulo 90 degrees at about 0.48 (MLP), pixels at 0.90.

| Id | Hypothesis | Predicts |
| --- | --- | --- |
| H7 | The CLS token or projector compresses it away | Patch tokens or CLS score higher than the projected latent |
| H8 | The encoder never learns it | All layers score low |
| H9 | Single-frame probes undersell a temporal model | Multi-frame probes score higher (not tested yet) |

**Sources:** Guillotine Regularization, probing a projector's input beats its output, by up to about 30 percent in self-supervised methods (tier 2: [Bordes and Balestriero, arXiv 2206.13378](https://arxiv.org/abs/2206.13378); [Xue et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/a5554e55d7a21e62d0d7a028ec0ea1c7-Abstract-Conference.html); [SSL Cookbook, arXiv 2304.12210](https://arxiv.org/abs/2304.12210)).

**Fix:** none to training. Evaluation of downstream readout moves to the backbone (CLS or patch tokens); the method is written up in `docs/journal/evaluation-methods.md`.

**Jobs:** offline layer probe on run1 final (laptop); probe trend and gate on the cloud for both runs (E04, E05).

### How it works (procedural)
1. Probe the same targets at four points: all patch tokens, mean patch token, CLS, projected latent; and pixels.
2. If only the projected latent is low, the projector loses it (H7); if all are low, the encoder never learned it (H8).

### Results [measured]
| Layer (run1 final) | cos / sin 4 x heading, MLP | abs(pos_y), MLP | pos_x, MLP |
| --- | --- | --- | --- |
| All patch tokens (PCA 256) | 0.88 / 0.87 | 0.49 | 0.29 |
| Mean patch token | 0.66 / 0.64 | 0.56 | 0.37 |
| CLS | 0.70 / 0.68 | 0.58 | 0.41 |
| Projected latent | 0.50 / 0.49 | 0.57 | 0.35 |
| Pixels (PCA 256) | 0.90 / about 0.89 | not run | not run |

- **H7 confirmed, H8 rejected.** H9 open.
**Probe trend, run1, projected latent [measured]**

| Checkpoint | cos / sin 4 x heading, MLP | abs(pos_y), MLP | pos_x, MLP | Room AUC |
| --- | --- | --- | --- | --- |
| 19,999 | 0.46 / 0.49 | 0.56 | 0.36 | 0.881 |
| 29,999 | 0.47 / 0.49 | 0.56 | 0.33 | 0.888 |
| 39,999 | 0.51 / 0.49 | 0.57 | 0.37 | 0.892 |
| 49,999 | 0.51 / 0.49 | 0.57 | 0.35 | 0.895 |
| 59,999 | 0.51 / 0.49 | 0.57 | 0.35 | 0.895 |
| 69,999 | 0.51 / 0.49 | 0.57 | 0.35 | 0.895 |
| final | 0.51 / 0.49 | 0.57 | 0.35 | 0.895 |

- **Plateau from step 40,000:** later changes are under 0.01, below the seed spread (about 0.03).
- **Original gate (INT-83) failed:** pos_x 0.33, pos_y -0.01, absolute heading about 0 (not identifiable from one frame, iteration 2), room AUC 0.895 against 0.95. Passed: completion, copy ratio 0.076, action sensitivity 21.2, rank 92.7, SIGReg 1.49, and beating both baselines on pos_x, room and food. Per INT-63, stop and ask before more compute.

### Limits and unknowns
- Whether the predictor needs the discarded heading detail is untested.
- Cost so far: about $8.00 of $40.

---

## Iteration 5 (2026-11-17): gap measurement and close

**One line:** both seeds evaluated with the new suite (PR #2); no gap blocks stage 1; the campaign closes as converged.

### What it is (declarative)
- **Question:** are there major functionality or performance gaps left (Kumar, 2026-11-17)?
- **Jobs:** E05 (run2 probe trend and gate), E06 and E07 (full evaluation of run1 and run2 final) on the cloud GPU; code `4e397b6` on branch `feature/eval-suite`, PR #2.
- **Results:** `docs/journal/world-model-gaps.md` sections 5 and 6.

### Results [measured]
- run2 probe trend plateaued like run1 (cos/sin 4 x heading MLP 0.47 / 0.49 at final; room 0.900).
- Rollout error relative to copying grows from 0.069 at step 1 to about 0.48 at step 16, same in both seeds.
- Novelty: one-step error 2.6 times higher with chasers, 1.5 times with the patch.
- Three frames plus actions do not resolve absolute heading.

## Closing (campaign 2026-11-17-lewm-stage1)

| Item | Value |
| --- | --- |
| Best result | run1 and run2 final (equivalent): val pred loss 0.0136, copy ratio 0.076, action sensitivity 21, rank 92, SIGReg 1.49; probes plateau from step 40,000; seeds agree |
| What limits it | Single-frame identifiability (world symmetry); projector compresses readout; compounding over long rollouts; rare room transitions |
| What the next stage needs | Short-term memory for localisation; a multi-step loss if planning needs long horizons; an ensemble for robust surprise; a revised gate |
| Cost | About $9.50 of $40 (instance 1 about $1.30; instance 2 about $8.20 to 16:00 UTC), plus filesystem storage |
| Time | About 4 hours of instance time on the second instance; 2 x 71,310 training steps |
| Code | Commits `19e315e`, `1809669`, `6e3067e`, `e0d79fe` (fixes, on develop); PR #1 (guard, merged); PR #2 (evaluation suite, open) |


---

# Campaign 2026-11-19-lewm-map-floor: stage 1 retrained on the map floor

## Iteration 1 (2026-11-19): two seeds on the map floor

**One line:** both seeds trained healthily on the map floor, but learned to use actions about 8,000 steps later than on the checkerboard. The heading-mod-90 shortcut is gone from the latent, and position and room information fell with it. Both pre-registered continuation rules failed, so the runs were not continued.

### What it is (declarative)
- **Setup:**
  - **Recipe:** unchanged.
  - **Data:** four map-floor shards, `map_pretrain_200` to `203`.
  - **Runs:** `map1` (seed 42) and `map2` (seed 43), together on one A100 40 GB.
  - **Code:** PR #3 (`feature/map-floor-data`, commit 371f186), with the world at agent_world `develop`, commit 860ce30.
- **Predictions, recorded before launch:**

  | Id | Prediction | Outcome |
  | --- | --- | --- |
  | P1 | The heading-mod-90 probe falls towards 0 | **Confirmed:** projected 0.055 / 0.066 (checkerboard 0.52 / 0.48); CLS 0.155 (was 0.70) |
  | P2 | Training stays healthy | **Confirmed:** SIGReg 1.48 / 1.49, rank 93.1 / 92.5, no guard stop |
  | P3 | One-step loss is higher than on the checkerboard | **Unclear:** val pred loss 0.0134 / 0.0132 against 0.0136; the latent spaces differ, so this does not compare directly |
  | P4 | The held-out floor map costs little extra error | **Confirmed:** ×1.07 / ×1.10 at h=1, so the map is not memorised |

- **A suspicion during the run:** at step 3,000 the predictor ignored actions. Copy ratio was 1.02 and action sensitivity 1.03; checkerboard run2 had 0.45 and 3.1 at the same step.

  | Id | Hypothesis | Prediction | Test | Outcome |
  | --- | --- | --- | --- | --- |
  | H1 | Late onset: textured input is harder | Action sensitivity above 1.5 by step 6k to 10k | Keep training (free) | **Supported:** 1.29 / 1.51 at step 10k, 6.7 / 7.1 at the end |
  | H2 | A data bug: actions do not drive motion | Low action-motion correlation | Correlate actions with recorded pose changes | **Rejected:** turn 0.951 (checkerboard 0.949), forward 0.919 (0.917) |
  | H3 | Unpredictable texture dominates the latent | Action sensitivity stays near 1 | Same as H1 | Rejected |

- **Continuation** (Kumar allowed it; the rule was recorded before the end):
  - **Built:** `--init-from` for a warm restart (PR #5, stacked on #3), because a plain resume would push the cosine schedule past its end.
  - **Rule (a), the late drop in val loss, fails:** steps 59k to 71k gave −2.2% on both seeds, the same as the converged checkerboard runs (−2.2% / −2.9%), and under twice the seed spread.
  - **Rule (b), the probe trend, fails:** from step 50k to final, every probe moved by less than 0.01 on both seeds; the seed spread is about 0.03.
- **Final evaluation (E02) and probe trend (E04):**

  | Measure (MLP R², or AUC) | Checkerboard run1 / run2 | Map map1 / map2 | Pixels (map) |
  | --- | --- | --- | --- |
  | cos(4 × heading), projected | 0.52 / 0.48 | 0.055 / 0.066 | 0.43 |
  | abs(pos_y), projected | 0.57 / 0.57 | 0.19 / 0.29 | 0.34 |
  | Room AUC, projected | 0.90 / 0.90 | 0.70 / 0.68 | 0.67 |
  | Room AUC, all patch tokens | 0.89 / 0.91 | 0.78 / 0.79 | 0.67 |
  | Rollout model/copy, h=1 / h=16 | 0.07 / 0.48 | 0.21 / 0.54 | n/a |
  | Novelty, chasers (error ratio, h=1) | ×2.5 / ×2.7 | ×1.9 / ×1.9 | n/a |
  | Novelty, patch (error ratio, h=1) | ×1.5 / ×1.6 | ×1.08 / ×1.05 | n/a |

### How it works (procedural)
1. **Removing the floor lattice removes two things:**
   - the heading-mod-90 cue, which the projected latent reads at R² 0.06;
   - most of the position and room information the latent had.
2. **The position probes stay flat from step 10k,** which rules out under-training as the cause.
3. **Pixels still read heading mod 90 at 0.43,** from the axis-aligned walls and boxes (kept by design). The encoder does not carry it into the latent.
4. **Actions are learned late (H1)** because motion has to be read from irregular texture rather than a regular lattice. The final action sensitivity is not comparable across latent spaces.

### Limits and unknowns
- **Why position fell is untested.** Candidates:
  - **Hp1:** the lattice's perspective foreshortening was a metric distance cue.
  - **Hp2:** latent capacity goes to texture motion.

  A discriminating test would probe distance-to-wall targets, or train on a plain (untextured) floor.
- **Copy ratio and action sensitivity** do not compare across floors.
- **Cost:** about $5.70 this session; about $15.20 of $40 for all cloud work.

### Results
- **Stopping rule holds:**
  - no open failure;
  - targets measured on held-out data against baselines;
  - probes flat from step 10k onwards;
  - seeds agree within their spread.
- **The model is converged for this recipe on this floor.** The remaining gaps (G11, G12 in `world-model-gaps.md`) are recipe and world questions for Kumar, not training failures.

## Closing (campaign 2026-11-19-lewm-map-floor)

| Item | Value |
| --- | --- |
| Best result | map1 and map2 final (equivalent): val pred loss 0.0133, copy ratio 0.22, action sensitivity 6.7 to 7.1, rank 93, SIGReg 1.48 |
| What limits it | Little position or room information in the latent (below or equal to pixels); single-frame identifiability; patch novelty hard to detect |
| Next | Test Hp1 against Hp2; decide the gate (G6); short-term memory (G4) |
| Cost | About $5.70 (2.9 hours of instance time); campaign total about $15.20 of $40 |
| Code | PR #3 (map-floor data), PR #5 (`--init-from`, not used); helper fixes in `scratch/hitl` (probe-trend pipefail) |
