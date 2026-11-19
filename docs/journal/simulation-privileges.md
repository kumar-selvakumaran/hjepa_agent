# Simulation privileges (the "luxuries")

Documented 2026-11-16 at Kumar's request (INT-82): every capability we have only because we work inside our own simulator, stated precisely, with what it is used for and whether a real-world agent could have it. Labels: see [README.md](README.md). Rule of use (INT-42a): privileged information may generate data, select data and evaluate; it must never be an input to the model or leak a notion of goals or planning into it (INT-11). Append a row whenever a new privilege is used or considered.

Status key: **Used** (in the plan now), **Planned**, **Considered** (not adopted), **Removed**.

| ID | Privilege | Exactly how it is used here | Real-world availability | Status |
| --- | --- | --- | --- | --- |
| SP-01 | **Ground-truth agent pose** (x, y, heading, exact) | Probe labels for evaluation; coverage histograms; importance weights (INT-56) | Needs odometry or localisation, which is noisy and drifts | Used |
| SP-02 | **Teleporting the agent to any collision-free pose after reset** | Dispersed start poses uniform over both rooms (INT-23) | A robot cannot be placed anywhere at will; resets are slow and costly | Used |
| SP-03 | **Ground-truth positions of obstacles, blocks and food** (per episode, and per step for food) | Situation classes (near wall, near obstacle or block, doorway, near food, corner) for importance weights and coverage reports; recorded at generation | Requires perception and mapping; food-like small items would be estimated, not known | Planned |
| SP-04 | **Ground-truth positions of chasers, and bump events** | Probe labels (distance, bearing) in the transfer sets only | Would need detection and contact sensing | Used (transfer sets only) |
| SP-05 | **Colour-rule visibility labels** (patch, chaser and food flags from pixel colours) | Probe targets: the renderer gives each entity a unique colour, so a few pixels of a colour label presence exactly | Real scenes have no unique colour per entity, lighting changes, and occlusion is ambiguous | Used |
| SP-06 | **Toggling world features** (chasers off, patch off) | Pretraining world without the later novelties (INT-21); transfer sets with one novelty at a time | Real environments cannot be switched feature by feature | Used |
| SP-07 | **Unlimited cheap data, reproducible by seed, in parallel** (hundreds of worlds per batch) | 1M windows regenerated at will; separate seeds for shards and transfer sets | Real data is scarce, single-embodiment and not repeatable | Used |
| SP-08 | **Exact episode resets and episode-level splits by layout** | Held-out 10% of episodes have layouts unseen in training | Real layouts repeat; clean held-out environments are hard to get | Used |
| SP-09 | **Exact, noise-free action logging** (the commanded forward and turn, with a fixed 0.1 s step) | Actions fed to the predictor as given | Real actions and timing are noisy and delayed | Used |
| SP-10 | **A deterministic renderer with no sensor noise, blur or lighting change** | Frames used as they are; no augmentation (INT-25) | Real cameras are noisy and vary | Used |
| SP-11 | **Known world constants** (step length, speeds, room size, doorway width) | Reasoning about frameskip and the size of change per step; probe design | Must be estimated or calibrated | Used (design only) |
| SP-12 | **Importance reweighting computed from privileged situation classes** | The dataloader approaches uniform coverage of states, affordances and situations (INT-56) | A real agent cannot know the situation classes of its own data without estimating them | Planned |
| SP-13 | **Perfect collision-free spawn sampling** | Rejection sampling against known obstacle, block and wall geometry for start poses (SP-02) | Not applicable | Planned |
| SP-14 | **Ground-truth-labelled early exit and exit gate** | The run stops, and the gate passes or fails, on probes against privileged labels (INT-63) | A real system would need labels from elsewhere, or label-free signals such as the SIGReg statistic and rank | Planned |
| SP-15 | **State-setting counterfactuals** (same state, different actions) | Action-sensitivity check beyond the dataset's own trajectories (possible, not yet planned) | Impossible: the real world cannot be rewound | Considered |
| SP-16 | **Teleporting objects for a surprise test** | Violation-of-expectation check, deferred to after the first run | Impossible without staged physical setups | Considered (deferred) |
| SP-17 | **Goal-seeking data generator steering by ground-truth pose** | Was used for a waypoint policy | Not a real-world capability for passive data collection, and it injected a notion of goals | Removed (INT-11) |

## Which of these a deployed agent would lack, in one line

Everything above except actions as given and the rendered frames themselves: a deployed agent would see noisy frames and noisy odometry and would have no labels, no resets and no way to toggle the world. Label-free signals (SIGReg statistic, effective rank, prediction error against a copy baseline) are the ones that would remain.
