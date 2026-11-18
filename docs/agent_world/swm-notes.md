# Notes on stable-worldmodel

A study of stable-worldmodel (swm), a library for world-model research, and how its
worlds compare with ours. See the [guide](README.md) for what the labels mean. Written
2026-11-15.

## How this was studied

- **Sources:** the repository (galilai-group/stable-worldmodel, commit 21446f1), cloned
  into `scratch/` and **read, never run**, and the paper (arXiv 2605.21800), of which
  all but the last 4,772 characters were read.
- **Method:** three read-only study agents (environments; infrastructure; planning and
  models), each asked to be critical. Their file notes are in `scratch/swm_notes/` and
  are not in the repository.
- **Reading the labels:** statements about swm were read in its code and not run, so
  runtime behaviour is **[Unverified]**. Ideas for us are **[Proposed]**.
- **VMAS:** nothing in the repository or the paper mentions VMAS. The only vectorized
  simulator the paper names is EnvPool.

## What swm is

- A toolbox with three parts: a data layer (Lance, HDF5, video), implementations of
  world-model baselines and planning solvers, and a suite of environments with
  controllable "factors of variation".
- About 150 environments, mostly wrappers of existing suites: DeepMind Control, classic
  control, OGBench, Fetch, Craftax, Atari, plus Two-Room, PushT, Piecewise and a point
  maze written for the library.
- Evaluation is goal-conditioned model-predictive control: the planner searches for
  actions whose predicted latent ends near the goal image's latent.

## Environments

| Environment | Observation | Action | Simulator | Goal | Hierarchy | Factors |
| --- | --- | --- | --- | --- | --- | --- |
| Two-Room | 10-d state (agent, target, 3 door centres); pixels added by wrappers | 2-d velocity | torch, 2D, no physics | target position or image | one door waypoint | 18 |
| Piecewise | 4-d state | 2-d | torch, 2D | position | none | 13 + 2 per zone |
| Point maze | 2-d + matplotlib render | 2-d | numpy | image | local detours | 12 |
| PushT | 7-d state + 224 px | 2-d relative | pymunk | block pose | implicit | 17 |
| OGBench cube, scene | state or 224 px | 5-d | MuJoCo, 3D | object poses | multi-step | 10, 11 |
| Others (DM Control, Fetch, Craftax, Atari, rocket, robocasa) | mostly proprioception or pixels | joints or discrete | MuJoCo, JAX, ALE, PyBullet | mostly none | at most a tech tree | 0 to many |

### What the core environments really are

- The library's own environments are 2D, single-agent, holonomic point agents, with no
  other acting entities and success defined as getting close to a target.
- **Two-Room:** Gaussian-blob rendering and no physics. Reward is always 0; success is
  distance below 16 (pixels). The code to force the target into the other room is
  commented out or unused, so same-room, trivial tasks occur. The scripted expert reads
  the door positions from the environment's internals and goes to the nearest door.
- **Piecewise:** a single open room divided into zones; position moves by
  `pos + action * speed + bias[zone]`, with a bias up to 4 against a default speed of 5,
  applied even with zero action. The bias is not in the observation. The expert inverts
  the dynamics exactly.
- **Point maze:** random floating blocks, not corridors; walls are not checked for
  reachability; the success check and the goal image can disagree.
- **PushT:** the one real physics environment (pymunk). Its "expert" is a random policy.
- **Horizon** is not in the environments: `World(max_episode_steps=100)`, and
  environments never truncate.
- **Observability is a toggle:** entries such as `render_target`, `show_goal` and
  `render_zones` in the variation space switch the goal or the zones in or out of view.

## Infrastructure

- **Variation spaces (`spaces.py`):** a typed tree of factors, each with a default, a
  current value, a sampling order and an optional constraint. Reset resamples only the
  listed factors, applies explicit overrides, then checks constraints. This is the idea
  most worth borrowing. Constraints only check the finished sample, and PushT's overlap
  constraint is a TODO.
- **`World` and EnvPool:** not real batching. EnvPool is a Python list of separate
  Gymnasium environments, stepped in a loop, with rendering and resizing on the CPU. The
  docs say "in parallel", which overstates it. The throughput table in the README
  measures dataset reading (Lance about 4.8k samples/s, HDF5 about 1.4k), not
  simulation; no simulation speed is reported anywhere.
- **To plug in an environment:** a Gymnasium environment whose `render()` returns RGB,
  with state and goal in `info`. A VMAS environment could only plug in through an adapter
  that would lose its GPU batching, unless a batched pool with EnvPool's interface were
  written (about 200 lines, estimated, **[Unverified]**).
- **Counterfactuals:** only at reset (set a state and factor values, then roll out).
  There is no snapshot and restore, no mid-episode intervention, no paired-rollout
  utility and no counterfactual metric.
- **Splits:** no train versus held-out factor values. The episode split is a random
  90/10, and the README's "zero-shot generalization" only means a factor can be set at
  each reset.
- **Reproducibility:** only watched factors are saved, and with `seed=None` the
  variation sampling is nondeterministic.

## Planning and evaluation

- **Protocol:** sample 50 (episode, start) pairs from a dataset, reset the environment
  to that state, and give as goal the dataset frame 25 steps later. Success is the
  environment's `terminated` flag at any step, not the final state. The result is a
  percentage over 50 episodes, roughly plus or minus 6%.
- **Planning:** the horizon is 5 action blocks of 5 steps, which is 25 steps, exactly the
  goal distance, so the goal always fits in one horizon. The cost is the distance between
  the last predicted latent and the goal latent. Solvers: CEM, iCEM, MPPI, Predictive
  Sampling, gradient descent and variants. No hierarchy anywhere: no subgoals, options or
  skills.
- **Baselines:** LeWM and PLDM are pixel-only (224 by 224, three frames of history).
  DINO-WM adds proprioception. The no-planning baseline (GCBC) scores 100% on Two-Room.
- **Critique:** the full goal image is given and always reachable, so greedy descent of
  latent distance is enough. The evaluation tests little multi-step or non-greedy
  reasoning, abstraction or counterfactuals. One seed, 50 episodes, any-time success.
  Evaluation may reuse the training dataset's name; which episodes are held out was not
  checked.

## Compared with our world

| Aspect | swm's worlds | Our world |
| --- | --- | --- |
| What it is | A suite of mostly existing environments plus a few of its own | One world with scenarios added to it |
| Agents and entities | One acting agent, no other actors | Several agents, scripted chasers, a ghost spectator |
| Observability | Mostly full; partial only through toggles and Craftax | Partial by design: a first-person camera with occlusion, and chasers that see only in range, angle and line of sight |
| Dynamics beyond the agent | Zone biases (Piecewise), rigid-body pushing (PushT, OGBench) | A rotating patch that cannot be steered against, solid obstacles and blocks, moving chasers, consumables |
| Hierarchy and planning depth | At most one waypoint; short horizons | Rooms, a doorway puzzle, hazards: designed, but none of it measured against a learner yet |
| Task and reward | Goal image or position, defined success | None yet: reward is zero |
| Throughput | Unreported; a Python loop of single environments | Batched on the GPU, about 350 world-steps/s at 256 worlds |
| Extensibility | Write a new Gymnasium environment | Add tensor logic and entities; the patch, chasers and food each took one effort |
| Ground truth | Varies; sometimes internal state leaks into the observation | Always available, and not in the observation |
| Factors of variation | A typed variation space, in nearly every environment | Config fields only; no named, sampled, constrained factors |
| Train, test, held-out splits | None | None |
| Baselines, datasets, evaluation | Yes: planners, models, a data layer, success rates | None |
| Interface | Gymnasium | VMAS (not Gymnasium) |
| Counterfactuals | Only at reset | Not built, but the state is small and fully known |

**Where we are weaker, plainly:** nothing learns here yet, so nothing in our world has
been shown to test anything. We have no goal, reward, dataset, baselines or evaluation,
no factor specification, and no held-out splits. Our scenarios are few and their
difficulty is unmeasured.

**Where swm is weaker:** its worlds are simple and 2D with a single holonomic agent, its
success criteria are easy, its planning evaluation is largely leaked, and nothing is
hierarchical or batched. It is a good standard for comparing world models, and a weak
place to find out whether a model reasons.

## Ideas to borrow

- **[Proposed]** A named variation space: factors with defaults, sampling and
  constraints, recorded in every episode, plus a split spec for held-out values.
- **[Proposed]** Snapshot and restore, so a counterfactual twin can be made: same state,
  one factor changed (a block moved, the patch's spin flipped), the same actions,
  compared.
- **[Proposed]** Evaluate against more than a model: a random policy, a no-planning goal
  policy and a greedy-latent planner, and report final success as well as any-time.
- **[Proposed]** Scenarios that trap a greedy planner (a detour that must first move away
  from the goal), and a sweep of how far away the goal is against the planning horizon.
- **[Proposed]** Probes of a model's latent for concepts such as "is lit" or "has line
  of sight", for the mirror puzzle.
- **[Proposed]** Scripted experts as demonstrators: swm's Two-Room expert is the pattern
  of the demonstrations strategy in [curriculum.md](curriculum.md).
- **[Proposed]** A batched pool with EnvPool's interface over a VMAS environment, so our
  worlds could run under swm's training and evaluation without losing the batching.

## Open questions

- **[Open]** Which of these is worth building, and in what order.
- **[Open]** Whether to expose our world through Gymnasium or a batched adapter at all.
- **[Open]** Details not verified: the data format internals, per-environment factor
  counts for several families, and any runtime behaviour.
