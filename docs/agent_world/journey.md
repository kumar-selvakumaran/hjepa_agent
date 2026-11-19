# Journey

The path from a research question about reasoning to building this world.
Chronological; each section says what was explored and what it left behind.
See the [guide](README.md) for the labels and for how far to trust the paper
references (most are unverified).

## The question

**[Kumar]** How can a system do human-like System 2 reasoning: slow,
deliberate, built up step by step, and checked against what it understands of
the world? The session began with GFlowNets as a candidate mechanism and ended
with building an environment to test such systems.

## 1. GFlowNets as System 2 samplers

- Bengio frames System 2 as structured, stepwise reasoning. GFlowNets build an
  object step by step and sample in proportion to a reward instead of
  maximising it, so they keep several hypotheses alive.
- Reference point: Hu et al. (2023), which fine-tunes a language model as a
  GFlowNet to sample chains of thought, treating the chain as a latent variable
  between question and answer.
- Kumar's alternatives were examined: asking the model to rationalise a given
  answer, and iterating guess-reason-revise. Both are known ideas with known
  weaknesses; they complement GFlowNet training but do not replace it.

## 2. What reasoning is

- **[Kumar]** People do not search end to end. They posit intermediate
  subgoals, check each for local consistency with their world model, and build
  up.
- **[Kumar]** Reasoning is **consistency with one's own subjective, imperfect
  world model**, not provable correctness. People reason with an imprecise
  model *and* an imprecise checker, and correct course.
- **[Agreed]** Two conditions keep this from being empty: the steps must do
  real work (changing a step changes the conclusion), and the reasoner must
  revise when it meets a contradiction.
- **[Kumar] Core thesis.** Reasoning is sampling or optimising consistency
  *over the space of logically constructible trajectories*: logic defines the
  search space, an energy ranks within it.
- **[Kumar]** Neural networks are unsound *and* incomplete approximations of
  the valid set, and this is *more* human-like than a sound, complete search.
- **[Kumar]** Changing a premise reshapes the energy landscape itself. This
  goes beyond LeCun's planner, where the world model and cost stay fixed.
- **[Agreed]** If an action can reach any latent state, the search space is a
  formality. Resolution: keep states continuous but **moves discrete**.
- Kambhampati's position was discussed. (corrected) His demand for provable
  correctness concerns *trusting* output, not what counts as reasoning.

## 3. A first agent design

- **[Proposed]** A full specification: a JEPA world model (encoder plus
  action-conditioned predictor); a fast System 1 (policy, subgoal proposer,
  reachability estimate); and a System 2 GFlowNet over partial plans whose
  actions insert subgoals or expand segments, trained with sub-trajectory
  balance, with good plans distilled back into System 1.
- **[Kumar] Blocker.** There is no data or reward that labels logical
  constructibility.
- **[Proposed]** Ways round it: put constructibility in the action space; use
  domains with exact checkers; use world-model prediction error as a soft
  label. Kumar noted that the first re-introduces a hard-coded bias. Middle
  ground: hard-code only minimal primitive moves and learn the rest.
- **[Agreed]** A GFlowNet is neutral. Inductive biases enter through the
  states and actions (what can exist), the reward (what is preferred) and the
  policy architecture (how it generalises).
- Compared with LeCun's H-JEPA: both reason in latent space with subgoals and
  a learned world model. The proposal differs in sampling rather than
  per-query optimisation, discrete moves, decomposition as an action, and
  revising premises.

## 4. Goyal & Bengio's inductive biases

- Ten biases were extracted from the paper: two systems; verbalisable
  high-level variables; semantic variables are causal; independent mechanisms;
  localised changes; stable versus unstable knowledge; a sparse factor graph;
  reusable schemas; short causal chains; top-down and bottom-up integration.
- (corrected) The paper does **not** propose GFlowNets for sampling thoughts.
  It is conceptual, with almost no formal definitions.
- **[Kumar]** Concepts are too many to list, people do not hold explicit
  causal graphs, and verbal accounts of our reasoning are lossy and often
  wrong. So variables and factors should not be explicit or predefined.
- **[Kumar]** Slots and factors must never be tied to predefined notions, and
  **being verbalisable must not be a requirement**; concepts should be free to
  be abstract.
- **[Agreed]** Knowledge stays implicit. Only a small **workspace** is
  explicit: a temporary structure built, checked and repaired during
  reasoning. A reasoning step is an edit to it.
- **[Agreed]** The problem splits into four parts: the **space** reasoning
  happens in, the **structure** imposed on it, the **operations** allowed, and
  the **coherence** measure. Keeping coherence separate lets different
  structures be compared under one measure. Written up in LaTeX for Kumar's
  notes.

## 5. Reading the paper closely

- On Kumar's challenge, Claude re-read the paper and grounded the list in it.
  Operations it describes: select by attention, bind, match and instantiate
  schemas, operate on few variables, seek coherence, broadcast, write to
  working memory, intervene, compose, reason over categories, recall,
  consolidate. Analogy and retracting a premise are **not** in the paper.
- **[Kumar]** Types and matching need not be explicit; they can be soft. The
  paper agrees ("distributed types").
- What System 2 adds over System 1, per the paper: flexible queries, a
  sequential and variable amount of computation, recombination on the fly, and
  immediate use of new knowledge.
- **[Kumar]** The paper's random variables need a defined support. A concept
  need not be a block of coordinates; it can be a latent factor spread across
  the whole activation.

## 6. Reasoning inside a language model

- **[Kumar]** Language models hold a useful world model, though inaccurate,
  incomplete and possibly inconsistent. The architecture fixes a class of
  learnable world models; training picks a useful one. The goal is to reason
  over the concepts in *that* model, the way people do.
- **[Kumar] Faithfulness concern.** Any map from the model's representation to
  one a reasoner can manipulate may *lose* what the model knows, or *add*
  knowledge of its own, so that the reasoning no longer reflects the model.
- **[Kumar]** So the reasoner should be minimal and integrated: an
  **alternative pathway inside the model**, which the model can choose instead
  of its native route. Stated formally as finding a model in the System 2
  class that is faithful to the one the language model learned.
- Related work found (mostly from search): Latent Thought Flow (abstract only),
  the coprocessor line, Coconut, adaptive and latent reasoning routers. None
  structures the latent space with these priors or tests faithfulness.
- **[Kumar]** A cost on computation could motivate the model to take a more
  efficient abstract route. **[Agreed]** as one term beside accuracy.
- **[Proposed]** A minimal experiment: k-hop function composition, training on
  few hops and testing on more, with baselines matched for size and compute.
- **[Kumar]** After asking whether human reasoning is known to follow the
  priors (it is not established), Kumar **lost confidence in all of them
  except sparsity and degree of abstraction**, and set factor graphs aside.

## 7. LeCun and Bengio side by side

- Compared from the three papers read directly. LeCun: reasoning as energy
  minimisation through a world model; abstraction from predictability over
  time; a hierarchy with no capacity limit. Bengio: reasoning as selecting and
  combining a few variables in a small workspace; abstraction from meaning and
  causal role.
- **[Kumar]** Agrees with Bengio that we know far more than we are thinking
  about at any moment: thought works on a sparse selection.
- **[Kumar]** We cannot assume knowledge is *stored* as a sparse factor graph;
  storage is messy. The structure belongs to the workspace, the form thoughts
  take when in focus.
- **[Kumar]** Reasoning works on *projections* of raw observations onto the
  world model. **[Agreed]** No paper treats that projection as part of
  reasoning, revisable on contradiction.
- Global workspace theory, and a reported result of a verbalisable workspace
  inside language models, were reviewed as background (unverified).
- Why JEPA world models are rare in games: collapse, including dropping
  whatever is hard to predict; no rewards; determinism; slow planning.

## 8. The turn to building a world

- **[Kumar]** Build an environment that *demands* the properties of a System 2
  reasoner; define its requirements precisely; then build general agents for
  it. The requirements are in [world-spec.md](world-spec.md).
- Candidates reviewed: Kinetix, VMAS, particle environments, XLand-MiniGrid,
  PHYRE, Craftax, MiniHack. **[Kumar]** chose a multi-agent world with basic
  physics, on VMAS.
- **[Kumar]** Fixed rules can still give complex behaviour, as in the Game of
  Life. **[Agreed]**, with the aim of behaviour that compresses well at a
  higher level.
- **[Kumar]** How to give an agent something like pain, and a bias towards
  survival. Discussed: valence, an immediate intrinsic cost, homeostasis,
  death as an absorbing state, and bodily signals as an input alongside vision.
- **[Kumar]** The scenario: a safe room with scarce food, doors to riskier and
  richer areas, and a door the agent must first learn to open.
- **[Kumar]** The view: a raised eye looking down, third person, with heights
  for vision over 2D physics.

## 9. First build (2026-11-14)

- **[Built]** The repository was set up as a uv project with VMAS as a
  dependency, and the world as a VMAS custom scenario: a walled room,
  obstacles of varied heights, steerable agents, and the third-person
  perspective view rendered by ray casting.
- **[Built]** A spectator for a person to explore the world as a ghost.
- **[Built]** Automatic use of a GPU when present, a benchmark script, and
  instructions for running it.
- Found on the way: rendering, not physics, is the cost. A first tuning pass
  made it about four times faster on CPU.
- Standing instructions given this day are recorded in
  [principles.md](principles.md).

## 10. First GPU run (2026-11-14)

- Found that the environment held a CPU-only build of PyTorch, so the GPU had
  never been used. Pointed torch at the CUDA index (cu128) in `pyproject.toml`.
- **[Built]** Measured on an RTX 3050 Laptop GPU (4 GB), 256 worlds, 1 agent,
  64x48 view: about 3,480 world-steps per second on the GPU, against about 478
  on the CPU of the same machine. This partly settles the GPU thread below; it
  says nothing yet about whether that is enough for large-scale training.
- **[Open]** The 4 GB of GPU memory bounds the number of parallel worlds.
  Chunking the rendering is one way round it; not yet investigated.

## 11. Which engine, and what the world is for (2026-11-14)

- **[Kumar]** Wanted flexible geometry, such as a U-shaped gripper body.
  VMAS has only box, sphere and line shapes, so a concave body was tested with
  joints. **[Built]** The joined U was unstable; see
  [physics-engines.md](physics-engines.md).
- Four engines were compared: VMAS, MuJoCo (MJWarp), Box2D and pymunk. Only
  VMAS and MuJoCo batch worlds on the GPU. Measurements are in the comparison.
- **[Built]** The MuJoCo renderer is much faster than our ray caster, but its
  physics was 8 to 15 times slower than VMAS's, and one substep tunnelled.
- **[Kumar]** The puzzles to build are tensor logic on top of convex shapes: a
  floor current, a rotating patch before a doorway, a thorn field cleared by a
  pushed plank, and a slanted wall. A door was dropped. See
  [world-spec.md](world-spec.md).
- **[Kumar]** The gripper is set aside: the agent is a box, and the engine stays
  VMAS. **[Built]** The agent is now a box. A switch to MuJoCo is not planned.
- **[Open]** Whether contacts with a slanted wall can bounce, since VMAS
  contacts are soft penalty forces.

## 12. Building the world up (2026-11-14, later)

- **[Built]** The VMAS probes showed that the earlier failures (walls and
  obstacles not holding, a plank not pushable) came from a box-shaped agent.
  The agent is a circle in the physics, drawn as a box inside it.
- **[Kumar]** Contacts: collision force 1,000 with 5 substeps. **[Built]** It
  roughly halves throughput. See [physics-engines.md](physics-engines.md).
- **[Kumar]** Two 8 by 8 rooms joined by a doorway. **[Built]**
- **[Kumar]** A rotating pink patch in the doorway, stronger than the agent can
  turn back (2.5 rad/s), with arrowheads twice as big as first built.
  **[Built]** It can be crossed by turning before the patch and slowing the spin.
- **[Kumar]** Second scenario: chasers (0.7 of the agent's speed, they see only
  within range, angle and line of sight, stop when they lose sight, and bump an
  agent they touch), consumable food drawn on the floor, blue blocks that make
  a touching chaser respawn far away, and red-and-black striped chasers.
  **[Open]** Being built in separate git worktrees.
- **[Kumar]** The explorer works with the new rooms and patch. After each
  verified change Kumar wants to be told, to try it in the explorer.
- **[Kumar]** Journey and rules: this file is updated after every effort.

## 13. Chasers and food (2026-11-14, later)

- **[Built]** Scripted chasers: two per room, red and black striped, 0.7 of the
  agent's speed. They see only within 5 units, 60 degrees and a clear line of
  sight, stop when they lose sight, are spun by the patch, and bump an agent
  they touch by 0.5 units over 3 steps with a 1 s cooldown.
- **[Built]** Food: 24 ochre discs per room painted on the floor, eaten by
  touching, respawning after 5 s. About 6.6 discs are in view on average.
- **[Built]** The patch arrowheads are twice as big, inside the disc.
  **[Kumar]** The patch turn rate of 2.5 rad/s feels right.
- **[Built]** Project paths in `config.py`, hard-linked environments, and a
  `.dockerignore`.
- **[Built]** Found and fixed a crash on reset with food switched off.
- Chasers and food were built in parallel in separate git worktrees, then merged
  one at a time. Merging them exposed a flaky test: chasers bumped the agent in
  contact tests, so those tests now switch chasers off.
- **[Built]** Throughput at 256 worlds is about 300 to 400 world-steps/s. The
  chasers cost the most (more than half), then the 5 substeps (about a half).
  Food and the patch cost almost nothing. Measured while the explorer may have
  been running.
- **[Kumar]** The scenario code is to be split into a package (layout B: VMAS
  adapters in `scenario/`, pure behaviours outside). **[Open]** Not yet done.
- **[Open]** Blue blocks that make a touching chaser respawn far away; making
  the chasers cheaper.

## 14. Throughput and training time (2026-11-14, later)

- **[Built]** Per step at 256 worlds (1 agent, default settings) after batching:
  about 564 ms, of which VMAS physics about 399, the observation render about
  130 and the chaser logic about 21. Without chasers the physics is about 140.
- **[Built]** The chaser logic is now batched across chasers (one line-of-sight
  call, no per-pair overlap checks): 84 to 21 ms, and the whole step 685 to 564
  ms. The remaining cost is VMAS checking each moving chaser against every wall
  and obstacle.
- **[Kumar]** Keep the chasers as they are for now.
- **[Proposed]** Faster options if training time becomes a problem, none built:
  - **Own chaser-versus-wall push-out** in tensor code (about 25 lines). Chasers
    stop colliding with walls and obstacles in VMAS, so VMAS skips those pairs.
    Estimated step 564 to about 300 ms (not measured). It is custom collision
    code, geometry only, for one kind of moving circle.
  - **One chaser per room:** about half the chaser cost; changes the design.
  - **Fewer physics substeps:** about twice as fast, but the contact leaks return.
  - **Fewer obstacles:** 6 instead of 12 is about 22% faster.
  - **A larger batch**, within the 4 GB limit of about 2,000 worlds.
- **Illustration, not a target:** at about 350 world-steps/s, 10 million samples
  take about 8 hours, 100 million about 3.3 days and 1 billion about 33 days.
  Before the rooms and features (about 1,400/s) they took 2 hours, 20 hours and
  8 days. **[Open]** Whether this is enough depends on a training budget that is
  not defined yet.
- **[Built]** VMAS's native renderer fails here: it needs the old pyglet, and the
  pyglet 2 override that keeps the explorer window working removed it.
  **[Kumar]** No top-down map view for now; the explorer stays as it is.

## 15. The chase demo, the cooldown and a north-star scenario (2026-11-14, later)

- **[Built]** A script, `scripts/chase_demo.py`, records a chase as the ghost spectator
  sees it, to `outputs/chase.gif`.
- **[Kumar]** The bump cooldown should give the agent time to turn a full circle to see
  what hit it, turn away and head off. **[Built]** It is now 7 seconds.
- **[Built]** Measured while building the demo: in contact an agent turns at about 1.43
  rad/s, not the commanded 2, so a full turn takes about 4.4 s. A chaser pressing on an
  agent settles about 0.004 apart, so a touch now counts within 0.02 of contact.
- **[Built]** The agent's real top speed is about 0.72 of the commanded 1.0, so a chaser
  at 0.7 keeps pace with it. **[Open]** Whether to fix this.
- **[Kumar]** A north-star scenario, logged and not to be built: a light-and-mirror
  puzzle needing multi-stage inference and non-greedy planning. See
  [world-spec.md](world-spec.md).

## 17. Learning curriculums (2026-11-14, later)

- **[Kumar]** The north-star scenario needs concepts the agent has no basis for: light,
  line of sight, reflection. A new document, [curriculum.md](curriculum.md), records
  strategies for teaching them.
- **[Kumar]** First strategy: demonstrations by scripted agents with deterministic
  policies. More are to be added. Nothing is built; the chasers are the existing
  building block.

## 18. The chaser stands still; agent drag removed (2026-11-14, later)

- **[Kumar]** Corrected my reading of the cooldown: after the bump the chaser is static
  for the cooldown, and the agent uses it to turn round, look, turn back and walk away.
  The chaser had kept pressing on the agent and carrying it along. **[Built]** Chaser
  speed is 0.3 of the agent's.
- **[Built]** Blue blocks are merged: 0.6 by 0.6 solid blocks, three per room; a chaser
  that touches one respawns far from the agents in its own room.
- **[Built]** Found why the agent reached only about 72% of its commanded speed and
  turn rate: VMAS applies a default drag of 0.25 on every physics substep and we use 5.
  **[Kumar]** Asked for the fix. The agent's drag is now 0, so 1.0 means 1.0.
- **[Built]** Contacts still hold at full speed, the patch is still crossable, and the
  chaser cooldown is 5 s. **[Open]** Whether the patch still feels right.
- **[Kumar]** Prefers GIFs made for him over running the explorer. GIFs of every
  phenomenon are to go in the README, where each is introduced.
- **[Built]** An earlier edit of these docs silently failed on a line-wrapped sentence,
  so this section also records what was missing.

## 19. GIFs of every phenomenon in the README (2026-11-14, later)

- **[Kumar]** Wanted a GIF of every phenomenon in the README, where it is introduced, and
  not to have to run the explorer. **[Built]** Seven GIFs in `docs/media`: the agent's
  view, the rooms and doorway, the patch spinning an agent, a doorway crossing, a chase
  with the bump and the cooldown, food being eaten, and a blue block making a chaser
  vanish (the intended use, with the agent luring it).
- **[Built]** The demo scripts are in `scripts/demos/` with a shared helper, and
  `scripts/chase_demo.py`. They were built in parallel in separate git worktrees.
- **[Open]** The agent is a small box, so its turning is hard to read in the third-person
  GIFs, the blue block GIF does not show where the chaser respawns, and the doorway
  crossing keeps the agent small. The patch spin is shown from the agent's own view.

## 20. Studying stable-worldmodel (2026-11-15)

- **[Kumar]** Saw stable-worldmodel, a library of world-model research environments, and
  asked for an objective comparison with our world and a critical study of the repository.
  Cloned into `scratch/` and read, not run. Notes in [swm-notes.md](swm-notes.md).
- Found: its core worlds are 2D single-agent point worlds with easy success; its
  "parallel" environments are a Python loop, not GPU batching; its evaluation hands the
  planner the goal and has no hierarchy or counterfactual test; its variation spaces
  are a good design. Nothing in it mentions VMAS.
- **[Open]** Which of the ideas to borrow, if any.

## 21. Is there a contender? (2026-11-15)

- **[Kumar]** Stated the core design inclination: a world that is general, simple and
  extendable, with diverse long-horizon tasks that are difficult in different ways, and
  asked whether any existing environment is a contender.
- **[Built]** Three research agents surveyed single-agent task generators, multi-agent and
  social worlds, and embodied and physics worlds. Their verdict, with caveats about how
  deep the research went: none combines GPU batching, cheap authoring of arbitrary
  mechanics, an occluded first-person camera, scripted adversaries and several difficulty
  axes; none offers an optics puzzle. See [contenders.md](contenders.md).
- The document also records what the surveyed worlds do better than ours, and that our
  flexibility claim is not yet tested on the hardest mechanic, optics.
- **[Open]** Whether to build a declarative authoring layer, and whether to build the
  optics mechanic soon.

## 22. The world as a survival game (2026-11-15)

- **[Kumar]** Stated that the world, as built and as intended, is a survival game, with
  intrinsic costs: hunger, pain avoidance, curiosity and discomfort with uncertainty. Asked
  how agents are typically conditioned to survive, citing Crafter.
- Logged as strategy 2 in [curriculum.md](curriculum.md), with a proposed mapping onto the
  world and the known tension between curiosity and discomfort with uncertainty.
- **[Open]** Defining health, hunger and pain in the world.

## 23. Signal shortcuts and a new floor (2026-11-18)

- A proposal from the hjepa_agent session (a world model trained on this world) found that
  the checkerboard floor leaks a compass: heading modulo 90 degrees and position modulo the
  tile size. **[Kumar]** The intention is a world that exposes signals the way the real world
  does but more simply, and with no compass. Recorded in [principles.md](principles.md).
- **[Kumar]** Chose a map floor to replace the checkerboard by default: circles and fine dots,
  rotated-rectangle sections, one precomputed 62 by 62 map cropped per world. Prototyped in
  `scratch/`: no preferred direction, motion as visible as the checkerboard, about 1 ms to seed
  1,024 worlds and about 3 ms per step for the lookup at 256 worlds. See
  [world-spec.md](world-spec.md).
- Existing skin packs were checked: none fits (photoreal textures, tile lattices, or tied to
  other simulators), so we keep our own generator.
- **[Kumar]** From now on questions are asked with the question wizard. **[Open]** Building
  the floor, and a sky.

## 24. Floor samples and the sky (2026-11-18)

- **[Kumar]** Asked to see samples of the map floor before it is built. Made six top-down
  crops from three maps and six agent views, in `scratch/` (not in the repository).
- **[Kumar]** Chose rectangle sections of at most 10% of an arena, 3 to 4 per arena, instead
  of 10% of the whole map, which filled whole episodes with one flat area.
- **[Kumar]** Deferred the sky map. **[Open]** Waiting for Kumar's look at the updated
  samples before building the floor.

## 25. The map floor, and a new workflow (2026-11-18)

- **[Kumar]** Redesigned the floor map: one single 30x map, rectangle sections placed one
  after another in the free area outside earlier bounding boxes until they cover about 45%,
  then a random window per world. Sizes stay absolute (sections 3% to 10% of an arena). Soften
  nothing; busy views are fine until ablations say otherwise.
- **[Built]** `floor.py`, `WorldConfig.floor` ("map" or "checker") and `floor_map_seed`;
  Pillow became a runtime dependency. 70 tests pass. Merged as PR 1.
- **[Kumar]** Adopted the hjepa_agent workflow: `develop` is the long-running branch, `main`
  is blank, efforts are `feature/`, `fix/` or `docs/` branches merged by Kumar through PRs,
  stacked PRs branch off the lower PR's branch, never force-push or delete branches.
- **[Built]** Reorganised the history: a blank root `main` (renamed on GitHub), a code-only
  `develop` of 104 commits with `docs/journal/` filtered out, and 31 restored effort
  branches. Old history kept as `main-legacy` on GitHub, a local tag and a bundle.
  **[Open]** My four docs commits had already reached the old `main` before this.
- **[Built]** Workflow text in `CLAUDE.md` and the Claude Code guard (36 tests); PR 2, merged.
  PR 3 regenerates the README GIFs (they grew from 1.4 MB to 4.2 MB).

## 26. Survival state deferred (2026-11-18)

- **[Kumar]** Asked what to build after the floor, I proposed the survival internal state
  (energy, health, pain) and asked four questions: scope, how the body signals reach the
  agent, death, reward. Kumar chose to document it as a todo and defer all four.
- **[Built]** Recorded in the world spec with the proposed defaults. The hooks exist:
  food eating, and `chasers.touched` for bumps.
- **[Open]** Waiting on Kumar's GitHub rulesets (step file in `scratch/hitl/`).

## 27. GitHub rulesets (2026-11-18)

- **[Kumar]** Created two active rulesets: `protect-develop-main` (restrict deletions, block
  force pushes, require a pull request with 0 approvals, merge method Merge only) and
  `no-force-no-delete` (all branches). Kumar is on both bypass lists (Repository admin, Always),
  so he can still force push; Claude's pushes share his token, so the local guard remains
  its only brake.
- **[Open]** The ruleset list said "targeting 1 branch" for `protect-develop-main` while its
  settings list `main` and `develop`; unexplained.

## Open threads

- **[Open]** Picture or entity list as the observation.
- **[Open]** GPU performance, and whether it is enough for large-scale training.
- **[Open]** Which of the remaining priors, beyond sparsity and abstraction,
  the world should test for.
- **[Open]** How the world's tasks and rewards will make hierarchy, memory and
  learned rules necessary rather than merely possible.
- **[Open]** What the first agent is: the language-model pathway of section 6,
  the JEPA-plus-GFlowNet design of section 3, or something simpler.
- **[Open]** How faithfulness to an existing world model would be measured.
