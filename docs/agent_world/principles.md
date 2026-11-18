# Design and implementation principles

How the world is designed and how the code is written. See the
[guide](README.md) for what the labels mean.

## Guiding priorities

1. **Simplicity first.** **[Kumar]** Build the simplest thing that meets the
   requirement. Do not over-engineer, and do not build something complicated on
   top of VMAS just because it was asked for.
2. **Scale and compute.** **[Kumar]** The world will be used to train agents at
   large scale, so scalability and compute cost are first-class concerns.
3. **Push back.** **[Kumar]** If a requirement goes much beyond a basic addition
   to what VMAS already supports, say so instead of building it.
4. **Built to grow.** **[Kumar]** Modular code and sound software engineering,
   so the codebase can scale, without adding structure before it is needed.

When these conflict, simplicity and compute win over features.

## World design

- **Complexity from structure, not from pixels or noise.** **[Kumar]** The
  difficulty should come from what the world demands of a reasoner, not from
  high-dimensional or noisy data.
- **Every requirement has a test.** **[Proposed]** A property the world is
  meant to demand should come with a check that fails for an agent lacking it
  (for example: a memoryless policy fails if the world is partially observable).
- **Separate knobs.** **[Proposed]** Horizon, observability, randomness and
  rule complexity should be separately adjustable, so it is clear which
  property breaks a given agent.
- **Ground truth is always available.** **[Proposed]** The true state, rules
  and subgoal structure can be read out, to check what an agent represents.
- **Simple fixed rules, compressible behaviour.** **[Agreed]** Fixed low-level
  rules can still give complex behaviour. The aim is behaviour that is simpler
  to describe at a higher level (stable entities and regularities), not chaos
  that can only be simulated step by step.
- **Scripted other agents before learning ones.** **[Proposed]** Other agents
  with fixed but hidden inclinations keep the world stationary enough to learn,
  and make "inclination" a controllable setting.
- **The agent chooses its own difficulty.** **[Kumar]** Harder regions are
  reachable only by first learning something; whether to go is up to the agent.
- **Physics stays 2D; height is visual only.** **[Agreed]** Entities have a
  height that affects what is seen and hidden, never how things collide.

## Implementation

- **Use VMAS as it is.** **[Built]** VMAS is a normal dependency, not a fork
  or a vendored copy. The world is a standard VMAS custom scenario. Prefer a
  VMAS feature over a new mechanism (the spectator is an agent with collisions
  off; room bounds are VMAS world bounds).
- **Keep VMAS at the edge.** **[Built]** Camera and ray-casting code is pure
  tensor maths with no VMAS imports. Only `scenario.py` touches VMAS. Either
  side can change without the other.
- **Everything is batched.** **[Built]** All code works on many worlds at once
  as tensors; no per-world Python loops or branches.
- **Device-agnostic.** **[Built]** No GPU-specific code paths. Every tensor is
  created on the environment's device; `make_env` picks the GPU when one is
  available. New code must follow the same rule.
- **Configuration is data.** **[Built]** Settings live in frozen dataclasses
  (`WorldConfig`, `CameraConfig`, `Palette`) with defaults; no constants buried
  in functions.
- **Costly features are off by default.** **[Built]** Anything that adds
  per-step cost and is not needed for training is a switch that defaults off
  (the spectator).
- **Measure before optimising.** **[Built]** `scripts/benchmark.py` is the
  reference measurement. Optimise only what it shows to be the bottleneck
  (currently rendering, not physics).

## Workflow

- **uv only.** **[Kumar]** Nothing is ever installed into the system Python.
  All dependencies go through uv into the project's own `.venv`.
- **Small, distinct commits.** **[Kumar]** One function per commit, with a
  crisp, simple, precise message.
- **Every commit passes the tests.** **[Built]**
- **Docs give exact commands.** **[Kumar]** Anything a person is expected to
  run is written out as a command that can be pasted.
- **`scratch/` is for throwaway work.** **[Built]** Its contents are ignored
  by git; `outputs/` likewise.
- **One branch per effort.** **[Kumar]** Each functional change or effort gets
  its own branch off `main`, with small atomic commits, and is merged into
  `main` when done. Until CI exists this is a local merge; once it exists, a
  pull request that passes CI will be required.
- **Probe before concluding.** **[Kumar]** To understand or investigate
  something, write a throwaway script `scratch/<task>.py` that writes its
  output to a text file in `scratch/`, and read that.
- **Raise gaps, indexed.** **[Kumar]** Ambiguities, uncertainties and blockers
  are raised as numbered questions (Q), concerns (C) or blockers (B), each with
  a default suggestion, before acting.
- **Docs are append-only.** **[Kumar]** Existing content is not removed. A
  change of intention is appended as a pivot, and only after it has been agreed.
- **Containerizable, not hardware-agnostic.** **[Kumar]** "Machine independent"
  means the code runs the same inside a container, with nothing tied to one
  person's machine: no personal paths, usernames or setup details in the repo.
  Depending on Linux or an NVIDIA GPU with CUDA is fine. Personal measurements
  and the research journey may be documented, as an example of a configuration.
- **Update the journey.** **[Kumar]** After every effort or decision, append a
  section to `journey.md`.
- **Say when to look.** **[Kumar]** After each verified change is merged, tell
  Kumar so he can try it in the explorer
  (`uv run python scripts/explore.py --scale 12`).
- **Ask with the question wizard.** **[Kumar]** Questions are asked with the question
  wizard (a dialog of options), the default as the first option, marked Recommended.
  Concerns and blockers stay as indexed text, each with a default.

## Signals like the real world, but simpler

- **[Kumar]** The intention, as relayed by the hjepa_agent session and confirmed by Kumar on
  2026-11-18:

  > The world should be as simple as possible, while exposing signals in the same *way* the
  > real world does, but with less complexity. For example: there are dynamic objects, as in
  > the real world; most objects are unambiguously distinguishable (eventually there will be
  > unique objects and landmarks); but textures, geometry and resolution are simple, so a
  > simpler encoder can learn the world and probes stay interpretable ("educational").
  > Location must be disambiguated the way people do it: without a compass, by tracking many
  > co-existing signals jointly, as in how we navigate and localise (memory, the co-existence
  > of signals from objects assumed static, motion estimated from one's own actions).

- **[Built]** Found by that session: the checkerboard floor shows heading modulo 90 degrees
  and position modulo the tile size from a single frame. See [world-spec.md](world-spec.md).

## Pivot 2026-11-18: pull-request workflow  **[Kumar]**
Replaces "One branch per effort ... merged into main when done. Until CI exists this is a local merge".
Effort branches (feature/, fix/, docs/) start from `develop` or stack on an open PR's branch; every change reaches
`develop` through a PR that Kumar reviews and merges; until CI exists the tests run locally. Never force-push, never
delete branches. `main` is the blank default branch. Journals stay out of git until Kumar confirms them; code reaches
GitHub through the code-only history. The repository's history was reorganised once to match (journey, 2026-11-18).
