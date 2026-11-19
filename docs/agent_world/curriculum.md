# Learning curriculums

Ways to condition, train or otherwise enable learning in the agent. See the
[guide](README.md) for what the labels mean. Started 2026-11-14.

## Why this exists

- **[Kumar]** The north-star scenario (the mirror puzzle in
  [world-spec.md](world-spec.md)) needs concepts the agent has no basis for yet:
  light, line of sight, reflection.
- **[Kumar]** The question is how to expose a curriculum to the agent so that it can
  learn them, and which kinds of signal can be used to train it.
- **[Open]** No agent or learning method exists yet. This document records strategies,
  not a plan.

## Strategies

### 1. Demonstrations by scripted agents

- **[Kumar]** Pre-programmed agents with hard-coded, deterministic policies solve
  problems in front of the learning agent, for example unblocking things, so that the
  learner comes to understand how the world works.
- **[Built]** The building block exists: the chasers are scripted agents with
  deterministic policies (`chaser.py`), added to the world as landmarks, not agents.
- **[Proposed]** What it would need:
  - The demonstrator's actions and their effects must be visible in the learner's
    view, for example a lever moving, a pillar turning and a beam shifting.
  - Demonstrations that are reproducible, because the policies are deterministic, and
    varied in their start states.
- **[Proposed]** Risks:
  - Watching shows what happens, not necessarily why.
  - Small effects are hard to see at 64x48 (the legibility concern).
- **[Open]** What the learner is trained on while it watches. Not defined.
- **[Proposed]** stable-worldmodel's scripted Two-Room expert (it goes to the nearest
  door that fits, then to the target) is the same pattern; see [swm-notes.md](swm-notes.md).

### 2. Survival drives and intrinsic costs

- **[Kumar]** The world, as built and as intended, is a survival game: survive as long
  as possible.
- **[Kumar]** The agent is meant to carry intrinsic costs while it acts: hunger (to
  survive) and avoidance of pain, but also curiosity and discomfort with uncertainty.
- **[Kumar]** The open question is how agents are typically conditioned to survive in such
  a world. Crafter was the example (tools such as a pickaxe and an axe to make, and
  zombies to kill).
- **[Proposed]** What is typical, from memory and **[Unverified]**: a sparse reward for
  unlocking achievements plus a small health term (Crafter); homeostatic reinforcement
  learning, where reward is the reduction of the distance between internal variables and
  their set points; intrinsic motivation such as prediction error or novelty; death as a
  terminal state; goals suggested by a language model.
- **[Proposed]** Mapping to this world: food to hunger; a chaser's bump, and the thorn
  field once built, to pain; hidden hazards and unvisited rooms to uncertainty and
  curiosity.
- **[Proposed]** A known tension to design for: curiosity seeks uncertainty out, while
  discomfort with uncertainty avoids it. Hunger and pain are what give the agent a reason
  to leave a safe, predictable corner.
- **[Open]** Health, hunger and pain are not yet defined in the world (see the pain
  questions in [world-spec.md](world-spec.md)).

### More strategies

- **[Open]** To be added as Kumar describes them. Kumar has said he has ideas for
  other kinds of signal.
