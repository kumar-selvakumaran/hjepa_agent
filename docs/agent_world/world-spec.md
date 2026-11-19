# World specification

What the world is for, what it must demand of an agent, and what exists so
far. See the [guide](README.md) for what the labels mean.

## Purpose

**[Kumar]** An environment in which every property a System 2 reasoning system
needs is *demanded by the environment itself*. First define precisely what the
world should have; then build it; then build general, non-task-specific agents
that can act in it.

The world is the test bed for the research question traced in
[journey.md](journey.md): what structured, deliberate reasoning adds over a
model's instinctive reaction, and how to build it.

## What the world must demand

### Stated requirements **[Kumar]**

1. **Not solvable by a simple local planner.**
2. **Needs planning at a higher level of abstraction.**
3. **Also needs low-level planning**, to carry out subtasks.
4. **Partially observable.**
5. **Dynamic**: things change on their own.
6. **Non-trivial behaviour that has to be learned** before reasoning is possible.
7. **Basic physics, with diverse dynamics and diverse tasks.**
8. **Multi-agent**, because other agents allow a much wider range of tasks,
   dynamics and inclinations.
9. **Simple**: most of the complexity comes from these requirements, not from
   high-dimensional data or noise.
10. **Trainable at large scale.**

### Derived from the papers **[Proposed]**

From LeCun (2022):

- Horizons long enough that flat planning fails and subgoals pay off (§4.7).
- Uncertainty: several possible futures from one state (§4).
- Partial observability that needs state estimation and memory (§3).
- A mix of routine situations (reactive policy) and novel ones (deliberate
  planning) (§3.1).
- Sparse state changes: an action alters a small part of the world (§4.9).

From Goyal & Bengio (2022):

- Reusable mechanisms that apply to many different objects (§3.9).
- Test tasks that recombine known pieces in new ways (§3.1, §3.9).
- Stable rules, with one or two changing between episodes (§3.6, §3.7).
- Relevant events far apart in time (§3.10).
- Actions that control specific variables, so causal structure can be found by
  acting (§4.4).

### Checks **[Proposed]**

| Requirement | It holds if |
| --- | --- |
| Real hierarchy | A flat planner on a fixed budget fails; one given the true subgoals succeeds |
| Low-level difficulty | Scripted subgoal executors alone do not solve it |
| Partial observability | A memoryless policy fails |
| Dynamics | A plan made at the start and run blind fails |
| Learned rules | Performance depends on knowing the rule; rules can vary between worlds |
| Composition | Held-out combinations measure systematic generalisation |
| No memorising | Layouts are generated fresh each episode |

## World content

### The scenario **[Kumar]**

- The agent starts in a **safe room** with few or no adversaries, where
  **food is scarce**.
- **Doors** lead to **higher-risk, higher-reward** areas containing
  adversarial agents.
- To get there the agent must first **learn to open the door**, which requires
  a basic understanding of the world's physics.
- Whether and when to move on is **the agent's own choice**.

Related ideas this draws on (from memory, unverified): foraging under
predation risk and the "landscape of fear"; the secure base in attachment
theory; learning-progress curiosity; ability gating in games.

### Other agents **[Proposed]**

Scripted agents with fixed but hidden inclinations (for example chaser,
avoider, helper, guard), sampled per episode and to be inferred from
behaviour. Scripted rather than learning, so the world stays stationary.

### Drives and bodily signals

- **[Kumar]** Observation should not be the only input. Signals such as touch
  or pain should reach the agent through a separate route that produces an
  immediate negative signal.
- **[Proposed]** Route such signals to a hard-wired *intrinsic cost* (LeCun's
  term), and also include them in the perceived state so a world model can
  learn to predict them.
- **[Proposed]** Survival through **homeostasis** (internal variables such as
  energy and damage, with reward for moving towards their set points) plus
  **death as an absorbing state**.

- **[Kumar]** Defer the first survival state (decided 2026-11-18). Todo, with the defaults
  proposed so it can be picked up later:
  - Scope: energy (falls over time, rises on eating) and health (falls on a chaser bump,
    recovers slowly); the thorn field and curiosity later.
  - Exposure: the observation becomes a dict of the image and a body vector (energy,
    health, speed, turn rate).
  - Death: done when energy or health reaches 0, behind a config flag.
  - Reward: homeostatic drive reduction (the change in a well-being score).

### Doors **[Proposed]**

The mechanism must be learnable but not openable by accident: for example a
hinged door that needs enough force, or a lever elsewhere in the room.

## The view

### Requirements **[Kumar]**

- The world is 2D, as in VMAS, but the **eye is raised above the ground** and
  **looks down**.
- The view is bounded at the **top by the horizon**, at the **bottom by the
  ground a little behind the agent**, and by **60 degrees to the left and
  right**.
- **Third person**: the agent sees itself. The eye is **a little taller than
  the agent**, directly above it (confirmed 2026-11-14).
- Entities may have **heights** that affect what is seen, while interaction
  stays 2D.

### As built **[Built]**

| Setting | Default | Meaning |
| --- | --- | --- |
| `eye_height` | 0.6 | Eye height above the ground (agent height is 0.25) |
| `eye_back` | 0.0 | Distance of the eye behind the agent; 0 is directly above |
| `rear_distance` | 0.2 | Bottom edge meets the ground this far behind the agent |
| `half_fov_deg` | 60 | Horizontal half field of view |
| `width` x `height` | 64 x 48 | Image size in pixels |
| `max_range` | 12.0 | Beyond this, background is drawn |

- A pinhole perspective camera, so straight lines stay straight. An earlier
  equal-angle version bent straight walls and was replaced.
- Each entity is extruded to a height (circles to cylinders, boxes to prisms)
  and rays are cast against them, giving occlusion and depth.
- The observation is an RGB image per agent, `[worlds, 48, 64, 3]`, values in
  `[0, 1]`. Depth is computed but not yet part of the observation.

### Open: picture or list **[Open]**

The agent can be given either the rendered **picture** (built), or a **list**
of visible entities with their projected position, apparent size and colour.
The list carries the same information at almost no cost but removes the need
to learn to see. To be decided after measuring the picture on a GPU.

## The spectator **[Built]**

**[Kumar]** A way for a person to explore the world as an agent: same
movement, same view, but invisible to other agents, unable to touch or
interact with them, and stopped only by the edge of the defined world.

Built as an ordinary VMAS agent with collisions off, hidden from every view
but its own, and held in by VMAS's world bounds. Enabled with
`WorldConfig(spectator=True)`; off by default because each agent costs one
rendered view per step. `scripts/explore.py` drives it from the keyboard
(not yet tested with a real display).

## Dynamic elements and physics puzzles

Added 2026-11-14. Requirements stated by Kumar; how each is built is open.

- **[Kumar]** The world should contain physics puzzles whose solution is found
  in the dynamics, not in extra pixels or noise.
- **[Kumar]** Moving walls, moving obstacles and a moving floor.
- **[Kumar]** Example: a floor current runs right to left, and a curved wall on
  the left throws agents back. To cross, the agent moves diagonally against
  the flow so its net displacement is forward.
- **[Kumar]** Doors are blocks that slide along the wall and protrude past it
  on both sides. The agent pushes a protruding face to open or close the door,
  from either side.
- **[Open]** Whether walls bounce agents or only block them.
- **[Open]** The engine: these elements (curved walls, moving bodies, joints)
  are not supported well by VMAS. See the engine comparison once recorded.

## Doorway puzzles and hazard zones

Added 2026-11-14. Requirements stated by Kumar.

- **[Kumar]** Kumar does not mind having no door. Two rooms may be joined by a
  doorway with no door. The sliding door above is set aside for now.
- **[Kumar]** A rotating patch in front of the doorway turns the agent's
  heading as it crosses. To get through, the agent keeps turning
  counter-clockwise while moving forward.
- **[Kumar]** A field of thorns that is not rendered. Crossing it causes strong
  pain, which reaches the agent through the separate pain route.
- **[Kumar]** Pushing a plank ahead of the agent clears the thorns behind the
  plank for a few seconds, after which they return to that spot.
- **[Proposed]** Thorn state is a coarse grid with one regrow timer per cell
  per world. Cells under the plank are cleared; the agent's cell decides pain.
- **[Open]** Whether the patch is visible, and whether it only turns the
  heading. Its rate must stay below the agent's maximum turn rate.
- **[Open]** Pain as a per-step signal, or as lasting damage.
- **[Open]** Plank width, regrow time and field length, which set the
  difficulty.

## Decisions of 2026-11-14 (later)

- **[Kumar]** The agent body is a box for now. The U-shaped gripper is set
  aside; concave physics is not needed for the puzzles described above.
- **[Kumar]** The engine stays VMAS for now, since the puzzles are possible in
  it. A switch to MuJoCo is not planned.
- **[Kumar]** Walls may be slanted and flat, so a push produces a normal force
  that sends the agent back. Block-and-slide is acceptable; a bounce would be
  better. This replaces the curved wall above.
- **[Kumar]** The rotating patch is visible to the agent as vision.
- **[Kumar]** The thorn field is now rendered too. This replaces the earlier
  statement that it is not rendered.
- **[Kumar]** Thorn regrowth time is fixed.
- **[Kumar]** The colour-to-behaviour mapping is standard for now. It should
  vary a little eventually.
- **[Open]** How to make the agent's contact with a slanted wall bounce. VMAS
  contacts are soft penalty forces; to be probed.

## Cues and pain

Added 2026-11-14. Requirements stated by Kumar.

- **[Kumar]** Cues share a family but vary a little. For example thorns look
  like grass: taller or greener grass.
- **[Kumar]** Pain is caused by events the agent can see before the pain
  arrives. For example an adversarial agent that touches the agent causes
  severe pain.
- **[Kumar]** Adversarial agents get a distinct visual signature, so the agent
  learns to avoid them from having been hurt.
- **[Proposed]** Pain and the agent's own motion (speed, turn rate) reach it as
  a separate non-image observation. Not yet agreed.
- **[Kumar]** Pain and the agent's own motion are a separate observation from the
  image (confirmed later on 2026-11-14).
- **[Open]** How that separate observation is exposed to the agent.

## Rotating patch

Added 2026-11-14.

- **[Built]** A pink disc in each doorway, 10 agent widths (2.0) across, with a
  black ring and four black arrowheads on it pointing the way it spins.
- **[Built]** An agent on it is turned clockwise at 1 rad/s, added after its
  command is clamped. To cross straight it keeps turning counter-clockwise.
- **[Open]** Whether "10 agents to fit in it" means 10 across (as built) or
  10 by area, which would be a radius of about 0.35.
- **[Open]** How the arrowheads were meant to touch the ring. Built as
  tangential arrowheads, with the ring running through their axis.
- **[Open]** At 64x48 the ring and arrows are only a few pixels from 2.5 units
  away; the pink disc is clear.

### Stronger patch (later on 2026-11-14)

- **[Kumar]** The patch should be stronger than the agent can turn back, so an
  agent on it cannot turn anti-clockwise, but not so fast that it is
  incomprehensible. The way through is to work with the spin.
- **[Built]** The rate is now 2.5 rad/s (143 degrees per second) against the
  agent's limit of 2. Full counter-steering still turns it clockwise at 0.5.
- **[Built]** The doorway is crossed by turning left while still off the patch,
  entering at an angle, and slowing the spin on it. A search of 780
  entry and command combinations found 157 that cross through the doorway.
- **[Open]** The value 2.5 is a proposal, to be confirmed.

### Bigger arrows (later on 2026-11-14)

- **[Kumar]** The floor arrows should be twice as big. It is acceptable for them
  to extend past the circle, but they should be pushed inwards so they do not.
- **[Built]** Arrowheads are 0.6 long and 0.6 wide (were 0.3), drawn as straight
  triangles tangent to the ring, which sits at 60% of the patch radius so the
  arrowheads stay inside the pink disc.
## Chasers

- **[Built]** Scripted chasers (`chaser.py`): movable circular landmarks, not
  agents, so they cost no extra rendered view. `chasers_per_room` (default 2)
  in each room; speed 0.7 of the agent's maximum, same maximum turn rate.
- **[Built]** Vision is partial: the nearest non-ghost agent is seen only within
  5 units, within 60 degrees of the heading, and with one ray (the existing
  `intersect`) not blocked by a wall or obstacle. Chasers do not see through
  the room walls; the doorway is open. Seeing: turn toward it and drive. Not
  seeing: stop dead (no wandering, no memory).
- **[Built]** The rotating patch turns a chaser's heading like an agent's.
- **[Built]** A touch bumps the agent 0.5 units along the chaser-to-agent
  direction over 3 steps (a force added each step, like the floor current);
  that chaser then waits 1 s. `scenario._chasers.touched` ([agents, B] bool) is
  True on the step a bump starts, for a later pain system.
- **[Built]** Spawn: random in the room; room-0 chasers start at least 4 units
  from every agent, chosen from 128 candidates per chaser (if none qualifies,
  the farthest is used, which can happen with several agents).
- **[Built]** Look: red and black bands across the body, 6 in all. The ray caster
  takes an optional per-entity second colour and stripe count.
- **[Open]** Health and damage from a bump (`touched` is the hook).
- **[Open]** Chasers can push each other and obstacles; whether they should is
  undecided. Chasers cannot be seen through low obstacles (all block the ray).
## Food

Added 2026-11-14.

- **[Built]** Ochre yellow floor discs of radius 0.08, painted by the ray caster, not entities.
- **[Built]** Eaten when a non-ghost agent's centre is within agent radius plus food radius;
  the disc vanishes at once and reappears at a new random spot in the same room after 5 s,
  so density stays constant. The ghost does not eat. Reset restores all food.
- **[Built]** 24 per room, from a probe over random positions and headings: about 6.6 discs in the
  60 degree, 5 unit view wedge on average (16 gave 4.4, 32 gave 8.8), ignoring occlusion.
- **[Open]** Food is placed uniformly, so some sits under an obstacle and cannot be reached
  until it respawns.
- **[Open]** At 64x48 a disc is only a pixel or two beyond about a unit away.
- **[Open]** Eating gives nothing yet; energy is not built.

### Patch rate confirmed (later on 2026-11-14)

- **[Kumar]** The patch turn rate of 2.5 rad/s feels right.

### Bump cooldown (later on 2026-11-14)

- **[Kumar]** Successive hits by a chaser should be far enough apart for the agent
  to turn a full circle to see what hit it, then turn again and head away.
- **[Built]** The cooldown is now 7 seconds (was 1). Measured: in contact the agent
  turns at about 1.43 rad/s, not the commanded 2, so a full turn takes about 4.4 s,
  turning away up to about 2.2 s more, plus a moment to start moving.
- **[Built]** A touch now counts within 0.02 of contact, because a chaser pressing
  on an agent settles 0.004 apart; before, continued pressing never bumped again.

## North-star scenario: the mirror puzzle (aspirational)

Logged 2026-11-14. **Not planned and not to be built now.** It records the kind of
reasoning the world is ultimately for.

- **[Kumar]** A puzzle like the light-and-mirror games Kumar played growing up. Rays
  come in through a small hole. The player arranges mirrors, by moving pillars and
  rotating things, so the rays are redirected and hit the right spot. The player must
  also work out where the rays have to hit.
- **[Kumar]** It needs inference over several stages and non-linear planning: a
  constructive approach, not a greedy one.
- **[Proposed]** What it would ask of the world, for later: reflective surfaces that
  rays bounce off, objects that can be moved and rotated, a goal that is not given and
  must be inferred, and effects that only show up after several linked changes.
- **[Kumar]** The rays of light are visible to the agent, as are the pillars. Some
  pillars are clamped to the ground and can only rotate, others are free to move;
  levers orient them.
- **[Proposed]** Feasibility in VMAS (assessed from what we know, **not tested**):
  possible with VMAS plus our own code. VMAS supplies the pillars as entities (its
  `movable` and `rotatable` flags give "rotate only" and "free"), their pushing and
  the walls. We would write the light and the levers: a batched ray tracer that
  reflects off mirror segments for a few bounces; levers as logic that sets a
  pillar's angle; beams painted on the floor like the patch; a target spot that
  lights when hit.
- **[Proposed]** Risks: thin mirrors may be crossed at speed (0.1 thick held for
  walls); at 64x48 a beam is only a pixel or two wide, so beams need to be bright and
  wide (the same legibility concern as the food and arrows); a hidden target must be
  found by exploring.
- **[Open]** If and when to build it. Today's scenarios (the patch, chasers, food, blue
  blocks) are simpler steps in the same direction.
## Blue blocks

Added 2026-11-14.

- **[Built]** `blocks_per_room` (default 3, 0 = off) solid blue boxes, `block_size` 0.6, height 0.5.
  Static and solid for agents, drawn like other entities, and they block the chasers' line of sight.
- **[Built]** Placement is stratified: each room is cut into equal strips along x and one block sits at a
  uniform random spot in each, away from the walls and at least the patch radius plus the block's
  half-diagonal from the doorway. Blocks are placed first; obstacles and agents avoid them. Reset
  places them per world.
- **[Built]** Chasers are not stopped by blocks (a collision filter on the chaser landmarks). A chaser whose
  centre is within its radius of a block box vanishes and respawns in its own room, 4 or more units from
  every agent and clear of blocks and obstacles, with a random heading, zero velocity and no cooldown.
  The candidate sampling is shared with the chasers' reset (`Chasers._spawn`).
- **[Open]** Respawn is best of 128 random candidates; if none qualifies the best compromise is used.
- **[Open]** Blocks are axis-aligned and never move; a block next to a wall may leave a gap an agent cannot use.

### Chaser stands still after a bump (later on 2026-11-14)

- **[Kumar]** After a chaser bumps an agent from behind it must be static for the whole
  cooldown, then approach again. The cooldown is long enough for the agent to turn
  round, look at the chaser, turn back to face away and take a few steps. Only after
  that may the chaser start approaching. Chaser speed is 0.3 of the agent's.
- **[Built]** A chaser that has just bumped neither moves nor turns until its cooldown
  ends. Speed fraction 0.3 (was 0.7). A touch counts within 0.02 of contact, because a
  chaser pressing on an agent settles about 0.004 apart.

### Agent drag removed (later on 2026-11-14)

- **[Built]** VMAS gives every entity a default drag of 0.25, applied on every physics
  substep. We run 5 substeps, so the agent was damped five times as often as VMAS
  intends, and reached only about 72% of its commanded speed and turn rate.
- **[Kumar]** Fix it. **[Built]** The agent's own drag is 0: speed 1.0 is now 1.0 and
  the turn rate 2.0 rad/s, as the code always said. A chaser at 0.3 is now 30% of the
  agent's real speed.
- **[Built]** The bump cooldown is now 5 s (it was 1 s, then 7, then 6): two half turns at
  2 rad/s (about 1.6 s each), a moment to look and a few steps.
- **[Built]** The contact tests still pass at the agent's true speed. The patch can
  still be crossed: of 780 entry and command combinations, 186 cross through the
  doorway.
- **[Open]** The patch feels different: under full counter-steering an agent now spins
  clockwise at 0.5 rad/s, not about 1.1. Kumar judged the earlier feel right; to be
  confirmed.

## Design inclination

Added 2026-11-15.

- **[Kumar]** The world should be general but simple, and extendable: simple, yet
  flexible enough to hold a diverse set of nuances, so that many long-horizon tasks can
  be defined.
- **[Kumar]** The reason: existing games are not extendable, so their difficulty can be
  taken to extreme amounts along a single axis. This world should capture many diverse,
  difficult challenges that are difficult in different ways.
- **[Kumar]** Examples: the light-beam and mirror puzzle (the north-star scenario above),
  and multiple agents with social behaviour.
- **[Open]** Whether any existing environment is a contender in this sense. A survey
  found none that combines all of it; see [contenders.md](contenders.md).

## The floor and signal shortcuts

Added 2026-11-18, from a review of which signals the world exposes that the real world does
not (from the hjepa_agent session, confirmed by Kumar).

- **[Built]** The checkerboard floor shows, from one frame, the agent's heading modulo 90
  degrees (the angle of the tile edges) and its position modulo the tile size. A probe in
  hjepa_agent read it at R2 0.88 from patch features and 0.90 from raw pixels. **[Unverified]**
  here: those results come from another repository.
- **[Kumar]** Replace the default floor with a "map floor" that shows motion but has no
  global frame, and keep the checkerboard as an option so models can be compared with and
  without it. **[Open]** Not built yet.
- **[Kumar]** The map floor: a base of big circles, small discs and fine dots with thick
  outlines, a few large sections of overlapping rotated rectangles on top (some hollow),
  cropped per world from one precomputed map about 30 times the arena's area (62 by 62 units),
  with a fixed map per run and a random crop and rotation each episode. Prototype results:
  preferred-direction score 0.06 (checkerboard 0.85); motion as visible as the checkerboard.
- **[Kumar]** Walls and unrotated box obstacles stay as they are: axis-aligned walls are a
  local cue, and real buildings are rectilinear too.
- **[Kumar]** Future scope: a floor current with a fixed direction would act as a compass;
  offer it as a mode. A model smart enough to remember that a place can serve as a compass
  is a good thing, and changing the direction regularly could be the next challenge.
- **[Kumar]** Future scope: actuation noise, since motion now follows commands exactly and
  path integration is cleaner than in reality.
- **[Proposed]** A sky map is deferred: with this camera the top row is the horizon, so a
  sky shows in only a few pixels, and a sky fixed to world direction would be a compass.
- **[Kumar]** Defer the sky map (decided 2026-11-18). Measured over 256 random poses, only
  0.06% of pixels are sky and only 18% of views contain any, all in the top three rows. A sky
  map would cost about 1 MB and a few milliseconds per step, but it would be almost invisible
  with this camera and, fixed to world direction, would act as a compass.
- **[Kumar]** The rotating patch, lighting and obstacle colours give no global cue: keep.

## Status

| Item | Status |
| --- | --- |
| Walled room, static obstacles of varied heights | **[Built]** |
| Agents steered by forward speed and turn rate | **[Built]** |
| Third-person perspective view with occlusion | **[Built]** |
| Multiple agents per world | **[Built]** |
| Spectator | **[Built]** |
| Runs on GPU when available | **[Built]**, not yet run on a GPU |
| Benchmark script and instructions | **[Built]** |
| Tasks and reward (reward is currently zero) | Not started |
| Food, energy, pain, death | Food **[Built]**; the rest not started |
| Rooms, doors, risk gradient | Not started |
| Scripted agents with hidden inclinations | Not started |
| Procedural generation beyond random placement | Not started |
| Rules that vary between worlds | Not started |
| Difficulty knobs and ground-truth read-out | Not started |

## Performance

Measured on 2 CPU threads, 64x48 view, 11 entities, one agent: about 750
world-steps per second at 64 worlds. Rendering is nearly all of the cost
(physics about 1 ms per batched step). Not yet measured on a GPU; see
[../benchmark.md](../benchmark.md).

**Update 2026-11-14 [Built]:** measured on a GPU (RTX 3050 Laptop, 4 GB), 256
worlds, 1 agent, 64x48 view: about 3,480 world-steps per second, against about
478 on the CPU of the same machine. GPU memory (4 GB) bounds the number of
parallel worlds. See [journey.md](journey.md), section 10.

**Update 2026-11-14 [Built]:** on the same GPU, memory grows linearly at about
1.3 MiB per world per agent-view, and throughput stops rising at about 2,048
worlds (about 5,250 world-steps/s, 1 agent, 2.7 GiB). Beyond the 4 GB the run
does not fail but slows about five times. Table in
[../benchmark.md](../benchmark.md). Chunking the render was considered and set
aside for now: **[Dropped]** because the GPU saturates before memory limits it.

**Update 2026-11-14 (later):** with two rooms, 12 obstacles, chasers, food and
5 physics substeps, about 350 world-steps/s at 256 worlds; the breakdown and the
options for going faster are in [journey.md](journey.md), section 14.

## Known limits of VMAS **[Agreed]**

- 2D only.
- A fixed number of entities per world: nothing can be created or destroyed
  mid-episode (consumed food would be moved away and respawned).
- Logic must be written as batched tensor operations with masks.
