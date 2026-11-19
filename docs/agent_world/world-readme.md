# agent_world

A small 2.5D world for studying agents that have to reason, built as a custom
scenario on top of [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator).

The physics is flat (VMAS, 2D, batched over many worlds). Vision is not: every
entity is extruded to a height and seen through a third-person perspective
camera, so agents get perspective, occlusion and their own body in view.

## Setup

Everything runs in a project-local environment managed by [uv](https://docs.astral.sh/uv/):

```bash
uv sync                               # creates .venv and installs everything
uv run pytest                         # tests
uv run python scripts/view_demo.py    # saves outputs/view.gif (what agent 0 sees)
uv run python scripts/explore.py      # walk around yourself (needs a display)
uv run python scripts/chase_demo.py   # saves outputs/chase.gif: a chaser hunting an agent, seen by the ghost
uv run python scripts/demos/patch.py  # the other demo GIFs in docs/media: also food_rooms.py, blue_block_lure.py
uv run python scripts/benchmark.py    # world-steps per second on this machine
```

### PyTorch and the GPU

`pyproject.toml` takes `torch` from PyTorch's CUDA 12.8 index
(`https://download.pytorch.org/whl/cu128`), because the default PyPI wheel on
Windows is CPU-only. This suits an NVIDIA GPU on Windows or Linux. On a machine
without one it still runs, on the CPU; on a platform that index does not cover
(for example macOS), remove the `[tool.uv.sources]` and `[[tool.uv.index]]`
entries and run `uv sync` again. Check what you got with:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### The explorer window and pyglet

VMAS pins `pyglet<=1.5.27`, which crashes when opening a window on Python 3.13
(`NameError: name 'COMPointer' is not defined`). `pyproject.toml` therefore
overrides it with `pyglet>=2` (`[tool.uv] override-dependencies`). This project
draws its own views and uses pyglet only for the explorer window. VMAS's own
`render()` has not been tried with pyglet 2.

### Credits

This project depends on [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator),
which is licensed under GPL-3.0.

## The world

Two walled 8 by 8 rooms joined by a doorway, with a few static obstacles (boxes and cylinders of
different heights) and one or more agents. Each agent is a circle in the physics and is drawn as a small box inside it (`agent_size`), so its heading is visible. Agents steer like a vehicle: the
action is `(forward speed, turn rate)`, each in `[-1, 1]`. There is no task
yet; the reward is zero.

![An agent drives from the first room through the doorway into the second](docs/media/rooms.gif)

*The two rooms and the doorway in the middle wall. The agent drives from the first room into the second; the white block at the bottom is the spectator's own body. Every GIF here is recorded from the ghost spectator's view.*

```python
import torch
from agent_world import WorldConfig, make_env

env = make_env(num_envs=32, config=WorldConfig(n_agents=2))
obs = env.reset()                     # one tensor per agent: [32, 48, 64, 3]
actions = [torch.zeros(32, 2, device=env.device) for _ in env.agents]
obs, rewards, dones, info = env.step(actions)
```

`make_env` puts everything (physics and rendering) on the GPU when there is
one, and on the CPU otherwise; pass `device="cpu"` or `device="cuda:1"` to
choose. Observations come back on `env.device`, so create actions there too.

### The rotating patch

Each doorway has a pink disc, 10 agent widths across (`patch_radius`), with a
black ring and four arrowheads pointing the way it spins. An agent on it is
turned clockwise at `patch_rate` (2.5 rad/s; the agent's own maximum is 2 rad/s),
so it cannot turn anti-clockwise on the patch: counter-steering only slows the spin.
To cross, turn before the patch and enter at an angle. Turn it off with
`WorldConfig(rotating_patches=False)`.

![The agent stands on the patch and the world spins round](docs/media/patch_spin.gif)

*The agent's own view as it drives onto the patch and stands still: the patch turns it clockwise at 2.5 rad/s, so the world sweeps round.*

![An agent crosses the doorway by turning before the patch and slowing the spin](docs/media/patch_cross.gif)

*Crossing: the agent starts off the patch, enters at an angle and keeps counter-steering, which only slows the spin, so it comes out in the second room.*

### Chasers

Each room holds `chasers_per_room` (2) red-and-black striped chasers. They are
scripted landmarks, not agents. A chaser sees the nearest agent only within 5
units, 60 degrees of its heading and a clear line of sight (walls included, the
doorway is open); it then turns toward the agent and drives at 0.3 of the agent's
speed, otherwise it stops. Touching an agent shoves it 0.5 units over 3 steps,
then the chaser stands completely still for 5 s, long enough for the agent to turn
round, look and walk away. The rotating patch turns chasers too. Ghosts are ignored.
`WorldConfig(chasers_per_room=0)` turns them off.

![A chaser bumps the agent, stands still, then follows slowly](docs/media/chase.gif)

*A chase seen by the spectator. The striped chaser bumps the agent from behind and then stands still for its cooldown while the agent turns round to look at it, turns back and walks away; the chaser follows slowly. The chasers ignore the spectator.*

### Blue blocks

Each room holds `blocks_per_room` (default 3, 0 for none) big blue blocks, 0.6 by 0.6, one at a
random spot in each third of the room. They are solid for agents and block the chasers' sight, but
chasers pass through them: a chaser touching a block vanishes and respawns in its own room, at
least 4 units from every agent.

![A chaser runs into a blue block and vanishes](docs/media/blue_block_lure.gif)

*The intended use of a blue block: a lure. The agent walks to the block's far side and stops just past its corner, hiding behind it. The chaser curves in along a clear line of sight, clips the block and vanishes, then respawns in its own room at least 4 units from the agent (lower left here). A chaser cannot see through a block, so the agent has to stand just beside the line, not exactly behind the block.*

### Food

Each room has `food_per_room` (24) ochre discs of radius `food_radius` (0.08) painted on the
floor; they are not entities, so they cost almost no throughput. An agent whose centre is within
its radius plus the food radius eats it (the ghost spectator does not). The food reappears at a
new random spot in the same room after `food_respawn_time` (5 s). About 6 are in view on average.

![An agent drives along a line of food discs and eats them](docs/media/food_eating.gif)

*Food: the agent drives along a line of discs and eats them. Each one vanishes at once and reappears elsewhere in its room after 5 s.*

### The view

The eye sits `eye_height` above the ground, above the agent and taller than
it, looking along the agent's heading and tilted down. The view is bounded by

- the **horizon** at the top,
- the ground a little **behind the agent** at the bottom (`rear_distance`),
- **60 degrees** to each side (`half_fov_deg`).

It is an ordinary pinhole camera, so straight lines stay straight. See
`CameraConfig` for the knobs, including `eye_back` to pull the eye back behind
the agent.

![What an agent sees as it drives](docs/media/view.gif)

*The agent's own 64x48 view, enlarged: the map floor (circles and rectangle sections, no preferred direction), food discs, an obstacle, big blue blocks, a wall and the pink patch.*

### The spectator

`WorldConfig(spectator=True)` adds one extra agent for a human. It moves and
sees exactly like the others, but it is a ghost: it passes through agents,
obstacles and walls, cannot push anything, and nobody else can see it. Only
the room's outer bounds stop it. It is always the last agent in `env.agents`.

Leave it off for training: each agent costs one rendered view per step.

## Layout

```
src/agent_world/
  chaser.py     scripted chasers
  camera.py     pose -> viewing rays            (pure tensor maths)
  raycast.py    rays + scene -> colour, depth   (pure tensor maths)
  scenario.py   the VMAS scenario: entities, physics, observation
scripts/        view_demo.py, explore.py
tests/
```

`camera.py` and `raycast.py` know nothing about VMAS, so the view can be
changed or replaced without touching the world, and the other way round.

## Performance

Rendering dominates: physics is about 1 ms per batched step, the view is
`pixels x entities` ray tests per world. On 2 CPU threads, a 64x48 view with
11 entities runs at roughly 750 world-steps per second. It is all batched
tensor code, so it should run much faster on a GPU; measure it on your machine
with `scripts/benchmark.py`.
