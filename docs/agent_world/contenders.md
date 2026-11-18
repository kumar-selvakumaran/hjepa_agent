# Is there a contender?

Whether any existing environment matches what this world is for. See the
[guide](README.md) for what the labels mean. Written 2026-11-15.

## The question

- **[Kumar]** The design inclination is a world that is general but simple, and
  extendable: flexible enough to define many long-horizon tasks that are difficult in
  different ways. Existing games are not extendable, so their difficulty gets pushed to
  extreme amounts along one axis. Examples: the light-beam and mirror puzzle, and
  multiple agents with social behaviour. See [world-spec.md](world-spec.md).
- **[Kumar]** Is any existing environment a contender in that sense?

## How this was checked, and how far to trust it

- Three research agents surveyed different families in parallel: single-agent task
  generators, multi-agent and social worlds, and embodied and physics worlds. Their notes
  are in `scratch/contenders/` and are not in the repository.
- The quality of the research varied. The embodied survey ran about nine searches and
  fetched no pages. The single-agent survey could not see last-commit dates on any
  GitHub page, and took ProcGen, TextWorld, Crafter, Boxoban and XLand's details from
  prior knowledge. The multi-agent survey marked many entries as from memory.
- Items that came from an agent's memory and not from a source are **[Unverified]**.
  Speeds are the authors' own claims unless stated.
- The agents were given our world's description and criteria, so "no contender" is
  partly a product of those criteria. It does not mean the other worlds are weak.

## What each family offers

### Single-agent task generators

| Candidate | Authoring | Difficulty axes | Observation | Speed | Optics puzzle? |
| --- | --- | --- | --- | --- | --- |
| MiniGrid / BabyAI | Low: a Python subclass | Size, partial observability, instruction depth | 7x7 symbolic, RGB render | CPU | Hacky; discrete directions only |
| NAVIX | Medium: JAX | Same as MiniGrid | Symbolic | Over 160k times faster, per authors | Hacky |
| XLand-MiniGrid | Medium to high: JAX, no guide | Rule-chain depth, hidden rules, size | Egocentric tiles | Millions of steps/s, per authors | No: rules are local |
| Craftax | High: a monolith | A fixed game | Symbolic | About 267k steps/s, per paper | No |
| MiniHack / NetHack | Low to medium: a description-file DSL | Traps, partial observability, stochasticity, long horizons | Glyphs, pixels | About 5k steps/s | Awkward **[Unverified]** |
| Griddly | Low: YAML | Partial observability, mobs, multi-agent | Vector, sprites | About 70k FPS per thread, claimed | Hard: no beam primitive |
| Kinetix | High to change the engine | Physics, one goal | Pixels, entities | JAX | No |
| Baba Is AI | Medium | Rule change | Grid | CPU | No |
| PHYRE, Virtual Tools, ProcGen, TextWorld, Crafter, Boxoban | Fixed or text-only | Fixed | Pixels or text | CPU | No |

### Multi-agent and social worlds

| Candidate | Authoring | Social content | Observation | Speed | Scripted agents |
| --- | --- | --- | --- | --- | --- |
| VMAS (our base) | Python tensors | Cooperation, adversaries, communication | Vectors, lidar, no camera | Tens of thousands of environments on a GPU | Yes (a football AI) |
| Melting Pot | Lua in Lab2D: heavy | Dilemmas, norms, reciprocity | Egocentric RGB | About 10k steps/s, CPU | Yes: background bots, the best construct |
| SocialJax / JaxMARL | JAX | Dilemmas, coordination | Small grids | Very fast, a claimed 3.4 million steps/s | Unverified |
| Neural MMO, Overcooked-AI, Hanabi, Football | Fixed domains | Survival, coordination, conventions | Mixed | CPU | Some |
| Concordia, Diplomacy harnesses, EnactToM | Language or Habitat | Negotiation, deception, theory of mind | Text or 3D | Slow | Unknown |

### Embodied and physics worlds

| Candidate | Authoring | Observation | Speed | Verdict for us |
| --- | --- | --- | --- | --- |
| Craftium (Minecraft-like) | Lua mods, days | First-person RGB | CPU, one process per environment | Flexible, not batched |
| ManiSkill3, Genesis, MJX, Isaac Lab | Python or JAX, plus assets | RGB-D, needs a batch renderer | Up to 30k FPS (ManiSkill3); Genesis claims far more | Fast, but mechanics are slow to author |
| Habitat, AI2-THOR, ProcTHOR | Weeks, assets dominate | Photoreal RGB | Hundreds of FPS | Cost goes to rendering and assets |
| Avalon, Alchemy, DMLab-30, ViZDoom, Meta-World | Engine-bound or fixed | Pixels or state | Mixed | Alchemy and DMLab-30 show distinct difficulty kinds |
| MineDojo, MineRL | Java mods | RGB | Under 50 steps/s | Stale: last commits 2023 and 2024 |

## The verdict

- **[Proposed]** No surveyed environment combines all of these: batched on the GPU,
  cheap to author arbitrary mechanics, a first-person camera with occlusion, scripted
  adversaries with limited vision, and several independent difficulty axes. Each is
  either extensible but CPU and grid-bound (MiniGrid, Griddly, MiniHack), fast on a GPU
  but fixed (Craftax, Jumanji, Kinetix, XLand-MiniGrid), or realistic but costly to
  extend (the 3D engines).
- **[Proposed]** No surveyed environment offers a light-and-mirror puzzle. Optics would
  be custom code in every one of them.
- **[Proposed]** Closest on one criterion at a time: Griddly and MiniHack on cheap
  authoring; XLand-MiniGrid on GPU rule chains; Melting Pot on the scenario construct;
  Craftium on flexibility.

## A critical reading

- **What this does not show:** that we are better. The surveyed worlds have things we
  lack: standard benchmarks, baselines, communities, declarative authoring and, often,
  far more speed.
- **Our flexibility claim is unproven.** Six mechanics have been built. None is an
  optics mechanic, and the north-star puzzle needs one. Each mechanic means writing
  code and tests, not editing a description, so extension is cheap but not declarative.
- **Our weak points:** about 350 world-steps/s at 256 worlds; VMAS's soft contacts
  needed tuning, and its box-against-box contact misbehaved; physics cost grows with the
  number of moving entities; VMAS is GPL-3.0; we have no learner, task, reward or
  evaluation yet.
- **Not surveyed:** DeepMind's Lab2D framework on its own (Melting Pot's engine, a
  general Lua 2D builder), generative or learned interactive worlds, and unpublished
  work.

## Ideas to borrow

- **[Proposed]** Griddly's declarative mechanic definitions: a small description layer
  for our zones and entities, so a scenario is configuration and not code.
- **[Proposed]** Melting Pot's scenario construct: a scenario is a world plus a
  population of background bots, which is what our chasers already are.
- **[Proposed]** XLand-MiniGrid's rule-chain task generation, for long horizons, and
  Baba's rule-change axis.
- **[Proposed]** Checks that a task is solvable and requires a minimum reasoning depth
  (from EnactToM).
- **[Proposed]** Using a language model to write scenario code that a verifier then
  checks (ReSyn, Feb 2026), applied to our tensor scenarios.

## Open questions

- **[Open]** Whether to build a declarative authoring layer, and when.
- **[Open]** Whether to build the optics mechanic soon, to test the flexibility claim
  where it matters.
- **[Open]** A deeper look at Griddly, MiniHack and Lab2D, which came closest on
  authoring.
