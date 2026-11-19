# Journal guide

This folder records the thinking behind `agent_world`: where the project came
from, what the world is meant to be, and the rules it is built by. Read this
page first; it says how to read the others.

## The documents

| Document | Answers | Changes when |
| --- | --- | --- |
| [journey.md](journey.md) | How did we get here, and what was established on the way? | A new line of thought is opened or closed |
| [world-spec.md](world-spec.md) | What must the world be and do? What exists so far? | A requirement is added, dropped or built |
| [principles.md](principles.md) | How do we design and build it? | A standing rule is added or changed |
| [physics-engines.md](physics-engines.md) | Which physics engine, and why? | An engine is measured, added or dropped |
| [curriculum.md](curriculum.md) | How could the agent be taught what the world needs? | A strategy is added, tried or dropped |
| [swm-notes.md](swm-notes.md) | What can we learn from stable-worldmodel, and how do we compare? | A source is studied or a comparison changes |
| [contenders.md](contenders.md) | Is any existing environment a contender for what this world is for? | A candidate is added, checked or dropped |

`journey.md` is history: it is added to, not rewritten. `world-spec.md` and
`principles.md` describe the current state: they are edited in place.

## Status labels

Every claim that matters carries one of these, so a reader can tell a decision
from a suggestion:

| Label | Meaning |
| --- | --- |
| **[Kumar]** | A position, requirement or instruction Kumar stated |
| **[Agreed]** | Proposed by one side and accepted by the other |
| **[Proposed]** | Suggested by Claude; not yet accepted or rejected |
| **[Open]** | A question nobody has settled |
| **[Built]** | Implemented and in the repository |
| **[Dropped]** | Considered and set aside; the reason is given |

If a statement has no label, treat it as background, not as a decision.

## How far to trust it

- **Source.** These documents were written by Claude on 2026-11-14 from the
  record of one long working session (roughly 2026-11-01 to 2026-11-14), much
  of it spoken. The early part was reconstructed from a summary, so detail from
  that part may be missing or compressed.
- **Papers.** Three papers were read directly in the session and are cited by
  section: Goyal & Bengio, *Inductive Biases for Deep Learning of Higher-Level
  Cognition* (2022); LeCun, *A Path Towards Autonomous Machine Intelligence*
  (2022); and Zhang et al., *H-JEPA* (2026). **Every other paper named here
  came from Claude's memory or a web search and has not been checked against
  the source.** Verify before citing. Some are flagged individually where
  Claude had only seen an abstract.
- **Corrections.** Claude made several claims in the session that it later
  withdrew. Only the corrected versions are recorded, marked "(corrected)".
- **Measurements.** Numbers are given with the machine they were measured on.
  Nothing here was measured on a GPU.

## Keeping it up to date

- Add to `journey.md` as a new dated section at the end. Do not edit old
  sections except to mark something as later reversed.
- When a requirement is built, change its label in `world-spec.md` to
  **[Built]**; do not delete it.
- When a **[Proposed]** item is accepted or rejected, relabel it
  **[Agreed]** or **[Dropped]** and say why in one line.
- Keep each entry short: the claim, its label, and the reason.
