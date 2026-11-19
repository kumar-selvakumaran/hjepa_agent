# Diagrams

Reference copies of Lucidchart diagrams. Source: Lucidchart document **System2-agent**, 3 pages
(`https://lucid.app/lucidchart/da281725-888d-434d-a34a-631a01aed117/edit`). Snapshot taken 2026-11-15. The Lucid original is the master; this is a read-only copy.

| Page in Lucid | Title | Copied here |
| --- | --- | --- |
| 1 | Reasoning Trace | Not copied (not asked) |
| 2 | Consistency Check | Not copied (not asked) |
| 3 | **HJepa-Agent** | `hjepa-agent.png` + transcription below |

## hjepa-agent.png

Left: LeCun (2022) Figure 2, embedded as an image in the Lucid page. Right: Kumar's redrawn module diagram.

### Modules (right side)

| Module | Colour | Notes |
| --- | --- | --- |
| Configurator | blue | Configures Perception, World Model, Actor and Cost |
| Perception | lilac | Reads from World |
| World Model | green | Writes to Short Term Memory; linked to Actor by "Planning" |
| Short Term Memory | orange | Fed by World Model |
| Actor | yellow | Acts on World |
| Cost | red frame | Contains **intrinsic cost** (red) and **Critic** (light red) |
| World | white circle | The environment (`agent_world`) |

### Edges

| From | To | Meaning |
| --- | --- | --- |
| World | Perception | Percept |
| World | Configurator | Percept (feeds configurator) |
| World | Cost | Percept (feeds cost) |
| Actor | World | Action |
| Configurator | Perception, World Model, Actor, Cost | Configuration (blue, thick) |
| World Model <-> Actor | | "Planning" (two-way) |
| World Model | Short Term Memory | State written (green) |
| Short Term Memory | World Model | Green, arrow at both ends |
| World Model | Critic | "Training the critic" (green) |
| Cost (Critic) | Actor | "Objective = cost" (red) |

Edge reading is from the Lucid JSON; the Configurator-to-Cost and World-to-Cost routing is read from the image and is **[unverified]**.
