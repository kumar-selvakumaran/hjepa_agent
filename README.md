# hjepa_agent

An agent instantiated as proposed by Yann LeCun in the APTAMI paper (*A Path Towards Autonomous Machine Intelligence*), to be extended to do further specified system-2 reasoning. It is built to act in the sibling [`agent_world`](https://github.com/kumar-selvakumaran/agent_world).

**Where things stand:** the first component being built is the **world model**, trained jointly with its perception encoder, LeWM style, on random-walk experience from `agent_world`. It contains no goals, costs or planning; those belong to the actor and cost modules that come later. The reasoning behind every decision is in [`docs/journal/`](docs/journal/README.md) (start with `intentions.md` and `stage1-spec.md`).

## Setup

Everything runs in a project-local environment managed by [uv](https://docs.astral.sh/uv/). Downloads and outputs stay inside the project.

```bash
uv sync                      # creates .venv; torch comes from the CUDA 12.8 index, agent_world from its GitHub repository
uv run pytest -q             # tests (about 40 s)
```

On Windows, set `PYTHONUTF8=1` before long runs, otherwise the logger prints encoding errors (training is unaffected).

## Data

Random-walk experience, no chasers and no turning patch in the pretraining world:

```bash
uv run python scripts/generate_data.py --name pretrain_100 --envs 512 --steps 500 --chasers 0 --seed 100
```

Each shard `data/<name>/` holds `frames.npy`, `truth.npz` (ground truth, for evaluation and weighting only) and `config.json`; the format is the contract in `docs/journal/stage1-spec.md` section 11. Transfer sets for testing novelties the model has never seen: add `--chasers 2` or `--patches`.

## Train

```bash
uv run python scripts/train.py --data pretrain_100 pretrain_101 pretrain_102 pretrain_103 --run-name run1 --weighted
uv run python scripts/train.py --data <shards> --run-name run1 --preflight-only      # fail-fast checks only
uv run python scripts/train.py --data <shards> --run-name run1 --resume              # continue from the newest checkpoint
```

Outputs go to `outputs/<run-name>/`: `checkpoints/` (rolling steps, `last_good.ckpt`, `on_exception.ckpt`, `final.ckpt`), `csv/` and `tb/` (metrics; view with `uv run tensorboard --logdir outputs/run1/tb`), `preflight.json`, `env.json`, `summary.json`.

## Evaluate

```bash
uv run python scripts/eval_probes.py --data <shards> --features ckpt --ckpt outputs/run1/checkpoints/final.ckpt --out outputs/run1/probes_ckpt.json
uv run python scripts/eval_probes.py --data <shards> --features pixels --out outputs/run1/probes_pixels.json
uv run python scripts/eval_probes.py --data <shards> --features random --out outputs/run1/probes_random.json
uv run python scripts/gate.py --run outputs/run1      # pass or fail against the exit-gate thresholds
```

## Layout

```
src/hjepa_agent/   recipe, model (LeWM from stable-worldmodel), data, weights, callbacks, gate, policies, world
scripts/           generate_data, train, eval_probes, gate
tests/
docs/journal/      decisions, intentions, specifications, studies (append-only)
docs/agent_world/  reference snapshot of the world's own docs
```

Costly or risky commands (cloud runs) follow a step file the user runs; see `hitl/`.
