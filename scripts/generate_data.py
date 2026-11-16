"""Generate frames, actions and ground truth from agent_world with a random-walk policy.

uv run python scripts/generate_data.py --name pretrain_100 --envs 512 --steps 500 --chasers 0 --seed 100
Writes data/<name>/{frames.npy, truth.npz, config.json}; the layout is the shard contract in
docs/journal/stage1-spec.md section 11. Start poses are dispersed over free space in both rooms.
"""

import argparse
import dataclasses
import importlib.metadata
import json
import subprocess
import time

import numpy as np
import torch
from agent_world import WorldConfig, make_env

from hjepa_agent.config import DATA_DIR, PROJECT_DIR
from hjepa_agent.policies import RandomWalkPolicy
from hjepa_agent.world import disperse_starts


def _git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def _world_rev():
    try:
        d = json.loads(importlib.metadata.distribution("agent-world").read_text("direct_url.json"))
        return d.get("vcs_info", {}).get("commit_id")
    except Exception:
        return None


def generate(name, envs, steps, chasers, patches, seed, out_root=None, device=None):
    cfg = WorldConfig(chasers_per_room=chasers, rotating_patches=patches)
    env = make_env(num_envs=envs, seed=seed, config=cfg, device=device)
    sc, dev = env.scenario, env.device
    pol = RandomWalkPolicy(env)
    ci = [i for i, (e, _) in enumerate(sc._visible) if e.name.startswith("chaser")]
    out = (out_root or DATA_DIR) / name
    out.mkdir(parents=True, exist_ok=True)
    frames = np.lib.format.open_memmap(out / "frames.npy", "w+", np.uint8, (envs, steps + 1, 48, 64, 3))
    pos, rot, cpos, acts, bump, fpos, fpres = [], [], [], [], [], [], []

    def truth():
        s = sc.scene()
        pos.append(s.pos[:, 0].cpu()); rot.append(s.rot[:, 0].cpu())
        cpos.append(s.pos[:, ci].cpu() if ci else torch.zeros(envs, 0, 2))
        fpos.append(sc._food_xy.cpu().clone()); fpres.append((sc._food_timer <= 0).cpu())

    env.reset()
    obs0 = disperse_starts(env)
    frames[:, 0] = (obs0.clamp(0, 1) * 255).byte().cpu().numpy()
    obst = torch.stack([o.state.pos for o in sc._obstacles], 1).cpu()
    block = torch.stack([b.state.pos for b in sc._blocks], 1).cpu() if sc._blocks else torch.zeros(envs, 0, 2)
    oh = sc._height[[sc._index[o.name] for o in sc._obstacles]].cpu()
    truth()
    t0 = time.time()
    for t in range(steps):
        act = pol()
        acts.append(act.cpu())
        obs, _, _, _ = env.step([act] + [torch.zeros(envs, 2, device=dev) for _ in env.agents[1:]])
        frames[:, t + 1] = (obs[0].clamp(0, 1) * 255).byte().cpu().numpy()
        truth()
        b = getattr(sc._chasers, "touched", None)
        bump.append((b[0] if b is not None else torch.zeros(envs, dtype=torch.bool)).cpu())
        if (t + 1) % 100 == 0:
            print(f"step {t + 1}/{steps}  {envs * (t + 1) / (time.time() - t0):.0f} world-steps/s", flush=True)
    frames.flush()
    st = lambda xs: torch.stack(xs, 1).numpy()
    np.savez_compressed(
        out / "truth.npz", agent_pos=st(pos).astype(np.float32), agent_rot=st(rot).astype(np.float32),
        actions=st(acts).astype(np.float32), bump=st(bump).astype(bool), chaser_pos=st(cpos).astype(np.float32),
        obstacle_pos=obst.numpy().astype(np.float32), obstacle_height=oh.expand(envs, -1).numpy().astype(np.float32),
        block_pos=block.numpy().astype(np.float32), food_pos=st(fpos).astype(np.float32), food_present=st(fpres).astype(bool),
        walk_scale=pol.scale.cpu().numpy().astype(np.float32), walk_p=pol.p.cpu().numpy().astype(np.float32))
    (out / "config.json").write_text(json.dumps({
        "world": dataclasses.asdict(cfg),
        "policy": {"name": "RandomWalkPolicy", "scale_range": [0.3, 1.0], "p_range": [0.04, 1.0], "pause_prob": 0.15,
                   "forward_range": [-0.5, 1.0], "turn_range": [-1.0, 1.0]},
        "seed": seed, "envs": envs, "steps": steps,
        "revisions": {"hjepa_agent": _git_rev(), "agent_world": _world_rev()}}, indent=1, default=str))
    print(f"saved {out}  frames {frames.shape}  {frames.nbytes / 1e9:.2f} GB  in {time.time() - t0:.0f}s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--chasers", type=int, default=2, help="chasers per room")
    ap.add_argument("--patches", action="store_true", help="rotating patch in each doorway (off by default: introduced after pretraining)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    generate(a.name, a.envs, a.steps, a.chasers, a.patches, a.seed)


if __name__ == "__main__":
    main()
