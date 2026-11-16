"""Generate frames, actions and ground truth from agent_world with a random-walk policy.

uv run python scripts/generate_data.py --name pretrain_100 --envs 512 --steps 500 --chasers 0 --seed 100
Writes data/<name>/{frames.npy [E,T+1,48,64,3] uint8, truth.npz}.
"""

import argparse
import time

import numpy as np
import torch
from agent_world import WorldConfig, make_env

from hjepa_agent.config import DATA_DIR
from hjepa_agent.policies import RandomWalkPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--chasers", type=int, default=2, help="chasers per room")
    ap.add_argument("--patches", action="store_true", help="rotating patch in each doorway (off by default: introduced after pretraining)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    env = make_env(num_envs=a.envs, seed=a.seed, config=WorldConfig(chasers_per_room=a.chasers, rotating_patches=a.patches))
    sc, dev = env.scenario, env.device
    pol = RandomWalkPolicy(env)
    ci = [i for i, (e, _) in enumerate(sc._visible) if e.name.startswith("chaser")]
    out = DATA_DIR / a.name
    out.mkdir(parents=True, exist_ok=True)
    frames = np.lib.format.open_memmap(out / "frames.npy", "w+", np.uint8, (a.envs, a.steps + 1, 48, 64, 3))
    pos, rot, cpos, acts, bump = [], [], [], [], []

    def truth():
        s = sc.scene()
        pos.append(s.pos[:, 0].cpu()); rot.append(s.rot[:, 0].cpu())
        cpos.append(s.pos[:, ci].cpu() if ci else torch.zeros(a.envs, 0, 2))

    obs = env.reset()
    frames[:, 0] = (obs[0].clamp(0, 1) * 255).byte().cpu().numpy()
    truth()
    t0 = time.time()
    for t in range(a.steps):
        act = pol()
        acts.append(act.cpu())
        obs, _, _, _ = env.step([act] + [torch.zeros(a.envs, 2, device=dev) for _ in env.agents[1:]])
        frames[:, t + 1] = (obs[0].clamp(0, 1) * 255).byte().cpu().numpy()
        truth()
        b = getattr(sc._chasers, "touched", None)
        bump.append((b[0] if b is not None else torch.zeros(a.envs, dtype=torch.bool)).cpu())
        if (t + 1) % 100 == 0:
            print(f"step {t + 1}/{a.steps}  {a.envs * (t + 1) / (time.time() - t0):.0f} world-steps/s", flush=True)
    frames.flush()
    np.savez_compressed(out / "truth.npz", agent_pos=torch.stack(pos, 1).numpy(), agent_rot=torch.stack(rot, 1).numpy(),
                        chaser_pos=torch.stack(cpos, 1).numpy(), actions=torch.stack(acts, 1).numpy(), bump=torch.stack(bump, 1).numpy())
    print(f"saved {out}  frames {frames.shape}  {frames.nbytes / 1e9:.2f} GB  in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
