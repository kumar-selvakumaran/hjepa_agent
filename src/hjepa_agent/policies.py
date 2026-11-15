"""Exploration policy for generating world-model data from agent_world: a random walk. No goals, no planning."""

import torch


class SmoothPolicy:
    """Persistent random walk: hold (forward, turn) and resample with probability 0.08 per step."""

    def __init__(self, env):
        n, dev = env.num_envs, env.device
        self.a = torch.zeros(n, 2, device=dev)
        self.fresh = torch.ones(n, dtype=torch.bool, device=dev)

    def __call__(self):
        n, dev = self.a.shape[0], self.a.device
        new = torch.stack([torch.rand(n, device=dev) * 0.8 + 0.2, torch.rand(n, device=dev) * 2 - 1], 1)
        r = self.fresh | (torch.rand(n, device=dev) < 0.08)
        self.a = torch.where(r[:, None], new, self.a)
        self.fresh[:] = False
        return self.a
