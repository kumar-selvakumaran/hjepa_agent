"""Exploration policy for generating world-model data from agent_world: a random walk. No goals, no planning."""

import math

import torch


class RandomWalkPolicy:
    """Random walk. Per episode: a speed scale in [0.3, 1.0] and a resample probability per step, drawn
    log-uniformly in [0.04, 1.0] (a new command every step up to a command held for about 25 steps).
    A new command: forward uniform in [-0.5, 1.0] x scale with a 15% chance of a pause (forward 0),
    turn uniform in [-1, 1]."""

    def __init__(self, env, p_range=(0.04, 1.0)):
        n, dev = env.num_envs, env.device
        self.scale = torch.rand(n, device=dev) * 0.7 + 0.3
        self.p = torch.exp(torch.empty(n, device=dev).uniform_(math.log(p_range[0]), math.log(p_range[1])))
        self.a = torch.zeros(n, 2, device=dev)
        self.fresh = torch.ones(n, dtype=torch.bool, device=dev)

    def __call__(self):
        n, dev = self.a.shape[0], self.a.device
        fwd = (torch.rand(n, device=dev) * 1.5 - 0.5) * self.scale
        fwd = torch.where(torch.rand(n, device=dev) < 0.15, torch.zeros_like(fwd), fwd)
        turn = torch.rand(n, device=dev) * 2 - 1
        resample = self.fresh | (torch.rand(n, device=dev) < self.p)
        self.a = torch.where(resample[:, None], torch.stack([fwd, turn], 1), self.a)
        self.fresh[:] = False
        return self.a
