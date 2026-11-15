"""Scripted exploration policies for generating data from agent_world. Batched: one target per world."""

import math

import torch


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class WaypointPolicy:
    """Drives to a target, then picks a new one. Targets: a random point in the current room (0.5), a point
    through the doorway (0.3), a random point anywhere (0.2). It does not seek or avoid chasers. Uses
    ground-truth pose for steering, so it is a data generator, not an agent."""

    def __init__(self, env, noise=0.1, max_hold=150):
        self.env, self.noise, self.max_hold = env, noise, max_hold
        self.cfg = env.scenario.cfg
        n, dev = env.num_envs, env.device
        self.target = torch.zeros(n, 2, device=dev)
        self.hold = torch.zeros(n, device=dev)
        self.fresh = torch.ones(n, dtype=torch.bool, device=dev)

    def _pose(self):
        s = self.env.scenario.scene()
        return s.pos[:, 0], s.rot[:, 0]

    def _new_targets(self, pos):
        c, n, dev = self.cfg, pos.shape[0], pos.device
        w, h, rs = c.room_size * c.n_rooms, c.room_size, c.room_size
        u = torch.rand(n, 3, device=dev)
        room = ((pos[:, 0] + w / 2) // rs).clamp(0, c.n_rooms - 1)
        in_room = torch.stack([-w / 2 + room * rs + 0.5 + u[:, 0] * (rs - 1), (u[:, 1] - 0.5) * (h - 1)], 1)
        anywhere = torch.stack([(u[:, 0] - 0.5) * (w - 1), (u[:, 1] - 0.5) * (h - 1)], 1)
        last = room >= c.n_rooms - 1
        door_x = torch.where(last, -w / 2 + room * rs, -w / 2 + (room + 1) * rs)  # the wall beside this room
        beyond = door_x + torch.where(last, -2.0, 2.0)
        door = torch.stack([beyond, torch.zeros_like(beyond)], 1)
        k = torch.rand(n, device=dev)
        return torch.where((k < 0.5)[:, None], in_room, torch.where((k < 0.8)[:, None], door, anywhere))

    def __call__(self):
        pos, rot = self._pose()
        redo = self.fresh | ((self.target - pos).norm(dim=-1) < 0.4) | (self.hold >= self.max_hold)
        self.target = torch.where(redo[:, None], self._new_targets(pos), self.target)
        self.hold = torch.where(redo, torch.zeros_like(self.hold), self.hold + 1)
        self.fresh[:] = False
        d = self.target - pos
        err = wrap(torch.atan2(d[:, 1], d[:, 0]) - rot)
        turn = (2.0 * err).clamp(-1, 1)
        fwd = torch.where(err.abs() < 0.4, torch.ones_like(err), torch.full_like(err, 0.2))
        a = torch.stack([fwd, turn], 1) + self.noise * torch.randn(pos.shape[0], 2, device=pos.device)
        return a.clamp(-1, 1)


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
