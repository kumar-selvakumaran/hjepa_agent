"""Helpers around agent_world for data generation: dispersed start poses (a simulation privilege, SP-02 and SP-13)."""

import torch


def _min_dist(cand, entities):
    """Distance from each candidate [N, K, 2] to the nearest of the entities' positions; inf if there are none."""
    if not entities:
        return torch.full(cand.shape[:2], float("inf"), device=cand.device)
    pos = torch.stack([e.state.pos for e in entities], 1)  # [N, E, 2]
    return torch.cdist(cand, pos).min(-1).values


@torch.no_grad()
def disperse_starts(env, n_candidates=64, wall_margin=0.3):
    """Teleport every world's agent 0 to a random collision-free pose, uniform over free space in all rooms, with a
    random heading and zero velocity. Rejects candidates near obstacles, blocks and interior walls (so never in the
    doorway gap). Worlds with no accepted candidate keep their pose. Returns agent 0's re-rendered observation
    [N, 48, 64, 3] (env.reset() returns frames rendered at the old pose)."""
    sc, dev, n = env.scenario, env.device, env.num_envs
    c = sc.cfg
    agent = env.agents[0]
    edge = c.wall_thickness / 2 + 2 * c.agent_radius
    w, h = c.arena_width, c.arena_height
    u = torch.rand(n, n_candidates, 2, device=dev)
    cand = torch.stack([-w / 2 + edge + u[..., 0] * (w - 2 * edge), -h / 2 + edge + u[..., 1] * (h - 2 * edge)], -1)
    ok = _min_dist(cand, sc._obstacles) > c.obstacle_size / 2 * 2**0.5 + c.agent_radius + 0.05
    ok &= _min_dist(cand, sc._blocks) > c.block_size / 2 * 2**0.5 + c.agent_radius + 0.05
    for k in range(1, c.n_rooms):  # interior walls
        ok &= (cand[..., 0] - (-w / 2 + k * c.room_size)).abs() > wall_margin
    has = ok.any(1)
    pick = ok.int().argmax(1)
    pos = torch.where(has[:, None], cand[torch.arange(n), pick], agent.state.pos)
    rot = torch.where(has[:, None], (torch.rand(n, 1, device=dev) * 2 - 1) * torch.pi, agent.state.rot)
    agent.set_pos(pos, None)
    agent.set_rot(rot, None)
    agent.set_vel(torch.zeros(n, 2, device=dev), None)
    agent.set_ang_vel(torch.zeros(n, 1, device=dev), None)
    return env.get_from_scenario(get_observations=True, get_rewards=False, get_infos=False, get_dones=False)[0][0]
