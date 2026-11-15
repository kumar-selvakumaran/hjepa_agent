import torch
from agent_world import WorldConfig, make_env

from hjepa_agent.policies import SmoothPolicy, WaypointPolicy


def test_policies_give_bounded_actions_and_waypoint_moves_agent():
    env = make_env(num_envs=4, seed=0, device="cpu", config=WorldConfig())
    env.reset()
    for cls in (SmoothPolicy, WaypointPolicy):
        pol = cls(env)
        a = pol()
        assert a.shape == (4, 2) and a.abs().max() <= 1.0
    pos0 = env.scenario.scene().pos[:, 0].clone()
    pol = WaypointPolicy(env)
    for _ in range(20):
        env.step([pol()])
    assert (env.scenario.scene().pos[:, 0] - pos0).norm(dim=-1).max() > 0.2
