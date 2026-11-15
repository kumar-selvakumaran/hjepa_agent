import torch
from agent_world import WorldConfig, make_env

from hjepa_agent.policies import SmoothPolicy


def test_random_walk_gives_bounded_actions_and_moves_agent():
    env = make_env(num_envs=4, seed=0, device="cpu", config=WorldConfig())
    env.reset()
    pol = SmoothPolicy(env)
    a = pol()
    assert a.shape == (4, 2) and a.abs().max() <= 1.0
    pos0 = env.scenario.scene().pos[:, 0].clone()
    for _ in range(20):
        env.step([pol()])
    assert (env.scenario.scene().pos[:, 0] - pos0).norm(dim=-1).max() > 0.2
