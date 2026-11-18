import importlib.util
import json
import pathlib

import numpy as np
import torch
from agent_world import WorldConfig, make_env

from hjepa_agent.world import disperse_starts

NO_NOVELTY = WorldConfig(chasers_per_room=0, rotating_patches=False)


def test_dispersed_starts_free_space_both_rooms_and_rerendered():
    env = make_env(num_envs=64, seed=1, device="cpu", config=NO_NOVELTY)
    obs_reset = env.reset()[0]
    sc, c = env.scenario, env.scenario.cfg
    obs = disperse_starts(env)
    assert obs.shape == obs_reset.shape and (obs - obs_reset).abs().mean() > 1e-3  # re-rendered at the new pose
    pos = sc.scene().pos[:, 0]
    assert pos[:, 0].abs().max() < c.arena_width / 2 and pos[:, 1].abs().max() < c.arena_height / 2
    assert (pos[:, 0] < 0).any() and (pos[:, 0] > 0).any()  # both rooms
    assert (pos[:, 0].abs() > 0.3).all()  # never in the wall or doorway
    for ents, size in ((sc._obstacles, c.obstacle_size), (sc._blocks, c.block_size)):
        p = torch.stack([e.state.pos for e in ents], 1)
        assert (torch.cdist(pos[:, None], p).min(-1).values > size / 2 * 2**0.5 + c.agent_radius).all()
    assert env.agents[0].state.vel.abs().max() == 0


def test_rendered_frame_matches_requested_pose():
    env = make_env(num_envs=2, seed=0, device="cpu", config=NO_NOVELTY)
    env.reset()
    a = env.agents[0]
    frames = []
    for xy in ([-6.0, 2.0], [6.0, -2.0]):
        a.set_pos(torch.tensor([xy, xy]), None)
        assert torch.allclose(env.scenario.scene().pos[:, 0], torch.tensor([xy, xy]))
        frames.append(env.get_from_scenario(True, False, False, False)[0][0].clone())
    assert (frames[0] - frames[1]).abs().mean() > 1e-3


def test_generate_writes_contract(tmp_path):
    spec = importlib.util.spec_from_file_location("gen", pathlib.Path(__file__).parents[1] / "scripts" / "generate_data.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    out = gen.generate("toy", envs=4, steps=6, chasers=2, patches=False, seed=3, out_root=tmp_path, device="cpu")
    E, T = 4, 6
    assert np.load(out / "frames.npy", mmap_mode="r").shape == (E, T + 1, 48, 64, 3)
    z = np.load(out / "truth.npz")
    exp = {"agent_pos": ((E, T + 1, 2), np.float32), "agent_rot": ((E, T + 1), np.float32), "actions": ((E, T, 2), np.float32),
           "bump": ((E, T), np.bool_), "chaser_pos": ((E, T + 1, 4, 2), np.float32), "obstacle_pos": ((E, 12, 2), np.float32),
           "obstacle_height": ((E, 12), np.float32), "block_pos": ((E, 6, 2), np.float32), "food_pos": ((E, T + 1, 48, 2), np.float32),
           "food_present": ((E, T + 1, 48), np.bool_), "walk_scale": ((E,), np.float32), "walk_p": ((E,), np.float32)}
    assert set(z.files) == set(exp)
    for k, (shape, dtype) in exp.items():
        assert z[k].shape == shape and z[k].dtype == dtype, k
        assert not np.isnan(z[k].astype(np.float32)).any(), k
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["seed"] == 3 and cfg["world"]["chasers_per_room"] == 2 and "revisions" in cfg
    assert cfg["world"]["floor"] == "map" and cfg["world"]["floor_map_seed"] == 0
