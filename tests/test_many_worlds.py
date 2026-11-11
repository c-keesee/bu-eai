"""
Sanity checks for the v0.2 building blocks (a few seconds, no long training).

    python -m pytest tests/ -q
"""

import copy

import numpy as np
import torch

from bueai.many_worlds import (V2_OBJECT_TYPES, V2_TEXTURES, ManyWorlds, _connected, grid_to_ascii,
                               is_asymmetric)
from bueai.model import Brain
from bueai.senses import Hearing, Touch, Vision
from bueai.training import STAY, Life, TrainConfig, curiosity_loss, train
from bueai.world import FIRST_OBJECT, GridWorld


def make(n_agents=40, seed=0):
    world = ManyWorlds(seed=seed)
    world.reset(n_agents)
    return world


def test_random_maps_are_usable():
    world = make()
    for i in range(world.grid.shape[0]):
        grid = world.grid[i]
        assert _connected(~world.solid[i])                       # every cell reachable
        assert is_asymmetric(grid)                               # no mirror / rotation matches
        for kind, count in ((0, 1), (1, 1), (2, 3), (3, 1)):     # 1 bell, 1 fountain, 3 moss, 1 TV
            assert (grid == FIRST_OBJECT + kind).sum() == count
        assert (~world.solid[i]).sum() >= 0.7 * 14 * 14          # mostly open floor


def test_every_map_is_different():
    world = make(200)
    assert len(world.map_fingerprints) == 200


def test_fast_senses_match_the_readable_ones():
    """many_worlds.py recomputes the three senses with array operations.
    They must agree exactly with the loop versions in senses.py."""
    world = make(6)
    for i in range(6):
        reference = GridWorld(grid_to_ascii(world.grid[i]), senses=[Vision(2), Touch(), Hearing()],
                              object_types=V2_OBJECT_TYPES, textures=V2_TEXTURES)
        free = ~world.solid[i]
        for name, table in world.tables.items():
            assert np.allclose(table[i][free], reference._sense_tables[name][free], atol=1e-6), name


def test_every_cell_is_distinguishable_by_its_senses():
    world = make(10)
    for i in range(10):
        free = np.argwhere(~world.solid[i])
        snapshot = np.concatenate([t[i][free[:, 0], free[:, 1]].reshape(len(free), -1)
                                   for t in world.tables.values()], axis=1)
        assert len(np.unique(snapshot.round(4), axis=0)) == len(free)


def test_tv_is_noise_and_everything_else_is_not():
    world = make(300)
    still = np.zeros(300, dtype=np.int64)                        # everyone stays put
    a, b = world.step(still), world.step(still)
    tv = world.ground_truth()["tv_in_view"]
    assert tv.any() and (~tv).any()
    changed = np.abs(a["vision"] - b["vision"]).reshape(300, -1).max(axis=1) > 0
    assert not changed[~tv].any()                                # no TV in view: nothing changes
    assert changed[tv].mean() > 0.6                              # TV in view: it usually flickers
    assert np.array_equal(a["touch"], b["touch"]) and np.array_equal(a["hearing"], b["hearing"])


def test_same_seed_same_worlds():
    a, b = ManyWorlds(), ManyWorlds()
    oa, ob = a.reset(12, seed=5), b.reset(12, seed=5)
    assert np.array_equal(a.grid, b.grid) and np.array_equal(oa["vision"], ob["vision"])
    c = ManyWorlds()
    c.reset(12, seed=6)
    assert not np.array_equal(a.grid, c.grid)


def test_new_life_gives_new_map_and_blank_mind():
    world = ManyWorlds()
    brain = Brain(world.sense_shapes, world.n_actions, core_size=32, curiosity_heads=True)
    life = Life(brain, world, n_agents=6, seed=0, life_length=10)
    life.age[:] = 0
    life.age[2] = 9                                              # agent 2 is about to finish its life
    before = world.grid.copy()
    with torch.no_grad():
        out = life.live(1)
    assert out["reborn"][0].tolist() == [False, False, True, False, False, False]
    assert not np.array_equal(before[2], world.grid[2])          # new map for agent 2
    assert np.array_equal(before[[0, 1, 3, 4, 5]], world.grid[[0, 1, 3, 4, 5]])
    assert life.core_state[2].abs().sum() == 0                   # blank mind
    assert life.core_state[0].abs().sum() > 0                    # the others keep theirs
    assert life.age[2] == 0 and life.last_action[2] == STAY


def test_curiosity_strength_zero_is_the_random_walk():
    """With no tilt, curious movement must have the random walk's action statistics."""
    world = ManyWorlds()
    brain = Brain(world.sense_shapes, world.n_actions, core_size=32, curiosity_heads=True)
    stats = {}
    for movement in ("random", "progress"):
        life = Life(brain, world, n_agents=400, seed=1, movement=movement, curiosity_strength=0.0)
        with torch.no_grad():
            actions = life.live(60)["actions"]
        repeats = (actions[1:] == actions[:-1]).float().mean().item()
        stats[movement] = (repeats, np.bincount(actions.flatten().numpy(), minlength=5) / actions.numel())
    assert abs(stats["random"][0] - stats["progress"][0]) < 0.02          # same momentum
    assert np.abs(stats["random"][1] - stats["progress"][1]).max() < 0.02  # same action mix


def test_curiosity_heads_do_not_touch_the_world_model():
    """The self-monitoring loss must send no gradient into the core, regions or predictors."""
    world = ManyWorlds()
    brain = Brain(world.sense_shapes, world.n_actions, core_size=32, curiosity_heads=True)
    life = Life(brain, world, n_agents=8, seed=0)
    sum(curiosity_loss(life.live(6)).values()).backward()
    for name, p in brain.named_parameters():
        is_head = name.startswith(("first_look", "second_look"))
        assert (p.grad is not None and p.grad.abs().sum() > 0) == is_head, name


def test_v02_training_runs_and_reduces_error():
    torch.manual_seed(0)
    world = ManyWorlds()
    brain = Brain(world.sense_shapes, world.n_actions, core_size=64, curiosity_heads=True)
    history = train(brain, world, TrainConfig(n_updates=30, n_agents=16, life_length=64,
                                              movement="progress", learning_rate=1e-3), verbose=False)
    for name in ("vision", "touch", "first_look"):
        assert np.mean(history[name][-5:]) < np.mean(history[name][:5]), name


def test_v01_brain_is_unchanged_by_v02_options():
    """Building a v0.1 brain must give exactly the same initial weights as before
    the v0.2 options existed (they are created after everything else)."""
    world = GridWorld()
    torch.manual_seed(0)
    plain = Brain(world.sense_shapes, world.n_actions)
    torch.manual_seed(0)
    with_heads = Brain(world.sense_shapes, world.n_actions, curiosity_heads=True)
    for (name, a), (_, b) in zip(plain.state_dict().items(), with_heads.state_dict().items()):
        assert torch.equal(a, b), name
    assert not plain.has_curiosity and copy.deepcopy(plain).has_curiosity is False
