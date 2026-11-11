"""
Quick sanity checks on the building blocks (a few seconds, no training run).

    python -m pytest tests/ -q

These are ordinary unit tests: they check the code does what it claims. They
are NOT the scientific tests (those are bueai/emergence_test.py and
bueai/darkroom_test.py, run by run_experiment.py).
"""

import copy
from collections import deque

import numpy as np
import torch

from bueai.model import Brain
from bueai.training import Life, TrainConfig, prediction_loss, train
from bueai.world import ACTION_NAMES, OBJECT_TYPES, GridWorld


def test_every_free_cell_is_reachable():
    world = GridWorld()
    start = tuple(world.free_cells[0])
    seen, queue = {start}, deque([start])
    while queue:                                    # flood fill
        y, x = queue.popleft()
        for dy, dx in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            cell = (y + dy, x + dx)
            if not world.solid[cell] and cell not in seen:
                seen.add(cell)
                queue.append(cell)
    assert len(seen) == len(world.free_cells)


def test_map_is_asymmetric():
    """No mirror image or rotation of the map equals the map, so no two
    places are interchangeable."""
    grid = GridWorld().grid
    variants = [np.fliplr(grid), np.flipud(grid), np.rot90(grid, 1), np.rot90(grid, 2),
                np.rot90(grid, 3), grid.T, np.rot90(grid, 2).T]
    assert not any(np.array_equal(grid, v) for v in variants)


def test_every_cell_is_distinguishable_by_its_senses():
    """Each free cell produces a different combination of sense values, so
    position is knowable in principle."""
    world = GridWorld()
    world.pos = world.free_cells.copy()             # one agent on every free cell
    obs = world.observe()
    flat = np.concatenate([v.reshape(len(world.pos), -1) for v in obs.values()], axis=1)
    assert len(np.unique(flat.round(4), axis=0)) == len(world.free_cells)


def test_each_object_type_shows_up_in_all_three_senses():
    world = GridWorld()
    for i, obj in enumerate(OBJECT_TYPES):
        y, x = next((y, x) for y, x, t in world.objects if t == i)
        # Stand on it if walkable, otherwise directly beside it.
        cell = (y, x) if not obj.solid else next(
            (y + dy, x + dx) for dy, dx in [(-1, 0), (0, 1), (1, 0), (0, -1)]
            if not world.solid[y + dy, x + dx])
        world.pos = np.array([cell])
        obs = world.observe()
        colors = obs["vision"][0].reshape(3, -1).T
        assert any(np.allclose(c, obj.color) for c in colors)        # seen
        assert obs["touch"][0].reshape(5, -1)[:, 2:].sum() >= 1       # felt (a non-wall texture)
        assert obs["hearing"][0][obj.sound_channel] >= obj.loudness / 2 - 1e-6   # heard


def test_walls_block_movement():
    world = GridWorld()
    world.pos = np.array([[1, 1]])                  # top-left corner
    world.step(np.array([ACTION_NAMES.index("north")]))
    assert world.pos.tolist() == [[1, 1]]
    world.step(np.array([ACTION_NAMES.index("east")]))
    assert world.pos.tolist() == [[1, 2]]


def test_brain_shapes():
    world = GridWorld()
    brain = Brain(world.sense_shapes, world.n_actions)
    obs = {k: torch.from_numpy(v) for k, v in world.reset(7).items()}
    action = torch.zeros(7, world.n_actions)
    core = brain.perceive(obs, action, brain.initial_state(7))
    assert core.shape == (7, brain.core_size)
    for name, guess in brain.predict(core, action).items():
        assert guess.shape == obs[name].shape


def test_brain_accepts_a_new_sense_without_changes():
    """The brain is built from a dict of shapes, so a made-up extra sense
    (and a different image size) needs no code change."""
    shapes = {"vision": (3, 9, 9), "touch": (25,), "smell": (4,)}
    brain = Brain(shapes, n_actions=6)
    obs = {k: torch.zeros(2, *s) for k, s in shapes.items()}
    core = brain.perceive(obs, torch.zeros(2, 6), brain.initial_state(2))
    assert set(brain.predict(core, torch.zeros(2, 6))) == set(shapes)


def test_blanked_outcomes_are_not_learned_from():
    world = GridWorld()
    brain = Brain(world.sense_shapes, world.n_actions)
    life = Life(brain, world, n_agents=4, seed=0)
    life.set_sense("vision", False)
    experience = life.live(5)
    assert experience["sensed"]["vision"].sum() == 0
    assert prediction_loss(experience)["vision"].item() == 0.0     # no signal from unseen outcomes
    assert prediction_loss(experience)["touch"].item() > 0.0


def test_same_seed_gives_the_same_walk():
    world = GridWorld()
    brain = Brain(world.sense_shapes, world.n_actions)
    with torch.no_grad():
        a = Life(brain, world, 8, seed=3).live(20, record=True)["truth"]["position"]
        b = Life(brain, world, 8, seed=3).live(20, record=True)["truth"]["position"]
    assert torch.equal(a, b)


def test_training_reduces_prediction_error():
    torch.manual_seed(0)
    world = GridWorld()
    brain = Brain(world.sense_shapes, world.n_actions)
    before = copy.deepcopy(brain.state_dict())
    history = train(brain, world, TrainConfig(n_updates=40, n_agents=16), verbose=False)
    for name, losses in history.items():
        assert np.mean(losses[-5:]) < np.mean(losses[:5]), name
    assert any(not torch.equal(before[k], v) for k, v in brain.state_dict().items())
