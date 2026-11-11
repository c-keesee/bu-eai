"""
Sanity checks for the v0.3 world and common-sense scenes (a few seconds).

    python -m pytest tests/ -q
"""

import numpy as np
import torch

from bueai.commonsense_test import HABITUATION, LEGAL_EVENTS, Stage, all_tests, run_test, score
from bueai.lively_world import (BALL, BLOCKS_AGENT, FLOOR, ROLLER, SCREEN, WALL, LivelyWorld, lively_map)
from bueai.many_worlds import _connected
from bueai.model import Brain
from bueai.training import TrainConfig, train

N, E, S, W = 1, 2, 3, 4


def room(rows, movers=(), n_balls=2, n_rollers=2):
    """A one-agent world from a text picture. '@' agent, 'o' ball, 'r' roller, 'S' screen."""
    codes = {".": FLOOR, "#": WALL, "S": SCREEN, "@": FLOOR, "o": FLOOR, "r": FLOOR}
    size = max(len(rows), len(rows[0]))                       # the world is square: pad with wall
    grid = np.full((size, size), WALL)
    grid[:len(rows), :len(rows[0])] = [[codes[c] for c in row] for row in rows]
    world = LivelyWorld(size=size, n_balls=n_balls, n_rollers=n_rollers, p_rest=0.0, p_start=0.0)
    world.shape = (len(rows), len(rows[0]))
    world._allocate(1)
    world.set_maps(np.array([0]), grid[None])
    world.alive[:] = False
    slots = {"o": list(np.flatnonzero(world.kind == BALL)), "r": list(np.flatnonzero(world.kind == ROLLER))}
    for y, row in enumerate(rows):
        for x, c in enumerate(row):
            if c == "@":
                world.pos[0] = (y, x)
            elif c in "or":
                j = slots[c].pop(0)
                world.alive[0, j], world.obj_pos[0, j] = True, (y, x)
    return world


def picture(world):
    height, width = world.shape
    return [row[:width] for row in world.render_ascii(0).split("\n")[:height]]


def act(world, *actions):
    for a in actions:
        world.step(np.array([a]))


def test_random_maps_are_usable():
    rng = np.random.default_rng(0)
    for _ in range(30):
        grid = lively_map(rng)
        assert _connected(~BLOCKS_AGENT[grid])
        assert (grid == SCREEN).sum() >= 1


def test_pushed_ball_rolls_until_blocked():
    world = room(["#######",
                  "#@o...#",
                  "#######"])
    act(world, E)
    assert picture(world)[1] == "#.@o..#"          # pushed one cell, agent follows
    act(world, 0)
    assert picture(world)[1] == "#.@.o.#"          # keeps rolling on its own
    act(world, 0, 0, 0)
    assert picture(world)[1] == "#.@..o#"          # stopped by the wall, stays there


def test_ball_against_a_wall_cannot_be_pushed():
    world = room(["#####",
                  "#.@o#",
                  "#####"])
    act(world, E)
    assert picture(world)[1] == "#.@o#"            # nothing moved


def test_ball_knocks_on_another_ball():
    world = room(["########",
                  "#@o.o..#",
                  "########"])
    act(world, E, 0, 0, 0, 0)
    assert picture(world)[1] == "#.@o..o#"         # first ball stopped, second took over the motion


def test_ball_does_not_move_by_itself():
    world = room(["#####",
                  "#@.o#",
                  "#...#",
                  "#####"])
    act(world, S, N, 0, 0)
    assert picture(world)[1] == "#@.o#"


def test_roller_rolls_turns_and_pushes():
    world = room(["#######",
                  "#r.o..#",
                  "#######"])
    world.obj_dir[0, 2] = E
    act(world, 0)
    assert picture(world)[1] == "#.ro..#"
    act(world, 0)
    assert picture(world)[1] == "#..ro.#"          # the roller pushed the ball
    act(world, 0, 0)
    assert picture(world)[1] == "#...ro#"          # ball at the wall, roller right behind it
    act(world, 0, 0)
    assert picture(world)[1] == "#.r..o#"          # blocked: the roller turned round and left


def test_screens_hide_movers_and_stop_the_agent():
    world = room(["######",
                  "#r.S.#",
                  "#..@.#",
                  "######"])
    world.obj_dir[0, 2] = E
    yellow_or_magenta = lambda obs: (np.abs(obs["vision"][0].transpose(1, 2, 0) - (0.9, 0.2, 0.9)).sum(axis=2) < 1e-6).any()
    assert yellow_or_magenta(world.step(np.array([0])))          # roller at x=2: visible
    obs = world.step(np.array([0]))                              # roller under the screen
    assert picture(world)[1] == "#..R.#" and not yellow_or_magenta(obs)
    assert obs["touch"].reshape(5, -1)[1].argmax() == 5          # the agent feels cloth to its north
    obs = world.step(np.array([N]))                              # the agent cannot walk onto a screen
    assert tuple(world.pos[0]) == (2, 3)
    assert picture(world)[1] == "#..Sr#" and yellow_or_magenta(obs)


def test_nothing_ever_shares_a_cell():
    world = LivelyWorld(seed=3)
    world.reset(60)
    rng = np.random.default_rng(0)
    for _ in range(150):
        world.step(rng.integers(0, 5, size=60))
        cells = np.concatenate([world.pos[:, None], world.obj_pos], axis=1)       # (agents, 1 + movers, 2)
        flat = cells[:, :, 0] * 100 + cells[:, :, 1]
        assert all(len(np.unique(row)) == len(row) for row in flat)
        y, x = world.obj_pos[..., 0], world.obj_pos[..., 1]
        assert not world.blocks_objects[np.arange(60)[:, None], y, x].any()
        assert not world.blocks_agent[np.arange(60), world.pos[:, 0], world.pos[:, 1]].any()


def test_same_seed_same_world():
    def run(seed):
        world = LivelyWorld()
        world.reset(8, seed=seed)
        rng = np.random.default_rng(1)
        for _ in range(40):
            obs = world.step(rng.integers(0, 5, size=8))
        return obs["vision"], world.obj_pos.copy()
    (a, pa), (b, pb), (c, pc) = run(5), run(5), run(6)
    assert np.array_equal(a, b) and np.array_equal(pa, pb) and not np.array_equal(pa, pc)


def test_possible_versions_use_no_magic():
    """A 'possible' version must be nothing but the simulator running."""
    for test in all_tests():
        for version in test.versions:
            if version.possible:
                assert not version.stuck and not version.ghost_walls, (test.name, version.label)
                assert all(event[1] in LEGAL_EVENTS for event in version.events), (test.name, version.label)
            else:
                magic = [e for e in version.events if e[1] not in LEGAL_EVENTS]
                assert version.stuck or version.ghost_walls or magic, (test.name, version.label)


def test_versions_are_identical_until_the_event():
    """All versions of a scene that start the same must look the same up to the window."""
    for test in all_tests():
        stage = Stage(test, 4, seed=0)
        frames = []
        for t in range(test.window[1] + 1):
            frames.append(stage.world.observe()["vision"].reshape(len(test.versions), 4, -1).copy())
            stage.world.advance(stage.actions(t))
            stage.apply_events(t + 1)
        frames = np.stack(frames)                                 # (moments, versions, scenes, pixels)
        for impossible, possible in test.pairs:
            # each matched pair must differ somewhere (in what is seen, or in what
            # the agent does), or there would be nothing to test
            differs = (not np.array_equal(frames[:, impossible], frames[:, possible])
                       or test.versions[impossible].actions != test.versions[possible].actions)
            assert differs, test.name
        # Versions that start out looking the same must stay the same through the quiet period.
        for v in range(len(test.versions)):
            if np.array_equal(frames[0, v], frames[0, 0]):
                assert np.array_equal(frames[:HABITUATION, v], frames[:HABITUATION, 0]), (test.name, v)


def test_expect_no_change_cannot_pass_matched_pairs():
    """In the pairs whose windows look identical, a rule that ignores history must tie."""
    brain = Brain(LivelyWorld().sense_shapes, 5, core_size=16)
    for test in all_tests():
        if test.name.startswith("permanence_screen"):
            _, yardstick = run_test(brain, test, n_scenes=20)
            assert score(test, yardstick)["pair_rates"] == [0.5, 0.5], test.name


def test_training_runs_in_the_lively_world_and_reduces_error():
    torch.manual_seed(0)
    world = LivelyWorld()
    brain = Brain(world.sense_shapes, world.n_actions, core_size=64)
    history = train(brain, world, TrainConfig(n_updates=30, n_agents=16, life_length=64, learning_rate=1e-3),
                    verbose=False)
    for name in ("vision", "touch"):
        assert np.mean(history[name][-5:]) < np.mean(history[name][:5]), name
