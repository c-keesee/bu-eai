"""
Sanity checks for the v0.3 brain, its episodic memory and the saved checkpoints.

    python -m pytest tests/ -q
"""

import numpy as np
import torch

from bueai.checkpoints import load_brain, save_brain
from bueai.commonsense_test import HABITUATION, Stage, continuity_jump
from bueai.lively_world import LivelyWorld
from bueai.model_v03 import BrainV3, EpisodicStore
from bueai.training import Life, TrainConfig, prediction_loss, train


def small_brain(**options):
    world = LivelyWorld()
    return world, BrainV3(world.sense_shapes, world.n_actions, core_units=32, filters=4, local_hidden=8, **options)


def test_same_interface_as_the_old_brain():
    world, brain = small_brain()
    obs = {name: torch.from_numpy(o) for name, o in world.reset(5).items()}
    state = brain.initial_state(5)
    assert state.shape == (5, 32 + 4 * 25) and state.abs().sum() == 0
    action = torch.eye(5)[[0, 1, 2, 3, 4]]
    state = brain.perceive(obs, action, state)
    prediction = brain.predict(state, action)
    assert {name: tuple(p.shape[1:]) for name, p in prediction.items()} == world.sense_shapes


def test_store_files_a_moment_every_few_moments_and_can_be_emptied():
    store = EpisodicStore(n_agents=3, slots=4, key_size=2, value_size=2, write_every=2)
    for moment in range(5):
        store.tick(torch.full((3, 2), float(moment)), torch.full((3, 2), float(moment)))
    assert store.used.sum(dim=1).tolist() == [3, 3, 3]                 # moments 0, 2 and 4 were filed
    assert store.values[0, :3, 0].tolist() == [0.0, 2.0, 4.0]
    store.forget([1])
    assert store.used.sum(dim=1).tolist() == [3, 0, 3]


def test_full_store_overwrites_its_oldest_memory():
    store = EpisodicStore(n_agents=1, slots=2, key_size=1, value_size=1, write_every=1)
    for moment in range(3):
        store.tick(torch.zeros(1, 1), torch.full((1, 1), float(moment)))
    assert sorted(store.values[0, :, 0].tolist()) == [1.0, 2.0]


def test_recall_is_empty_before_anything_is_stored_and_only_reading_is_learned():
    world, brain = small_brain(episodic_memory=True, memory_slots=8, write_every=1)
    life = Life(brain, world, n_agents=6, seed=0)
    first = life.live(1, record=True)
    recollection = first["core"][0][:, -brain.recall_size:]
    assert torch.allclose(recollection, recollection[:1].expand_as(recollection))   # nothing stored yet: the same
    # blank "recollection" for everyone, whatever they are looking at
    experience = life.live(6)
    sum(prediction_loss(experience).values()).backward()
    assert brain.key.weight.grad is None                               # the address code is never trained
    assert brain.query.weight.grad.abs().sum() > 0                     # the questions are


def test_new_life_empties_the_store():
    world, brain = small_brain(episodic_memory=True, memory_slots=8, write_every=1)
    life = Life(brain, world, n_agents=4, seed=0, life_length=10)
    life.age[:] = 0
    life.age[2] = 6
    with torch.no_grad():
        life.live(4)                                                   # agent 2 is reborn after the 4th moment
    assert life.memory.used.sum(dim=1).tolist() == [4, 4, 0, 4]


def test_training_with_memory_runs_and_reduces_error():
    torch.manual_seed(0)
    world, brain = small_brain(episodic_memory=True, memory_slots=16, write_every=4)
    history = train(brain, world, TrainConfig(n_updates=30, n_agents=16, life_length=64, learning_rate=3e-3, clip=None),
                    verbose=False)
    for name in ("vision", "touch"):
        assert np.mean(history[name][-5:]) < np.mean(history[name][:5]), name


def test_checkpoint_round_trip(tmp_path):
    world, brain = small_brain(episodic_memory=True)
    path = tmp_path / "brain.pt"
    save_brain(path, brain, {"note": "test"}, core_units=32, filters=4, local_hidden=8, episodic_memory=True)
    again, info = load_brain(path)
    assert info["note"] == "test" and again.has_memory
    assert all(torch.equal(a, b) for a, b in zip(brain.state_dict().values(), again.state_dict().values()))


def test_saved_watcher_brain_knows_that_rollers_keep_rolling():
    """The Stage 1 positive control: after seeing a roller move, it must bet on the next cell along."""
    brain, info = load_brain("results/v0.3/checkpoints/pilot/watcher.pt")
    assert "POSITIVE CONTROL" in info["note"]
    test = continuity_jump()
    n = 40
    stage = Stage(test, n, seed=3)
    world = stage.world
    to_torch = lambda obs: {name: torch.from_numpy(o) for name, o in obs.items()}
    magenta = torch.tensor([0.65, -0.05, 0.65])                        # roller colour minus floor colour
    roller_ness = lambda image: ((image - 0.25).permute(0, 2, 3, 1) @ magenta) / (magenta @ magenta)
    obs, state, stay = to_torch(world.observe()), brain.initial_state(2 * n), torch.eye(5)[[0] * (2 * n)]
    with torch.no_grad():
        for t in range(HABITUATION + 3):
            state = brain.perceive(obs, stay, state)
            prediction = brain.predict(state, stay)
            world.advance(stage.actions(t))
            stage.apply_events(t + 1)
            obs = to_torch(world.observe())
    # the last prediction was made after two seen moves; score it on the "rolls on normally" scenes
    where_it_went = roller_ness(obs["vision"][:n]) > 0.5
    assert (roller_ness(prediction["vision"][:n]) * where_it_went).sum(dim=(1, 2)).mean() > 0.8


def test_saved_v02_brains_load():
    brain, info = load_brain("results/v0.2/checkpoints/random_seed0.pt")
    assert brain.core_size == 256 and brain.has_curiosity and "v0.2" in info["note"]
