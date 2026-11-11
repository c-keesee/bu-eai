"""
memory_test.py -- v0.3: does the brain remember WHAT IS WHERE in a new place?

This is v0.2's new-map test (newmap_test.py) moved into the lively world.
v0.2 found that the brain could work out where IT was in a place it had never
seen, but did not store where the bell was. The memory conditions of v0.3 are
meant to help with exactly that, so we measure it again.

Frozen brains are dropped into maps they have never seen and wander for 400
moments. Linear probes then try to read three things out of the mind state:

    position   where am I?                 (a skill: find walls, count steps)
    bell       where is the bell?          (a fact about THIS map)
    ball       where is the first ball?    (a fact about this map that can change)

As in v0.2 there is ONE probe per question for all maps, fitted on one set of
new maps and scored on a different set, so it cannot memorise any map.
Baselines: the same network untrained, and chance (always guess the average).
"""

import torch

from .lively_world import LivelyWorld
from .newmap_test import LinearProbe
from .training import Life

TARGETS = ("position", "bell", "ball")
FIT_SEED, HELD_OUT_SEED = 60_000, 70_000       # different from training and from the common-sense scenes
WINDOWS = {"arrival (1-10)": (0, 10), "90-110": (89, 110), "350-400": (349, 400)}


@torch.no_grad()
def explore(brain, seed, n_agents=200, n_steps=400, memoryless=False, world_args=None):
    """n_agents copies of a frozen brain each live one life in their own new map."""
    world = LivelyWorld(**(world_args or {}))
    life = Life(brain, world, n_agents, seed=seed, memoryless=memoryless)
    out = life.live(n_steps, record=True)
    return {"mind": out["core"], **{t: out["truth"][t] for t in TARGETS}}


def where_curves(fit, held_out):
    """Fit one probe on `fit`, score it on `held_out`. Returns target -> error per moment (grid cells)."""
    X = fit["mind"].flatten(0, 1)
    Y = torch.cat([fit[t].flatten(0, 1) for t in TARGETS], dim=1)
    probe = LinearProbe(X, Y)
    curves = {t: [] for t in TARGETS}
    for step in range(held_out["mind"].shape[0]):
        guess = probe(held_out["mind"][step])
        for i, t in enumerate(TARGETS):
            curves[t].append((guess[:, 2 * i:2 * i + 2] - held_out[t][step]).norm(dim=1).mean().item())
    return curves


def where_test(trained, untrained, seed, memoryless=False, world_args=None, n_agents=200):
    """Returns {"trained" | "untrained" | "chance": {target: {window: mean error in cells}}}."""
    out = {}
    for label, brain in (("trained", trained), ("untrained", untrained)):
        fit = explore(brain, FIT_SEED + seed, n_agents, memoryless=memoryless, world_args=world_args)
        held_out = explore(brain, HELD_OUT_SEED + seed, n_agents, memoryless=memoryless, world_args=world_args)
        curves = where_curves(fit, held_out)
        out[label] = {t: {w: float(torch.tensor(c[a:b]).mean()) for w, (a, b) in WINDOWS.items()}
                      for t, c in curves.items()}
        if label == "trained":            # chance: always guess the average over the fitting maps
            out["chance"] = {}
            for t in TARGETS:
                average = fit[t].flatten(0, 1).mean(dim=0)
                error = (held_out[t] - average).norm(dim=2).mean(dim=1)
                out["chance"][t] = {w: float(error[a:b].mean()) for w, (a, b) in WINDOWS.items()}
    return out
