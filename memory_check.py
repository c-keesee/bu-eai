"""
memory_check.py -- v0.3 follow-up: do the brains with an episodic store actually use it?

    python memory_check.py          # uses the saved brains in results/v0.3/checkpoints/

The common-sense tests showed no clear advantage for the brains with an
external memory. That could mean "the store is ignored" or "the store is used,
but not for what the tests measure". This script tells the two apart with a
simple intervention: let each brain wander new maps twice, once normally and
once with its store emptied before every moment ("amnesia"), and compare how
well it predicts its next sensation. Nothing is trained here.
Output is also saved to results/v0.3/memory_check.txt.
"""

import os

import numpy as np
import torch

from bueai.checkpoints import load_brain
from bueai.lively_world import LivelyWorld
from bueai.training import Life
from run_v03 import WORLD, nickname

OUT = "results/v0.3"
MAP_SEED = 80_000
PERIODS = {"moments 20-100": (20, 100), "moments 300-400": (300, 400)}


@torch.no_grad()
def wander(brain, seed, amnesia, n_agents=200, n_steps=400):
    """Mean prediction error per sense (x 1000) in each period of a life in new maps."""
    life = Life(brain, LivelyWorld(**WORLD), n_agents, seed=seed)
    errors = {name: [] for name in brain.sense_shapes}
    for _ in range(n_steps):
        if amnesia:
            life.memory.forget(np.arange(n_agents))        # nothing stored survives to the next moment
        moment = life.live(1)
        for name in errors:
            errors[name].append(moment["error"][name][0].mean().item())
    return {name: {p: 1000 * float(np.mean(e[a:b])) for p, (a, b) in PERIODS.items()} for name, e in errors.items()}


def main():
    torch.set_num_threads(os.cpu_count() or 1)
    lines = ["EPISODIC STORE CHECK: prediction error x 1000 with the store working / emptied every moment "
             "(and how much worse without it)", ""]
    for condition in ("memory", "both"):
        for seed in range(3):
            path = os.path.join(OUT, "checkpoints", f"{condition}_seed{seed}.pt")
            if not os.path.exists(path):
                continue
            brain, _ = load_brain(path)
            normal, without = wander(brain, MAP_SEED + seed, False), wander(brain, MAP_SEED + seed, True)
            lines.append(f"{condition} seed {seed} ({nickname(condition, seed)})   "
                         f"learned matching sharpness {brain.sharpness.exp().item():.1f}")
            for sense in normal:
                cells = [f"{p}: {normal[sense][p]:.3f} / {without[sense][p]:.3f} "
                         f"(+{100 * (without[sense][p] / normal[sense][p] - 1):.0f}%)" for p in PERIODS]
                lines.append(f"   {sense:<8} " + "    ".join(cells))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, "memory_check.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
