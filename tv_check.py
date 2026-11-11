"""
tv_check.py -- v0.2 follow-up: a closer look at the noisy TV.

    python tv_check.py            # uses the saved brains in results/v0.2/checkpoints/

The main run (Test 4) showed that NEITHER kind of curiosity got stuck at the
TV. This script asks why, by looking at three things separately:

    1. Is the TV really hard to predict?           (the world)
    2. Do the self-monitoring heads know that?     (the curiosity scores)
    3. Does the agent act on it?                   (the behaviour)

It loads the finished brains from results/v0.2/checkpoints/, lets each one live
400 moments in new maps moving in its own style, and prints one table.
Nothing is trained here. Output is also saved to results/v0.2/tv_check.txt.
"""

import os

import numpy as np
import torch

from bueai import checkpoints
from bueai.newmap_test import explore
from bueai.training import ERROR_SCALE, STAY

OUT = "results/v0.2"
SEEDS = [0, 1, 2]
STYLES = ["random", "error", "progress"]       # brain trained with this style, moving in this style
MAP_SEED = 30_000                              # different from training and from the other tests


def load_brain(style, seed):
    brain, _ = checkpoints.load_brain(os.path.join(OUT, "checkpoints", f"{style}_seed{seed}.pt"))
    return brain


@torch.no_grad()
def check(style, seed, n_agents=512, n_steps=400):
    """One brain, one set of new maps. Returns a dict of plain numbers."""
    brain = load_brain(style, seed)
    life = explore(brain, MAP_SEED + seed, movement=style, n_agents=n_agents, n_steps=n_steps)

    tv = life["tv_in_view"].bool().numpy()               # (moment, agent): TV in view right now?
    tv_now, tv_next = tv[:-1], tv[1:]

    # 1. The world: how wrong was the brain really, with and without the TV in
    #    the picture it was trying to predict? (Same units as the heads use.)
    real = sum(life["error"][name] for name in life["error"]).numpy()[:-1] * ERROR_SCALE

    # 2. The heads: ask them about all 5 actions at every moment.
    first, second = [], []
    for core in life["core"]:
        states = core.repeat_interleave(brain.n_actions, dim=0)
        actions = torch.eye(brain.n_actions).repeat(n_agents, 1)
        f, s = brain.expected_errors(states, actions)
        first.append(f.reshape(n_agents, -1))
        second.append(s.reshape(n_agents, -1))
    first, second = torch.stack(first).numpy(), torch.stack(second).numpy()
    progress = np.maximum(first - second, 0.0)            # the "learning progress" score

    return {
        "real error, TV in view":        real[tv_next].mean(),
        "real error, no TV":             real[~tv_next].mean(),
        "  ...no TV, moments 1-50":      real[:50][~tv_next[:50]].mean(),      # does ordinary scenery get
        "  ...no TV, moments 200-400":   real[200:][~tv_next[200:]].mean(),    # more predictable with time?
        "error score, TV in view":       first[tv].mean(),
        "error score, no TV":            first[~tv].mean(),
        "second-look guess, TV in view": second[tv].mean(),
        "second-look guess, no TV":      second[~tv].mean(),
        "progress score, TV in view":    progress[tv].mean(),
        "progress score, no TV":         progress[~tv].mean(),
        # 3. The behaviour.
        "TV stays in view next moment":  tv_next[tv_now].mean(),      # "holding on" to the TV
        "TV comes into view":            tv_next[~tv_now].mean(),     # "running into" the TV
        "share of time TV in view":      tv.mean(),
    }


def main():
    torch.set_num_threads(os.cpu_count() or 1)
    results = {style: [check(style, seed) for seed in SEEDS] for style in STYLES}

    L = ["NOISY-TV CHECK  (each brain moving in the style it was trained with; mean [one value per seed])",
         "Errors and scores are in the heads' units: mean squared error x 100, summed over the senses.", ""]
    L.append(" " * 32 + "".join(f"{s:<30}" for s in STYLES))
    for key in results[STYLES[0]][0]:
        cells = []
        for style in STYLES:
            values = [r[key] for r in results[style]]
            cells.append(f"{np.mean(values):.3f} [{' '.join(f'{v:.3f}' for v in values)}]")
        L.append(f"  {key:<30}" + "".join(f"{c:<30}" for c in cells))
    text = "\n".join(L)
    print(text)
    with open(os.path.join(OUT, "tv_check.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
