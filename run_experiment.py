"""
run_experiment.py -- the whole v0.1 experiment, start to finish.

    python run_experiment.py                 # 3 seeds, a few minutes on a laptop CPU
    python run_experiment.py --quick         # 1 seed, short life: a fast smoke test
    python run_experiment.py --seeds 0 1 2 3 4 --updates 3000
    python run_experiment.py --no-feedback --out results_no_feedback   # ablation

For every random seed it:
    1. builds the world and a brand-new brain
    2. keeps a copy of that brain as the "untrained" baseline
    3. lets the brain live and learn from prediction error only
    4. freezes it and runs the emergence test (linear probes) under each
       sensing condition, and the dark-room test
Then it prints the results for all seeds side by side and saves three plots
plus the raw numbers into results/.

Why several seeds? A seed fixes every random choice (initial weights, where
the agents start, how they wander). One run could look good or bad by luck.
If three different seeds tell the same story, it is probably not a fluke.
"""

import argparse
import copy
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from bueai import plots
from bueai.darkroom_test import darkroom_test
from bueai.emergence_test import emergence_test
from bueai.model import Brain
from bueai.training import TrainConfig, train
from bueai.world import GridWorld

# The sensing conditions the probes are run under: label -> senses switched off.
CONDITIONS = {
    "lights on": (),
    "in darkness": ("vision",),                     # touch + hearing left
    "dark and silent": ("vision", "hearing"),       # only touch left
}
DARKROOM = "in darkness"          # which condition the dark-room test uses


def run_seed(seed, n_updates, use_feedback=True, verbose=True):
    """Everything for one seed. Returns a dictionary of plain numbers/lists."""
    torch.set_num_threads(1)      # seeds run side by side, one CPU core each
    torch.manual_seed(seed)       # fixes the brain's initial weights
    world = GridWorld()

    brain = Brain(world.sense_shapes, world.n_actions, use_feedback=use_feedback)
    untrained = copy.deepcopy(brain)          # the very same network, before its life

    config = TrainConfig(seed=seed, n_updates=n_updates)
    start = time.time()
    history = train(brain, world, config, verbose=verbose, tag=f"[seed {seed}] ")
    train_seconds = time.time() - start

    brain.eval()                              # frozen from here on: no more learning
    return {
        "seed": seed,
        "train_seconds": train_seconds,
        "chunk_len": config.chunk_len,
        "moments_lived": config.n_updates * config.chunk_len,
        "history": history,
        "probes": {
            label: emergence_test(brain, untrained, world, seed, blind)
            for label, blind in CONDITIONS.items()
        },
        "darkroom": darkroom_test(brain, world, seed, CONDITIONS[DARKROOM]),
        "darkroom_condition": DARKROOM,
        "darkroom_blind": list(CONDITIONS[DARKROOM]),
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _cell(values, digits):
    """'mean [seed values]' for one table cell."""
    values = np.asarray(values, dtype=float)
    each = " ".join(f"{v:.{digits}f}" for v in values)
    return f"{values.mean():.{digits}f} [{each}]" if len(values) > 1 else f"{values.mean():.{digits}f}"


def _table(title, row_names, col_names, get, digits=2):
    """Print a small table. get(row, col) -> list of per-seed values."""
    cells = [[_cell(get(r, c), digits) for c in col_names] for r in row_names]
    width = max(22, max(len(x) for row in cells for x in row) + 3)
    print(f"\n{title}")
    print(" " * 20 + "".join(c.ljust(width) for c in col_names))
    for name, row in zip(row_names, cells):
        print(f"  {name:<18}" + "".join(x.ljust(width) for x in row))


def print_report(runs):
    sources = ["trained core", "untrained core", "raw senses", "chance"]
    conditions = list(CONDITIONS)
    n = len(runs)
    line = "=" * 78

    print(f"\n{line}\nRESULTS   ({n} seed{'s' if n > 1 else ''}: "
          f"{', '.join(str(r['seed']) for r in runs)};  each cell = mean [one value per seed])\n{line}")
    print(f"Life lived per agent: {runs[0]['moments_lived']:,} moments. "
          f"Training time per seed: {np.mean([r['train_seconds'] for r in runs]):.0f} s.")

    print("\nPrediction error, start of life -> end of life (average of first / last 50 updates):")
    for sense in runs[0]["history"]:
        first = np.mean([np.mean(r["history"][sense][:50]) for r in runs])
        last = np.mean([np.mean(r["history"][sense][-50:]) for r in runs])
        print(f"  {sense:<8} {first:.4f} -> {last:.4f}   ({first / last:.0f}x lower)")

    print("\n--- EMERGENCE TEST: linear probes, scored on a held-out stretch of life ---")
    _table("WHERE AM I?  Position error in grid cells (lower is better)", sources, conditions,
           lambda s, c: [r["probes"][c][s]["position_error"] for r in runs])
    _table("WHICH OBJECT IS NEAREST?  Accuracy (higher is better)", sources, conditions,
           lambda s, c: [r["probes"][c][s]["object_accuracy"] for r in runs])

    print("\nHow to read it: the trained core has to beat BOTH the untrained core and the")
    print("raw senses for the result to count as something that emerged from experience.")
    for c in conditions:
        trained = np.array([r["probes"][c]["trained core"]["position_error"] for r in runs])
        untrained = np.array([r["probes"][c]["untrained core"]["position_error"] for r in runs])
        raw = np.array([r["probes"][c]["raw senses"]["position_error"] for r in runs])
        print(f"  Position, {c}: trained core beats the untrained core in "
              f"{int((trained < untrained).sum())}/{n} seeds, the raw senses in {int((trained < raw).sum())}/{n} seeds.")

    print(f"\n--- DARK-ROOM TEST: prediction error after vision is blanked ({DARKROOM}) ---")
    cols = ["lights on", "lights out", "guess 'no change'", "guess the average"]
    keys = dict(zip(cols, ["lit_mean", "dark_mean", "no_change", "average"]))
    senses = [k for k in runs[0]["darkroom"] if k != "lit_steps"]
    names = {s: (f"{s} (imagined)" if s in runs[0]["darkroom_blind"] else s) for s in senses}
    _table("Mean squared prediction error over the 100 dark moments (lower is better)",
           [names[s] for s in senses], cols,
           lambda row, c: [r["darkroom"][next(s for s in senses if names[s] == row)][keys[c]] for r in runs],
           digits=4)
    print("\n'(imagined)': the brain's guess at what it WOULD see, compared with the true")
    print("image it was never shown. The last two columns are brain-free yardsticks.")
    print(line)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BU-EAI v0.1 experiment")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--updates", type=int, default=TrainConfig().n_updates,
                        help="weight updates per seed (life length = updates * chunk_len)")
    parser.add_argument("--quick", action="store_true", help="1 seed, short life (smoke test)")
    parser.add_argument("--jobs", type=int, default=None,
                        help="seeds to run at the same time (default: one per CPU core)")
    parser.add_argument("--no-feedback", action="store_true",
                        help="ablation: cut the top-down connections (core -> sensory regions)")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    if args.quick:
        args.seeds, args.updates = [0], 150

    world = GridWorld()
    print("BU-EAI v0.1  --  the world ('#' wall, '.' floor, B bell, F fountain, m moss):\n")
    print(world.render_ascii())
    print(f"\nSenses: {world.sense_shapes}")
    print(f"Running seeds {args.seeds}, {args.updates} updates each"
          f"{' (top-down feedback CUT)' if args.no_feedback else ''}...\n")

    # The seeds do not depend on each other, so we run them at the same time
    # in separate processes (one CPU core each). With --jobs 1 they simply
    # run one after another in this process.
    use_feedback = not args.no_feedback
    jobs = args.jobs or min(len(args.seeds), os.cpu_count() or 1)
    if jobs > 1:
        n = len(args.seeds)
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            runs = list(pool.map(run_seed, args.seeds, [args.updates] * n, [use_feedback] * n))
    else:
        runs = [run_seed(seed, args.updates, use_feedback) for seed in args.seeds]

    print_report(runs)

    os.makedirs(args.out, exist_ok=True)
    plots.plot_training_loss(runs, os.path.join(args.out, "training_loss.png"))
    plots.plot_probe_results(runs, os.path.join(args.out, "probe_results.png"))
    plots.plot_darkroom(runs, os.path.join(args.out, "darkroom.png"))
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(runs, f)
    print(f"\nSaved to {args.out}/: training_loss.png, probe_results.png, darkroom.png, results.json")


if __name__ == "__main__":      # required so the worker processes can import this file safely
    main()
