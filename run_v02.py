"""
run_v02.py -- the v0.2 experiment: can the brain map a NEW place within one life?

    python run_v02.py                     # everything: 4 conditions x 3 seeds (about 2 hours on 4 cores)
    python run_v02.py --quick             # a 2-minute smoke test (numbers mean nothing)
    python run_v02.py --resume            # carry on after an interruption
    python run_v02.py --report-only       # just re-print the report and redraw the plots

(The v0.1 experiment is still `python run_experiment.py`.)

Four kinds of brain are trained, 3 seeds each. They share one architecture and
one world generator and differ in exactly one thing:

    random        moves by momentum random walk (as in v0.1)
    progress      curious: prefers actions with high expected learning progress
    error         naively curious: prefers actions with high expected error
    no_feedback   random movement, top-down connections cut

Every brain lives many short lives, each in a brand-new random map, and learns
only from sensory prediction error. Then it is frozen and tested in maps it
has never seen (bueai/newmap_test.py).

Each finished brain is saved under results/v0.2/runs/ straight away, so a long
run can be interrupted and resumed.
"""

import argparse
import copy
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from bueai import plots_v02
from bueai.many_worlds import ManyWorlds
from bueai.model import Brain
from bueai.newmap_test import movement_test, new_map_test, new_place_darkroom
from bueai.training import TrainConfig, train

# What differs between the four conditions.
CONDITIONS = {
    "random":      dict(movement="random",   use_feedback=True),
    "progress":    dict(movement="progress", use_feedback=True),
    "error":       dict(movement="error",    use_feedback=True),
    "no_feedback": dict(movement="random",   use_feedback=False),
}

# What they all share. See RESULTS.md for why these sizes were chosen.
BRAIN_SIZE = dict(core_size=256, predictor_size=256, curiosity_heads=True)
TRAINING = dict(n_agents=128, chunk_len=32, n_updates=6000, learning_rate=1e-3,
                final_learning_rate=3e-4, life_length=512, log_every=500)


def run_one(task):
    """Train and test one brain. task = (condition, seed, n_updates, out_dir, quick)."""
    condition, seed, n_updates, out_dir, quick = task
    setup = CONDITIONS[condition]
    torch.set_num_threads(1)          # several brains train side by side, one CPU core each
    torch.manual_seed(seed)           # fixes the initial weights (the same for all 4 conditions)

    world = ManyWorlds()
    brain = Brain(world.sense_shapes, world.n_actions, use_feedback=setup["use_feedback"], **BRAIN_SIZE)
    untrained = copy.deepcopy(brain)

    config = TrainConfig(seed=seed, movement=setup["movement"], **{**TRAINING, "n_updates": n_updates})
    start = time.time()
    history = train(brain, world, config, tag=f"[{condition} seed {seed}] ")
    train_seconds = time.time() - start
    brain.eval()

    n_agents = 64 if quick else 256
    new_map = new_map_test(brain, untrained, seed, setup["movement"], n_agents)
    test_maps = new_map.pop("maps")
    result = {
        "condition": condition,
        "seed": seed,
        "train_seconds": train_seconds,
        "moments_lived": config.n_updates * config.chunk_len,
        "maps_in_training": world.maps_made,
        # Honesty check: none of the test maps may have appeared in training.
        "test_maps": len(test_maps),
        "test_maps_seen_in_training": len(test_maps & world.map_fingerprints),
        "history": {k: _thin(v) for k, v in history.items()},
        "new_map": new_map,
        "movement": movement_test(brain, seed, n_agents=n_agents),
        "darkroom": new_place_darkroom(brain, untrained, seed, n_agents=n_agents),
    }

    os.makedirs(os.path.join(out_dir, "runs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "weights"), exist_ok=True)
    torch.save(brain.state_dict(), os.path.join(out_dir, "weights", f"{condition}_seed{seed}.pt"))
    with open(_run_file(out_dir, condition, seed), "w") as f:
        json.dump(_rounded(result), f)
    print(f"  [{condition} seed {seed}] done: trained {train_seconds / 60:.1f} min, "
          f"tested {(time.time() - start - train_seconds) / 60:.1f} min", flush=True)
    return condition, seed


def _run_file(out_dir, condition, seed):
    return os.path.join(out_dir, "runs", f"{condition}_seed{seed}.json")


def _thin(values, window=20):
    """Average a long loss history in blocks of `window` updates (smaller files)."""
    values = np.asarray(values[:len(values) // window * window], dtype=float)
    return values.reshape(-1, window).mean(axis=1).tolist() if len(values) else []


def _rounded(x):
    """Round every number to 4 decimals so the saved files stay small."""
    if isinstance(x, dict):
        return {k: _rounded(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_rounded(v) for v in x]
    return round(x, 4) if isinstance(x, float) else x


def load_runs(out_dir):
    """All finished runs: {condition: [result per seed, ordered by seed]}."""
    runs = {}
    for condition in CONDITIONS:
        for seed in range(100):
            path = _run_file(out_dir, condition, seed)
            if os.path.exists(path):
                with open(path) as f:
                    runs.setdefault(condition, []).append(json.load(f))
    return runs


# ---------------------------------------------------------------------------
# The printed report
# ---------------------------------------------------------------------------

# "Moments since arrival" windows used in the tables: label -> (first, last) moment.
WINDOWS = {"arrival (1-10)": (0, 10), "40-60": (40, 60), "90-110": (90, 110),
           "190-210": (190, 210), "350-400": (350, 400)}


def window_mean(curve, window):
    return float(np.mean(curve[window[0]:window[1]]))


def _cell(values, digits=2):
    values = np.asarray(values, dtype=float)
    each = " ".join(f"{v:.{digits}f}" for v in values)
    return f"{values.mean():.{digits}f} [{each}]" if len(values) > 1 else f"{values.mean():.{digits}f}"


def _table(lines, title, rows, cols, get, digits=2, name_width=22):
    """rows / cols are lists of labels; get(row, col) -> list of per-seed values."""
    cells = [[_cell(get(r, c), digits) for c in cols] for r in rows]
    width = max(max(len(x) for row in cells for x in row), max(len(c) for c in cols)) + 3
    lines.append(f"\n{title}")
    lines.append(" " * (name_width + 2) + "".join(c.ljust(width) for c in cols))
    for name, row in zip(rows, cells):
        lines.append(f"  {name:<{name_width}}" + "".join(x.ljust(width) for x in row))


def build_report(runs):
    """The whole report as a list of text lines."""
    L, bar = [], "=" * 100
    main = runs["random"]
    seeds = [r["seed"] for r in main]
    L.append(f"{bar}\nBU-EAI v0.2 RESULTS   (seeds {seeds}; each cell = mean [one value per seed])\n{bar}")
    L.append(f"Each brain lived {main[0]['moments_lived']:,} moments per agent across "
             f"about {main[0]['maps_in_training']:,} different maps.")
    for condition, rs in runs.items():
        minutes = np.mean([r["train_seconds"] for r in rs]) / 60
        seen = sum(r["test_maps_seen_in_training"] for r in rs)
        L.append(f"  {condition:<12} {len(rs)} seeds, {minutes:5.1f} min training each; "
                 f"test maps that had appeared in training: {seen} of {sum(r['test_maps'] for r in rs)}")

    L.append("\nPrediction error at the start and end of training (random-movement brains):")
    for sense in ("vision", "touch", "hearing"):
        first = np.mean([r["history"][sense][0] for r in main])
        last = np.mean([r["history"][sense][-1] for r in main])
        L.append(f"  {sense:<8} {first:.4f} -> {last:.4f}")

    sources = ["trained core", "untrained core", "raw senses", "chance"]
    cols = list(WINDOWS)
    L.append("\n--- TEST 1: NEW MAPS, LIGHTS ON, RANDOM MOVEMENT. Error in grid cells against moments since arrival ---")
    _table(L, "WHERE AM I?  (lower is better)", sources, cols,
           lambda s, c: [window_mean(r["new_map"]["curves"][s]["position"], WINDOWS[c]) for r in main])
    _table(L, "WHERE IS THE BELL?  (lower is better; the weights cannot know this)", sources, cols,
           lambda s, c: [window_mean(r["new_map"]["curves"][s]["bell"], WINDOWS[c]) for r in main])

    L.append("\n--- TEST 2: DARK ROOM IN A NEW PLACE. Lights out after 8 or 300 moments of looking around ---")
    dark_rows = [f"{s}, explored {n}" for s in ("trained core", "untrained core", "raw senses") for n in ("8", "300")]
    _table(L, "WHERE AM I, in the dark?  Mean over the 100 dark moments (chance: "
           f"{np.mean([np.mean(r['darkroom']['300']['position']['chance']) for r in main]):.2f})",
           dark_rows, ["position error"],
           lambda row, c: [np.mean(r["darkroom"][row.split()[-1]]["position"][row.split(",")[0]]) for r in main],
           name_width=30)
    senses = ["touch", "hearing", "vision"]
    _table(L, "PREDICTION ERROR in the dark (trained brain; vision = imagined, never shown)",
           senses, ["explored 8", "explored 300", "guess 'no change'", "guess the average"],
           lambda s, c: [np.mean(r["darkroom"]["8" if c == "explored 8" else "300"]["prediction"][s]) if c.startswith("explored")
                         else r["darkroom"]["300"]["no_change" if "no change" in c else "average"][s] for r in main],
           digits=4)

    if "no_feedback" in runs:
        L.append("\n--- TEST 3: DOES TOP-DOWN FEEDBACK MATTER HERE?  Same maps and walks, feedback on vs cut ---")
        pairs = [("with feedback", runs["random"]), ("feedback cut", runs["no_feedback"])]
        for target, title in (("position", "WHERE AM I?"), ("bell", "WHERE IS THE BELL?")):
            _table(L, f"{title}  (new maps, lights on)", [p[0] for p in pairs], cols,
                   lambda name, c: [window_mean(r["new_map"]["curves"]["trained core"][target], WINDOWS[c])
                                    for r in dict(pairs)[name]])
        _table(L, "WHERE AM I, in the dark, after exploring 300 moments", [p[0] for p in pairs], ["position error"],
               lambda name, c: [np.mean(r["darkroom"]["300"]["position"]["trained core"]) for r in dict(pairs)[name]])

    styles = [s for s in ("random", "progress", "error") if s in runs]
    if len(styles) > 1:
        L.append("\n--- TEST 4: MOVEMENT STYLES. Each brain moves the way it was trained to ---")
        own = lambda style, key: [r["movement"][style][key] for r in runs[style]]
        _table(L, "SHARE OF THE MAP SEEN  (higher is better)", styles, ["after 50", "after 100", "after 200", "after 400"],
               lambda s, c: [curve[int(c.split()[-1]) - 1] for curve in own(s, "coverage")])
        _table(L, "NOISY-TV CHECK: share of time with the TV in view", styles, ["moments 1-400", "moments 200-400"],
               lambda s, c: [np.mean(curve if "1-400" in c else curve[200:]) for curve in own(s, "tv_in_view")], digits=3)
        _table(L, "WHERE AM I?  (lower is better)", styles, cols,
               lambda s, c: [window_mean(curve, WINDOWS[c]) for curve in own(s, "position")])
        _table(L, "WHERE IS THE BELL?  (lower is better)", styles, cols,
               lambda s, c: [window_mean(curve, WINDOWS[c]) for curve in own(s, "bell")])

        L.append("\n--- TEST 4b: the SAME brain (trained with random movement) steered by each style ---")
        _table(L, "SHARE OF THE MAP SEEN", styles, ["after 50", "after 100", "after 200", "after 400"],
               lambda s, c: [r["movement"][s]["coverage"][int(c.split()[-1]) - 1] for r in main])
        _table(L, "NOISY-TV CHECK: share of time with the TV in view", styles, ["moments 1-400", "moments 200-400"],
               lambda s, c: [np.mean(r["movement"][s]["tv_in_view"] if "1-400" in c else r["movement"][s]["tv_in_view"][200:])
                             for r in main], digits=3)
        _table(L, "WHERE AM I?", styles, cols,
               lambda s, c: [window_mean(r["movement"][s]["position"], WINDOWS[c]) for r in main])
    L.append(bar)
    return L


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BU-EAI v0.2 experiment")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--updates", type=int, default=TRAINING["n_updates"])
    parser.add_argument("--jobs", type=int, default=None, help="brains to train at once (default: one per CPU core)")
    parser.add_argument("--quick", action="store_true", help="1 seed, very short training (smoke test)")
    parser.add_argument("--resume", action="store_true", help="skip brains that are already finished")
    parser.add_argument("--report-only", action="store_true", help="no training; report and plots from saved runs")
    parser.add_argument("--out", default="results/v0.2")
    args = parser.parse_args()
    if args.quick:
        args.seeds, args.updates = [0], 60
        if args.out == "results/v0.2":
            args.out = "results/v0.2_quick"

    if not args.report_only:
        tasks = [(c, s, args.updates, args.out, args.quick) for s in args.seeds for c in args.conditions
                 if not (args.resume and os.path.exists(_run_file(args.out, c, s)))]
        jobs = args.jobs or min(len(tasks), os.cpu_count() or 1)
        print(f"BU-EAI v0.2: training {len(tasks)} brains, {jobs} at a time, {args.updates} updates each.\n"
              f"An example map ('#' wall, B bell, F fountain, m moss, T noisy TV):\n")
        example = ManyWorlds(seed=0)
        example.reset(1)
        print(example.render_ascii(0) + "\n", flush=True)
        if jobs > 1:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                list(pool.map(run_one, tasks))
        else:
            for task in tasks:
                run_one(task)

    runs = load_runs(args.out)
    report = "\n".join(build_report(runs))
    print(report)
    with open(os.path.join(args.out, "report.txt"), "w") as f:
        f.write(report + "\n")
    saved = plots_v02.make_all(runs, args.out)
    print(f"\nSaved to {args.out}/: report.txt, " + ", ".join(saved))


if __name__ == "__main__":      # required so the worker processes can import this file safely
    main()
