"""
run_v03.py -- the v0.3 experiment: does common sense about objects emerge?

    python run_v03.py                      # every condition x 3 seeds
    python run_v03.py --quick              # a 2-minute smoke test (numbers mean nothing)
    python run_v03.py --resume             # carry on after an interruption
    python run_v03.py --report-only        # re-print the report, redraw the plots
    python run_v03.py --conditions core --seeds 0     # one brain

(v0.2 is still `python run_v02.py`, v0.1 is `python run_experiment.py`.)

Each brain lives many short lives in the lively world (bueai/lively_world.py:
balls, rollers, screens) and learns only from sensory prediction error. It is
never shown an impossible event while growing up. Then it is frozen and given
the common-sense tests (bueai/commonsense_test.py), with its own prediction
error as the surprise meter.

The tests are also run at regular points DURING training, which gives a
"growing up" curve: how much experience each piece of common sense takes.

Finished brains are saved in results/v0.3/checkpoints/, so every test can be
re-run later without training again.
"""

import argparse
import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from bueai.commonsense_test import commonsense_tests
from bueai.lively_world import LivelyWorld
from bueai.model import Brain
from bueai.training import TrainConfig, train

# What differs between the conditions.
CONDITIONS = {
    # The v0.2 brain. Its only memory is the recurrent core's own state.
    "core": dict(),
    # Stage 1 check: the same brain with its core state wiped before every
    # moment. It cannot remember anything, so it MUST fail every test that
    # needs memory. If it passed one, that test would be broken.
    "memoryless": dict(memoryless=True),
}

# What they all share.
WORLD = dict(n_balls=3, n_rollers=3)
BRAIN_SIZE = dict(core_size=256, predictor_size=256)
TRAINING = dict(n_agents=128, chunk_len=32, n_updates=6000, learning_rate=1e-3,
                final_learning_rate=3e-4, decay_from=0.5, life_length=512, log_every=500)
TEST_SEED = 50_000          # the test scenes use their own maps, never the training ones


def run_one(task):
    """Train and test one brain. task = (condition, seed, n_updates, out_dir, quick)."""
    condition, seed, n_updates, out_dir, quick = task
    setup = dict(CONDITIONS[condition])
    memoryless = setup.get("memoryless", False)
    world_setup = dict(WORLD, **setup.pop("world", {}))
    torch.set_num_threads(1)          # brains are trained side by side, one CPU core each
    torch.manual_seed(seed)           # fixes the initial weights (the same for every condition)

    world = LivelyWorld(**world_setup)
    brain = Brain(world.sense_shapes, world.n_actions, **BRAIN_SIZE)
    untrained = copy.deepcopy(brain)

    for folder in ("runs", "checkpoints", "weights"):
        os.makedirs(os.path.join(out_dir, folder), exist_ok=True)
    name = f"{condition}_seed{seed}"
    curve_path = os.path.join(out_dir, "runs", f"{name}.growing.json")
    growing = json.load(open(curve_path)) if os.path.exists(curve_path) else []

    def look_in(update, brain):
        """Called now and then during training: a quick round of the tests."""
        if any(point["update"] == update for point in growing):
            return
        quick_tests = commonsense_tests(brain, n_scenes=40 if quick else 100, seed=TEST_SEED + seed,
                                        memoryless=memoryless)
        growing.append({"update": update,
                        "pass_rate": {k: v["trained"]["pass_rate"] for k, v in quick_tests.items()},
                        "pair_rates": {k: v["trained"]["pair_rates"] for k, v in quick_tests.items()}})
        with open(curve_path, "w") as f:
            json.dump(growing, f)

    config = TrainConfig(seed=seed, movement="random", **dict(TRAINING, n_updates=n_updates, **{
        "memoryless": memoryless, "log_every": 50 if quick else TRAINING["log_every"]}))
    history = train(brain, world, config, tag=f"[{condition} seed {seed}] ", checkpoint=look_in,
                    save_path=os.path.join(out_dir, "weights", f"{name}.in_progress"))
    brain.eval()
    torch.save(brain.state_dict(), os.path.join(out_dir, "checkpoints", f"{name}.pt"))

    result = {
        "condition": condition, "seed": seed, "train_seconds": history.pop("seconds")[0],
        "moments_lived": config.n_updates * config.chunk_len, "chunk_len": config.chunk_len,
        "maps_lived_in": world.maps_made,
        "history": {k: [float(np.mean(v[i:i + 10])) for i in range(0, len(v), 10)] for k, v in history.items()},
        "growing": growing,
        "tests": commonsense_tests(brain, untrained, n_scenes=60 if quick else 300, seed=TEST_SEED + seed,
                                   memoryless=memoryless),
    }
    with open(os.path.join(out_dir, "runs", f"{name}.json"), "w") as f:
        json.dump(result, f)
    os.remove(os.path.join(out_dir, "weights", f"{name}.in_progress"))
    print(f"  [{condition} seed {seed}] finished", flush=True)
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def load_runs(out_dir, conditions=None, seeds=None):
    """{condition: [result per seed]} for every finished run found on disk."""
    runs = {}
    for condition in conditions or CONDITIONS:
        for seed in seeds or range(100):
            path = os.path.join(out_dir, "runs", f"{condition}_seed{seed}.json")
            if os.path.exists(path):
                runs.setdefault(condition, []).append(json.load(open(path)))
            elif seeds is None:
                break
    return runs


def _cell(rates, ratios):
    """'pass rate  x ratio', e.g. '87% x1.4'; with several seeds the per-seed rates follow in brackets."""
    rates, ratios = np.asarray(rates, dtype=float), np.asarray(ratios, dtype=float)
    text = f"{100 * rates.mean():.0f}% x{min(ratios.mean(), 99):.2f}"
    if len(rates) > 1:
        text += " [" + " ".join(f"{100 * v:.0f}" for v in rates) + "]"
    return text


def build_report(runs):
    """The full report as a list of lines, and the short summary as another."""
    full, short = [], []
    line = "=" * 110
    first = next(iter(runs.values()))
    seeds = [r["seed"] for r in first]
    full += [line, f"BU-EAI v0.3 RESULTS   (seeds {seeds}; each cell = mean [one value per seed])", line,
             f"Each brain lived {first[0]['moments_lived']:,} moments per agent across about "
             f"{first[0]['maps_lived_in']:,} maps."]
    for condition, rs in runs.items():
        full.append(f"  {condition:<12} {len(rs)} seed(s), {np.mean([r['train_seconds'] for r in rs]) / 60:5.1f} min training each")

    tests = list(first[0]["tests"])
    columns = [(c, "trained") for c in runs] + [(next(iter(runs)), "untrained"), (next(iter(runs)), "no change")]
    names = [c if who == "trained" else {"untrained": "untrained", "no change": "expect no change"}[who]
             for c, who in columns]

    def table(title, row_names, get, out):
        out += ["", title, " " * 24 + "".join(f"{n:<26}" for n in names)]
        for row in row_names:
            out.append(f"  {row:<22}" + "".join(f"{get(row, c, who):<26}" for c, who in columns))

    title = ("COMMON-SENSE TESTS.  Each cell: share of scenes where the impossible versions were more surprising "
             "(chance = 50%),\nthen 'x' how many times more surprising they were on average (x1.00 = no difference).")
    for out in (full, short):
        table(title, tests, lambda t, c, who: _cell([r["tests"][t][who]["pass_rate"] for r in runs[c]],
                                                    [r["tests"][t][who]["ratio"] for r in runs[c]]), out)
    short += ["", "(Full tables: report.txt.)"]

    full += ["", "--- MATCHED PAIRS: impossible version vs the possible version that ends the same way ---"]
    for t in tests:
        info = first[0]["tests"][t]
        full += ["", f"{t}  ({info['principle']}: {info['question']})"]
        full.append(" " * 4 + f"{'':<62}" + "".join(f"{n:<26}" for n in names))
        for i, (impossible, possible) in enumerate(info["pairs"]):
            label = f"'{impossible}' vs '{possible}'"
            full.append(f"    {label:<62}" + "".join(
                f"{_cell([r['tests'][t][who]['pair_rates'][i] for r in runs[c]], [r['tests'][t][who]['pair_ratios'][i] for r in runs[c]]):<26}"
                for c, who in columns))
        full.append("    mean surprise in the window, per version (x 1000):")
        for v, label in enumerate(info["versions"]):
            mark = "possible  " if info["possible"][v] else "IMPOSSIBLE"
            full.append(f"      {mark} {label:<50}" + "".join(
                f"{1000 * np.mean([r['tests'][t][who]['surprise'][v] for r in runs[c]]):<26.2f}" for c, who in columns))
    full.append(line)
    return full, short


def main():
    parser = argparse.ArgumentParser(description="BU-EAI v0.3 experiment")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))   # (list is read when main() runs)
    parser.add_argument("--updates", type=int, default=TRAINING["n_updates"])
    parser.add_argument("--jobs", type=int, default=None, help="brains to train at once (default: one per CPU core)")
    parser.add_argument("--quick", action="store_true", help="1 seed, very short training (smoke test)")
    parser.add_argument("--resume", action="store_true", help="skip brains that are already finished")
    parser.add_argument("--report-only", action="store_true", help="no training; report and plots from saved runs")
    parser.add_argument("--out", default="results/v0.3")
    args = parser.parse_args()
    if args.quick:
        args.seeds, args.updates = [0], 100
        args.out = args.out + "_quick" if args.out == "results/v0.3" else args.out

    if not args.report_only:
        tasks = [(c, s, args.updates, args.out, args.quick) for s in args.seeds for c in args.conditions]
        if args.resume:
            done = lambda c, s: os.path.exists(os.path.join(args.out, "runs", f"{c}_seed{s}.json"))
            tasks = [t for t in tasks if not done(t[0], t[1])]
        jobs = args.jobs or min(len(tasks), os.cpu_count() or 1)
        print(f"BU-EAI v0.3: training {len(tasks)} brain(s), {jobs} at a time, {args.updates} updates each.", flush=True)
        if jobs > 1:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                list(pool.map(run_one, tasks))
        else:
            for task in tasks:
                run_one(task)

    runs = load_runs(args.out, args.conditions)
    if not runs:
        print("No finished runs found.")
        return
    full, short = build_report(runs)
    with open(os.path.join(args.out, "report.txt"), "w") as f:
        f.write("\n".join(full) + "\n")
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write("\n".join(short) + "\n")
    try:
        from bueai import plots_v03
        written = plots_v03.make_all(runs, args.out)
    except ImportError:
        written = []
    print("\n".join(short))
    print(f"\nSaved to {args.out}/: summary.txt, report.txt" + "".join(f", {w}" for w in written))


if __name__ == "__main__":      # required so the worker processes can import this file safely
    main()
