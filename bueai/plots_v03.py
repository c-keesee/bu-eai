"""
plots_v03.py -- the v0.3 figures. Drawing only; nothing here affects results.

`runs` is {condition: [result per seed]} as loaded by run_v03.load_runs().

Colour code:
    one colour per kind of brain (see CONDITION_COLORS); blue is the plain v0.2 brain
    orange        the same network before training
    grey diamond  "expect no change", the brain-free yardstick
    in the surprise traces: dark grey = possible versions, violet = impossible versions
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from .plots import BLUE, GREEN, GREY, INK_SOFT, ORANGE

VIOLET, YELLOW, PINK, DARK_GREY = "#6250d6", "#eda100", "#e87ba4", "#6f6e69"
CONDITION_COLORS = {"core": BLUE, "memoryless": PINK, "pays": VIOLET, "memory": GREEN, "both": YELLOW}
CONDITION_NAMES = {"core": "core only", "memoryless": "memoryless (check)", "pays": "remembering pays",
                   "memory": "external memory", "both": "memory + pays"}
TEST_NAMES = {
    "permanence_screen_1": "Permanence: under a screen, 1 moment",
    "permanence_screen_3": "Permanence: under a screen, 3 moments",
    "permanence_dark_5": "Permanence: lights out, 5 moments",
    "permanence_dark_30": "Permanence: lights out, 30 moments",
    "solidity_roller": "Solidity: roller meets wall",
    "solidity_ball": "Solidity: ball meets wall (rare event)",
    "causality_watch": "Causality: watching a collision",
    "causality_act": "Causality: pushing it myself",
    "self_starting": "Causality: ball vs roller setting off",
    "continuity_jump": "Continuity: jump in plain view",
    "continuity_screens": "Continuity: two screens and a gap",
}


def _seeds_text(runs_list):
    n = len(runs_list)
    return f"mean of {n} seeds; small dots = single seeds" if n > 1 else "1 seed"


def _save(fig, title, path, top=0.93):
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scores(runs, path):
    """The headline figure: pass rate on every test, every kind of brain, with baselines."""
    first = next(iter(runs.values()))
    tests = list(first[0]["tests"])
    rows = np.arange(len(tests))[::-1]                    # first test at the top
    fig, ax = plt.subplots(figsize=(10.5, 0.52 * len(tests) + 2.2))

    def dots(values_per_test, color, label, offset, marker="o", filled=True):
        values = 100 * np.array(values_per_test, dtype=float)             # (tests, seeds)
        face = color if filled else "none"
        if values.shape[1] > 1:
            for column in values.T:
                ax.scatter(column, rows + offset, s=12, color=color, alpha=0.35, linewidths=0)
        ax.scatter(values.mean(axis=1), rows + offset, s=70, marker=marker, facecolors=face,
                   edgecolors=color, linewidths=1.6, label=label, zorder=3)

    n = len(runs) + 2
    offsets = np.linspace(0.3, -0.3, n)
    for (condition, rs), offset in zip(runs.items(), offsets):
        dots([[r["tests"][t]["trained"]["pass_rate"] for r in rs] for t in tests],
             CONDITION_COLORS.get(condition, DARK_GREY), CONDITION_NAMES.get(condition, condition), offset)
    dots([[r["tests"][t]["untrained"]["pass_rate"] for r in first] for t in tests],
         ORANGE, "untrained", offsets[-2])
    dots([[r["tests"][t]["no change"]["pass_rate"] for r in first] for t in tests],
         GREY, "expect no change", offsets[-1], marker="D", filled=False)

    ax.axvline(50, color=GREY, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(51, -0.55, "chance", ha="left", va="bottom", fontsize=8.5, color=INK_SOFT)
    for boundary in rows[:-1] - 0.5:
        ax.axhline(boundary, color="#e4e3df", linewidth=0.8)
    ax.set_yticks(rows, [TEST_NAMES.get(t, t) for t in tests])
    ax.set_ylim(-0.6, len(tests) - 0.2)
    ax.set_xlim(-3, 103)
    ax.set_xticks(range(0, 101, 25), [f"{v}%" for v in range(0, 101, 25)])
    ax.set_xlabel("share of scenes where the impossible versions were more surprising")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=min(n, 5), handletextpad=0.3,
              columnspacing=1.4, borderaxespad=0)
    _save(fig, f"Common-sense tests: is the impossible more surprising?   ({_seeds_text(first)})", path,
          top=0.95)


def plot_traces(runs, path, condition=None):
    """Surprise over time in each test, for one kind of brain: possible vs impossible versions."""
    condition = condition or next(iter(runs))
    rs = runs[condition]
    tests = list(rs[0]["tests"])
    fig, axes = plt.subplots(3, 4, figsize=(16, 14))
    for ax in axes.flatten()[len(tests):]:
        ax.set_visible(False)
    styles = ["-", (0, (4, 2)), (0, (1, 1.5))]
    for ax, t in zip(axes.flatten(), tests):
        info = rs[0]["tests"][t]
        course = 1000 * np.mean([r["tests"][t]["trained"]["course"] for r in rs], axis=0)   # (versions, moments)
        first, last = info["window"]
        observations = np.arange(1, course.shape[1] + 1)
        # In the lights-out tests the moments before the window are dark: nothing to show.
        show = observations >= first - (0 if "dark" in t else 4)
        used = {True: 0, False: 0}
        for v, label in enumerate(info["versions"]):
            possible = info["possible"][v]
            ax.plot(observations[show] - first, course[v][show], color=DARK_GREY if possible else VIOLET,
                    linestyle=styles[used[possible]], linewidth=2 if not possible else 1.6,
                    label=("possible: " if possible else "IMPOSSIBLE: ") + label)
            used[possible] += 1
        ax.axvspan(-0.5, last - first + 0.5, color=GREY, alpha=0.12, linewidth=0)
        ax.set_title(TEST_NAMES.get(t, t))
        ax.set_xlabel("moments from the start of the event")
        ax.set_ylabel("surprise (prediction error x 1000)")
        ax.set_ylim(0, None)
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.2), fontsize=8, handlelength=2.6, borderaxespad=0)
    n = len(rs)
    _save(fig, f"The surprise meter during each test: {CONDITION_NAMES.get(condition, condition)} brain   "
               f"({'mean of ' + str(n) + ' seeds' if n > 1 else '1 seed'}; shaded = the moments that are scored)",
          path, top=0.95)


def plot_growing_up(runs, path):
    """Pass rate on each test against how long the brain has lived."""
    first = next(iter(runs.values()))
    tests = list(first[0]["tests"])
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    for ax in axes.flatten()[len(tests):]:
        ax.set_visible(False)
    for ax, t in zip(axes.flatten(), tests):
        for condition, rs in runs.items():
            color = CONDITION_COLORS.get(condition, DARK_GREY)
            x = np.array([p["update"] for p in rs[0]["growing"]]) * rs[0]["chunk_len"] / 1000
            rows = 100 * np.array([[p["pass_rate"][t] for p in r["growing"]] for r in rs])
            if len(rows) > 1:
                for row in rows:
                    ax.plot(x, row, color=color, linewidth=0.8, alpha=0.3)
            ax.plot(x, rows.mean(axis=0), color=color, linewidth=2, label=CONDITION_NAMES.get(condition, condition))
        ax.axhline(50, color=GREY, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.set_ylim(-3, 103)
        ax.set_yticks(range(0, 101, 25), [f"{v}%" for v in range(0, 101, 25)])
        ax.set_title(TEST_NAMES.get(t, t), pad=24)
        ax.set_xlabel("moments of life lived (thousands, per agent)")
        ax.set_ylabel("scenes passed")
        ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 0.99), ncol=3, handlelength=1.6,
                  columnspacing=1.2, borderaxespad=0, fontsize=8.5)
    n = len(first)
    _save(fig, f"Growing up: when does each piece of common sense appear?   "
               f"({'mean of ' + str(n) + ' seeds; faint lines = single seeds' if n > 1 else '1 seed'}; dashed = chance)",
          path, top=0.94)


def make_all(runs, out_dir):
    """Draw every figure. Returns the file names written."""
    jobs = [("commonsense_scores.png", plot_scores), ("surprise_traces.png", plot_traces),
            ("growing_up.png", plot_growing_up)]
    for name, draw in jobs:
        draw(runs, os.path.join(out_dir, name))
    return [name for name, _ in jobs]
