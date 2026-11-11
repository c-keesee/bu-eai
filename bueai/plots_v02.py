"""
plots_v02.py -- the v0.2 figures. Drawing only; nothing here affects results.

`runs` is {condition: [result per seed]} as loaded by run_v02.load_runs().

Colour code:
    blue / orange / green   trained core / untrained core / raw senses (as in v0.1)
    grey / violet / yellow  movement style: random / learning progress / plain error
    blue / pink             top-down feedback on / cut
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from .plots import BLUE, GREEN, GREY, INK_SOFT, ORANGE, SOURCE_COLORS, _legend_above

VIOLET, YELLOW, PINK, DARK_GREY = "#6250d6", "#eda100", "#e87ba4", "#6f6e69"
STYLE_COLORS = {"random": DARK_GREY, "progress": VIOLET, "error": YELLOW}
STYLE_NAMES = {"random": "random walk", "progress": "learning progress", "error": "plain error"}
TARGET_TITLES = {"position": "Where am I?", "bell": "Where is the bell?"}


def _rows(runs, get):
    return np.array([get(r) for r in runs], dtype=float)


def _smooth(rows, window=5):
    """Light moving average along time, so the curves are readable."""
    kernel = np.ones(window) / window
    return np.array([np.convolve(np.pad(r, (window // 2, window // 2), mode="edge"), kernel, mode="valid")
                     for r in rows])


def _line(ax, rows, color, label, smooth=5, **style):
    rows = _smooth(rows, smooth) if smooth > 1 else rows
    x = np.arange(1, rows.shape[1] + 1)
    for row in rows:
        ax.plot(x, row, color=color, linewidth=0.8, alpha=0.28)
    ax.plot(x, rows.mean(axis=0), color=color, linewidth=2, label=label, **style)


def _chance(ax, level, x):
    ax.axhline(level, color=GREY, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(x, level, "chance", ha="right", va="bottom", fontsize=8.5, color=INK_SOFT)


def _finish(fig, title, runs_list, path, top=0.93):
    n = len(runs_list)
    seeds = f"mean of {n} seeds; faint lines = single seeds" if n > 1 else "1 seed"
    fig.suptitle(f"{title}   ({seeds})", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------

def plot_new_map_learning(runs, path):
    """THE key figure: probe error against moments since arriving in a never-seen map."""
    main = runs["random"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, target in zip(axes, TARGET_TITLES):
        for source, color in SOURCE_COLORS.items():
            _line(ax, _rows(main, lambda r: r["new_map"]["curves"][source][target]), color, source)
        chance = np.mean([r["new_map"]["curves"]["chance"][target] for r in main])
        _chance(ax, chance, 400)
        ax.set_ylim(0, chance * 1.15)
        ax.set_title(TARGET_TITLES[target], pad=26)
        ax.set_xlabel("moments since arriving in a brand-new map")
        ax.set_ylabel("probe error (grid cells)")
        _legend_above(ax)
    _finish(fig, "Finding its bearings in a place it has never seen: lights on, random movement", main, path)


def plot_darkroom(runs, path):
    """Dark room in a new place, after a long look around vs almost none."""
    main = runs["random"]
    senses = ["touch", "hearing", "vision"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    ax = axes[0]
    for lit, style in (("300", {}), ("8", {"linestyle": (0, (2, 1.5))})):
        _line(ax, _rows(main, lambda r: r["darkroom"][lit]["position"]["trained core"]), BLUE,
              f"trained, explored {lit}", **style)
    _line(ax, _rows(main, lambda r: r["darkroom"]["300"]["position"]["untrained core"]), ORANGE, "untrained, explored 300")
    _line(ax, _rows(main, lambda r: r["darkroom"]["300"]["position"]["raw senses"]), GREEN, "raw senses")
    chance = np.mean([r["darkroom"]["300"]["position"]["chance"] for r in main])
    _chance(ax, chance, 100)
    ax.set_ylim(0, chance * 1.15)
    ax.set_title("Where am I?  (in the dark)", pad=40)
    ax.set_xlabel("moments since the lights went out")
    ax.set_ylabel("probe error (grid cells)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 0.99), ncol=2, handlelength=1.6,
              columnspacing=1.2, borderaxespad=0)

    for ax, sense in zip(axes[1:], senses):
        for lit, style in (("300", {}), ("8", {"linestyle": (0, (2, 1.5))})):
            _line(ax, _rows(main, lambda r: r["darkroom"][lit]["prediction"][sense]), BLUE,
                  f"explored {lit} moments first", **style)
        for key, label in (("average", "guess the average"), ("no_change", "guess 'no change'")):
            level = np.mean([r["darkroom"]["300"][key][sense] for r in main])
            ax.axhline(level, color=GREY, linestyle=(0, (1, 2)), linewidth=1.2)
            ax.text(100, level, label, ha="right", va="bottom", fontsize=8.5, color=INK_SOFT)
        ax.set_yscale("log")
        ax.set_title(f"Predicting next {sense}" + ("  (imagined)" if sense == "vision" else ""), pad=40)
        ax.set_xlabel("moments since the lights went out")
        ax.set_ylabel("prediction error (log scale)")
        _legend_above(ax)
    _finish(fig, "Dark room in a new place: does looking around first help?", main, path, top=0.96)


def plot_movement(runs, path):
    """Movement styles: each brain moving the way it was trained to move."""
    styles = [s for s in STYLE_COLORS if s in runs]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    ax = axes[0]
    for s in styles:
        _line(ax, _rows(runs[s], lambda r: r["movement"][s]["coverage"]), STYLE_COLORS[s], STYLE_NAMES[s], smooth=1)
    ax.set_ylim(0, 1)
    ax.set_title("Share of the map seen", pad=26)
    ax.set_ylabel("share of walkable cells seen so far")

    ax = axes[1]
    for s in styles:
        _line(ax, _rows(runs[s], lambda r: r["movement"][s]["tv_in_view"]), STYLE_COLORS[s], STYLE_NAMES[s], smooth=15)
    ax.set_ylim(0, None)
    ax.set_title("Noisy-TV check: TV in view", pad=26)
    ax.set_ylabel("share of agents with the TV in view")

    for ax, target in zip(axes[2:], TARGET_TITLES):
        for s in styles:
            _line(ax, _rows(runs[s], lambda r: r["movement"][s][target]), STYLE_COLORS[s], STYLE_NAMES[s])
        ax.set_ylim(0, None)
        ax.set_title(TARGET_TITLES[target], pad=26)
        ax.set_ylabel("probe error (grid cells)")

    for ax in axes:
        ax.set_xlabel("moments since arriving in a brand-new map")
        _legend_above(ax)
    _finish(fig, "Three ways of moving, each brain trained and tested with its own style", runs[styles[0]], path, top=0.96)


def plot_feedback(runs, path):
    """Top-down feedback on vs cut, on the same maps and walks."""
    pairs = [("with feedback", runs["random"], BLUE), ("feedback cut", runs["no_feedback"], PINK)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, target in zip(axes, TARGET_TITLES):
        for label, rs, color in pairs:
            _line(ax, _rows(rs, lambda r: r["new_map"]["curves"]["trained core"][target]), color, label)
        ax.set_ylim(0, None)
        ax.set_title(f"{TARGET_TITLES[target]}  (lights on)", pad=26)
        ax.set_xlabel("moments since arriving in a brand-new map")
        ax.set_ylabel("probe error (grid cells)")
        _legend_above(ax)
    ax = axes[2]
    for label, rs, color in pairs:
        _line(ax, _rows(rs, lambda r: r["darkroom"]["300"]["position"]["trained core"]), color, label)
    ax.set_ylim(0, None)
    ax.set_title("Where am I?  (in the dark, after exploring 300)", pad=26)
    ax.set_xlabel("moments since the lights went out")
    ax.set_ylabel("probe error (grid cells)")
    _legend_above(ax)
    _finish(fig, "Does top-down feedback matter in the harder setting?", runs["random"], path)


def plot_training(runs, path):
    """Prediction error over training, all four conditions."""
    colors = {"random": BLUE, "progress": VIOLET, "error": YELLOW, "no_feedback": PINK}
    names = {"random": "random walk", "progress": "learning progress", "error": "plain error",
             "no_feedback": "random walk, feedback cut"}
    senses = ["vision", "touch", "hearing"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, sense in zip(axes, senses):
        for condition, rs in runs.items():
            rows = _rows(rs, lambda r: r["history"][sense])
            x = np.linspace(0, rs[0]["moments_lived"], rows.shape[1])
            for row in rows:
                ax.plot(x, row, color=colors[condition], linewidth=0.8, alpha=0.28)
            ax.plot(x, rows.mean(axis=0), color=colors[condition], linewidth=2, label=names[condition])
        ax.set_yscale("log")
        ax.set_title(sense, pad=40)
        ax.set_xlabel("moments of life lived (per agent)")
        ax.xaxis.set_major_formatter(lambda value, _: f"{value / 1000:.0f}k")   # 50000 -> "50k"
        ax.set_ylabel("prediction error (log scale)")
        ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 0.99), ncol=2, handlelength=1.6,
                  columnspacing=1.2, borderaxespad=0)
    _finish(fig, "Learning to predict the next sensation, across many maps", next(iter(runs.values())), path, top=0.92)


def make_all(runs, out_dir):
    """Draw every figure the finished runs allow. Returns the file names written."""
    jobs = [("new_map_learning.png", plot_new_map_learning, ("random",)),
            ("darkroom_new_place.png", plot_darkroom, ("random",)),
            ("movement_styles.png", plot_movement, ("random", "progress")),
            ("feedback.png", plot_feedback, ("random", "no_feedback")),
            ("training_loss.png", plot_training, ("random",))]
    written = []
    for name, draw, needs in jobs:
        if all(c in runs for c in needs):
            draw(runs, os.path.join(out_dir, name))
            written.append(name)
    return written
