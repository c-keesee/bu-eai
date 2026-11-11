"""
plots.py -- turns the numbers from run_experiment.py into three saved figures.

Nothing in here affects the experiment; it only draws. Every function takes
`runs`: a list with one results dictionary per random seed (see run_seed() in
run_experiment.py for what is inside).

Colour code, the same in every figure:
    blue    the trained brain
    orange  the untrained brain (same network before its life)
    green   raw senses (no brain)
    grey    reference lines (chance, yardsticks, the lights-on control)
"""

import matplotlib

matplotlib.use("Agg")           # draw straight to files; no window needed
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
GREY, GRID, INK, INK_SOFT, SURFACE = "#8a8983", "#e4e3df", "#0b0b0b", "#52514e", "#fcfcfb"
SOURCE_COLORS = {"trained core": BLUE, "untrained core": ORANGE, "raw senses": GREEN}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_SOFT, "text.color": INK,
    "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 9, "font.size": 9.5,
})


def _seed_label(runs):
    n = len(runs)
    return f"mean of {n} seeds" if n > 1 else "1 seed"


def _stack(runs, getter):
    """One row per seed -> 2D array (n_seeds, n_points)."""
    return np.array([getter(run) for run in runs], dtype=float)


def _mean_line(ax, x, rows, color, label, **style):
    """Draw each seed faintly and the mean across seeds on top."""
    for row in rows:
        ax.plot(x, row, color=color, linewidth=0.8, alpha=0.3)
    ax.plot(x, rows.mean(axis=0), color=color, linewidth=2, label=label, **style)


def _legend_above(ax):
    """Legend in one row between the title and the plot, clear of the data."""
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 0.99), ncol=3,
              handlelength=1.6, columnspacing=1.4, borderaxespad=0)


# ---------------------------------------------------------------------------

def plot_training_loss(runs, path):
    """Prediction error over the agent's life, one panel per sense."""
    senses = list(runs[0]["history"])
    chunk = runs[0]["chunk_len"]
    fig, axes = plt.subplots(1, len(senses), figsize=(4 * len(senses), 3.4))

    for ax, sense in zip(axes, senses):
        rows = _stack(runs, lambda run: run["history"][sense])
        # Smooth with a moving average so the trend is readable.
        window = max(1, rows.shape[1] // 60)
        kernel = np.ones(window) / window
        rows = np.array([np.convolve(row, kernel, mode="valid") for row in rows])
        moments = (np.arange(rows.shape[1]) + window) * chunk
        _mean_line(ax, moments, rows, BLUE, None)
        ax.set_yscale("log")
        ax.set_title(f"{sense}")
        ax.set_xlabel("moments of life lived (per agent)")
    axes[0].set_ylabel("prediction error (log scale)")
    fig.suptitle(f"Learning to predict the next sensation   ({_seed_label(runs)}; faint lines = single seeds)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------

def plot_probe_results(runs, path):
    """Bars: how well a linear probe finds position / nearest object, for the
    trained core vs the baselines, under each sensing condition."""
    conditions = list(runs[0]["probes"])
    sources = list(SOURCE_COLORS)
    metrics = [
        ("position_error", "Where am I?", "position error (grid cells)   lower is better"),
        ("object_accuracy", "Which object is nearest?", "accuracy   higher is better"),
    ]
    fig, axes = plt.subplots(len(metrics), len(conditions),
                             figsize=(3.7 * len(conditions), 3.5 * len(metrics)), squeeze=False)

    for r, (metric, question, ylabel) in enumerate(metrics):
        top = 0
        for c, condition in enumerate(conditions):
            ax = axes[r, c]
            for i, source in enumerate(sources):
                values = np.array([run["probes"][condition][source][metric] for run in runs])
                ax.bar(i, values.mean(), width=0.62, color=SOURCE_COLORS[source], zorder=2)
                # one dark dot per seed, so the spread is visible
                ax.scatter(np.full(len(values), i) + np.linspace(-0.12, 0.12, len(values)), values,
                           s=14, color=INK, zorder=3, linewidths=0)
                ax.annotate(f"{values.mean():.2f}", (i, values.max()), xytext=(0, 5),
                            textcoords="offset points", ha="center", va="bottom", fontsize=9, color=INK)
                top = max(top, values.max())
            chance = np.mean([run["probes"][condition]["chance"][metric] for run in runs])
            ax.axhline(chance, color=INK_SOFT, linestyle=(0, (4, 3)), linewidth=1.1, zorder=4)
            # leave empty room right of the last bar for the "chance" label
            ax.set_xlim(-0.5, len(sources) + 0.25)
            ax.text(len(sources) + 0.2, chance, f"chance\n{chance:.2f}", ha="right", va="bottom",
                    fontsize=8.5, color=INK_SOFT, linespacing=1.1)
            top = max(top, chance)
            ax.set_xticks(range(len(sources)), [s.replace(" ", "\n") for s in sources])
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{question}\n{condition}", fontsize=10)
            if c == 0:
                ax.set_ylabel(ylabel)
        for ax in axes[r]:
            ax.set_ylim(0, top * 1.22)

    fig.suptitle(f"Linear probes on held-out life   (bars = {_seed_label(runs)}, dots = single seeds)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------

def plot_darkroom(runs, path):
    """Top row: can a probe still find the agent as the blackout goes on?
    Bottom row: prediction error before and after the lights go out."""
    dark = runs[0]["darkroom"]
    senses = [k for k in dark if k != "lit_steps"]
    lit_steps = dark["lit_steps"]
    blind_conditions = [c for c in runs[0]["probes"] if c != "lights on"]
    n_cols = max(len(senses), len(blind_conditions))
    fig, axes = plt.subplots(2, n_cols, figsize=(4.2 * n_cols, 7.2), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)                      # switch panels on as we use them

    # --- top row: position probe through each blackout ---------------------
    for ax, condition in zip(axes[0], blind_conditions):
        ax.set_visible(True)
        for source, color in SOURCE_COLORS.items():
            rows = _stack(runs, lambda run: run["probes"][condition][source]["position_error_by_step"])
            _mean_line(ax, np.arange(1, rows.shape[1] + 1), rows, color, source)
        chance = np.mean([run["probes"][condition]["chance"]["position_error"] for run in runs])
        ax.axhline(chance, color=GREY, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.text(rows.shape[1], chance, "chance", ha="right", va="bottom", fontsize=8.5, color=INK_SOFT)
        ax.set_ylim(0, chance * 1.15)
        ax.set_title(f"Where am I?  Probe, {condition}", pad=26)
        ax.set_xlabel("moments since the senses were blanked")
        ax.set_ylabel("position error (grid cells)")
        _legend_above(ax)

    # --- bottom row: prediction error around lights-out ---------------------
    for ax, sense in zip(axes[1], senses):
        ax.set_visible(True)
        x = np.arange(len(dark[sense]["lit"])) - lit_steps
        show = x >= -30                                        # skip the warm-up
        lit_rows = _stack(runs, lambda run: run["darkroom"][sense]["lit"])[:, show]
        dark_rows = _stack(runs, lambda run: run["darkroom"][sense]["dark"])[:, show]
        _mean_line(ax, x[show], lit_rows, GREY, "lights stay on")
        _mean_line(ax, x[show], dark_rows, BLUE, "lights go out at 0")
        for key, label in [("average", "guess the average"), ("no_change", "guess 'no change'")]:
            level = np.mean([run["darkroom"][sense][key] for run in runs])
            ax.axhline(level, color=GREY, linestyle=(0, (1, 2)), linewidth=1.2)
            ax.text(x[show][-1], level, label, ha="right", va="bottom", fontsize=8.5, color=INK_SOFT)
        ax.axvline(0, color=INK_SOFT, linewidth=0.8)
        ax.set_yscale("log")       # the errors span a wide range; log keeps all of it readable
        imagined = sense in runs[0]["darkroom_blind"]
        ax.set_title(f"Predicting next {sense}" + ("  (imagined, never shown)" if imagined else ""), pad=26)
        ax.set_xlabel("moments relative to lights-out")
        ax.set_ylabel("prediction error (log scale)")
        _legend_above(ax)

    fig.suptitle(f"Dark room   ({_seed_label(runs)}; faint lines = single seeds)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
