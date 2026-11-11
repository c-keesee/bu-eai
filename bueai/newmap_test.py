"""
newmap_test.py -- v0.2: drop a frozen brain into places it has never seen.

THE QUESTION: v0.1's brain knew one map by heart. Can this one work out a
NEW place during a single life? Every test here uses freshly generated maps
that never appeared in training, so nothing about them can be in the weights.

Three read-outs, all as a function of "moments since arrival":

    Where am I?        A linear probe reads the agent's (x, y) from the core.
    Where is the bell? A probe reads the BELL's (x, y) in this map. The
                       weights cannot know this. If the core holds it, the
                       brain has stored a fact about this particular place.
    How much have I seen?   Share of the map that has been in view (this one
                       comes from the world, not the brain; it measures how
                       well the movement style explores).

WHAT "POSITION" MEANS WHEN EVERY MAP IS DIFFERENT
(x, y) is measured inside the arena, whose outer wall has the same size in
every map. So position is knowable, but not at a glance: the agent has to
find the walls and keep track of its own movement from there.

ONE PROBE FOR ALL MAPS. The probe is fitted on one set of new maps and scored
on a different set of new maps. It is never fitted on the maps it is scored
on, so it cannot memorise them; it can only succeed if the brain uses the
same code for position in every place.

The baselines are the same as in v0.1: the same network untrained, the raw
senses, and chance (always guess the average).
"""

import torch

from .many_worlds import ManyWorlds
from .training import Life

TARGETS = ("position", "bell", "fountain")      # each is an (x, y) pair


@torch.no_grad()        # frozen brain: no learning in any test
def explore(brain, seed, movement="random", n_agents=256, n_steps=400,
            lights_out_at=None, blind=("vision",), size=16):
    """Let n_agents copies of a frozen brain each live one life in its own new map.

    seed:          decides the maps and the walks. Different seed, different maps.
    lights_out_at: if set, the senses in `blind` are blanked from this moment on.

    Returns a dict of tensors with leading shape (n_steps, n_agents), plus
    "fingerprints": the set of maps used (to check they are new).
    """
    world = ManyWorlds(size=size)
    life = Life(brain, world, n_agents, seed=seed, movement=movement)
    if lights_out_at is None:
        parts = [life.live(n_steps, record=True)]
    else:
        parts = [life.live(lights_out_at, record=True)]
        for name in blind:
            life.set_sense(name, False)
        parts.append(life.live(n_steps - lights_out_at, record=True))

    join = lambda get: torch.cat([get(p) for p in parts])
    out = {
        "core": join(lambda p: p["core"]),
        "senses": join(lambda p: p["senses"]),
        "error": {name: join(lambda p: p["error"][name]) for name in brain.sense_shapes},
        "outcome": {name: join(lambda p: p["outcome"][name]) for name in brain.sense_shapes},
        "fingerprints": world.map_fingerprints,
    }
    for key in parts[0]["truth"]:
        out[key] = join(lambda p: p["truth"][key])
    return out


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

class LinearProbe:
    """Least-squares linear read-out, as in v0.1's emergence_test.py.

    Fitted on a random sample of the recorded moments to keep memory use low.
    """

    def __init__(self, X, Y, ridge=1.0, max_rows=40_000):
        if len(X) > max_rows:
            pick = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))[:max_rows]
            X, Y = X[pick], Y[pick]
        X, Y = X.double(), Y.double()
        self.mean, self.std = X.mean(dim=0), X.std(dim=0) + 1e-6
        Z = self._prepare(X)
        A = Z.T @ Z + ridge * torch.eye(Z.shape[1], dtype=torch.float64)
        self.W = torch.linalg.solve(A, Z.T @ Y)

    def _prepare(self, X):
        Z = (X.double() - self.mean) / self.std
        return torch.cat([Z, torch.ones(len(Z), 1, dtype=torch.float64)], dim=1)

    def __call__(self, X):
        return self._prepare(X) @ self.W


def probe_curves(fit, held_out, features, fit_from=0):
    """Fit one probe per target on `fit`, score it on `held_out` moment by moment.

    features: "core" or "senses".
    fit_from: only moments from this one on are used for fitting.
    Returns dict target -> list (one entry per moment) of the mean distance, in
    grid cells, between the probe's guess and the truth.
    """
    X = fit[features][fit_from:].flatten(0, 1)
    Y = torch.cat([fit[t][fit_from:].flatten(0, 1) for t in TARGETS], dim=1)     # (rows, 6)
    probe = LinearProbe(X, Y)

    curves = {t: [] for t in TARGETS}
    for step in range(held_out[features].shape[0]):                 # one moment at a time (memory)
        guess = probe(held_out[features][step])
        for i, t in enumerate(TARGETS):
            distance = (guess[:, 2 * i:2 * i + 2] - held_out[t][step]).norm(dim=1)
            curves[t].append(distance.mean().item())
    return curves


def chance_curves(fit, held_out):
    """Knowing nothing: always guess the average (taken from the fitting maps)."""
    curves = {}
    for t in TARGETS:
        average = fit[t].flatten(0, 1).mean(dim=0)
        distance = (held_out[t] - average).norm(dim=2)              # (steps, agents)
        curves[t] = distance.mean(dim=1).tolist()
    return curves


# ---------------------------------------------------------------------------
# Test 1: finding your bearings in a new place (lights on)
# ---------------------------------------------------------------------------

FIT_SEED, HELD_OUT_SEED = 10_000, 20_000      # + the run's seed: two separate sets of new maps


def new_map_test(trained, untrained, seed, movement="random", n_agents=256):
    """Position / bell / fountain probe error against moments since arrival.

    Returns {"curves": {source: {target: [error per moment]}},
             "coverage": [share of map seen per moment],
             "tv_in_view": [share of agents with the TV in view per moment],
             "maps": set of map fingerprints used}

    The untrained brain cannot steer, so it always walks at random. When
    `movement` is "random" too, both brains live exactly the same walks.
    """
    curves, maps = {}, set()
    for name, brain, style in [("trained core", trained, movement), ("untrained core", untrained, "random")]:
        fit = explore(brain, FIT_SEED + seed, style, n_agents)
        held_out = explore(brain, HELD_OUT_SEED + seed, style, n_agents)
        curves[name] = probe_curves(fit, held_out, "core")
        maps |= fit["fingerprints"] | held_out["fingerprints"]
        if name == "trained core":
            coverage = held_out["coverage"].mean(dim=1).tolist()
            tv_in_view = held_out["tv_in_view"].float().mean(dim=1).tolist()
    # `fit` / `held_out` now hold the untrained brain's (random) walk.
    curves["raw senses"] = probe_curves(fit, held_out, "senses")
    curves["chance"] = chance_curves(fit, held_out)
    return {"curves": curves, "coverage": coverage, "tv_in_view": tv_in_view, "maps": maps}


# ---------------------------------------------------------------------------
# Test 2: movement styles
# ---------------------------------------------------------------------------

def movement_test(trained, seed, styles=("random", "progress", "error"), n_agents=256):
    """The same frozen brain explores new maps with each movement style.

    Returns {style: {"position": [...], "bell": [...], "coverage": [...],
                     "tv_in_view": [...]}}, each a list with one entry per moment.

    "tv_in_view" is the noisy-TV check: the share of agents that have the TV
    inside their view. An agent stuck on the TV keeps this high.
    """
    results = {}
    for style in styles:
        fit = explore(trained, FIT_SEED + seed, style, n_agents)
        held_out = explore(trained, HELD_OUT_SEED + seed, style, n_agents)
        curves = probe_curves(fit, held_out, "core")
        results[style] = {
            "position": curves["position"],
            "bell": curves["bell"],
            "coverage": held_out["coverage"].mean(dim=1).tolist(),
            "tv_in_view": held_out["tv_in_view"].float().mean(dim=1).tolist(),
        }
    return results


# ---------------------------------------------------------------------------
# Test 3: dark room in a new place
# ---------------------------------------------------------------------------

def new_place_darkroom(trained, untrained, seed, explore_for=(8, 300), dark_steps=100, n_agents=256):
    """Explore a new map with the lights on, then the lights go out.

    Run twice: after a real look around (300 moments) and after almost none
    (8 moments). In v0.1 the layout was in the weights, so exploring first
    would not have mattered. Here the weights know nothing about this map:
    whatever the long explorer can do in the dark that the short one cannot,
    it must have learned by looking around during this life.

    Returns {"8": {...}, "300": {...}}, each with
        "position":   {source: [probe error per dark moment]}
        "prediction": {sense: [prediction error per dark moment]}   (trained brain)
        "no_change", "average": {sense: yardstick}   (brain-free, as in v0.1)
    """
    results = {}
    for lit in explore_for:
        total = lit + dark_steps
        entry = {"position": {}}
        for name, brain in [("trained core", trained), ("untrained core", untrained)]:
            fit = explore(brain, FIT_SEED + seed, n_agents=n_agents, n_steps=total, lights_out_at=lit)
            held_out = explore(brain, HELD_OUT_SEED + seed, n_agents=n_agents, n_steps=total, lights_out_at=lit)
            dark = {k: (v[lit:] if torch.is_tensor(v) else v) for k, v in held_out.items()}
            entry["position"][name] = probe_curves(fit, dark, "core", fit_from=lit)["position"]
            if name == "trained core":
                entry["prediction"] = {s: held_out["error"][s][lit:].mean(dim=1).tolist()
                                       for s in trained.sense_shapes}
        entry["position"]["raw senses"] = probe_curves(fit, dark, "senses", fit_from=lit)["position"]
        entry["position"]["chance"] = chance_curves(
            {t: fit[t][lit:] for t in TARGETS}, {t: dark[t] for t in TARGETS})["position"]

        entry["no_change"], entry["average"] = {}, {}
        for s in trained.sense_shapes:
            outcome = held_out["outcome"][s]
            now, before = outcome[lit:], outcome[lit - 1:-1]
            entry["no_change"][s] = ((now - before) ** 2).mean().item()
            entry["average"][s] = ((now - now.mean(dim=(0, 1), keepdim=True)) ** 2).mean().item()
        results[str(lit)] = entry
    return results
