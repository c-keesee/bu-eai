"""
emergence_test.py -- did the core learn things nobody told it?

THE QUESTION: the brain was only ever trained to predict its next sensation.
Nobody gave it coordinates or object names. Does its core state nevertheless
contain "I am at (x, y)" or "the nearest thing is a fountain"?

THE TOOL: a linear probe. Freeze the brain, record its core state at many
moments together with the true answer (from world.ground_truth()), then fit
the simplest possible read-out:

        answer  ~=  weights * core_state + bias

A linear probe is deliberately weak. It can only add up the numbers that are
already there. If such a weak read-out finds the answer, the information must
be sitting in the core state in a plain, ready-to-use form.

THE COMPARISONS (a probe result means nothing by itself):
    trained core     the brain after its life of experience
    untrained core   the same architecture with random weights. A random
                     recurrent network still mixes its inputs over time, so
                     this is a stronger baseline than it sounds.
    raw senses       the probe reads the current sense values directly,
                     no brain at all. Shows how much is given away for free
                     (e.g. loudness partly tells you where you are).
    chance           always guess the average position / the most common object.

HELD-OUT DATA: each probe is fitted on one stretch of life and scored on a
separate stretch (different starting cells, different walk) that it has never
seen. Otherwise a probe could score well just by memorising its fitting data.

BLIND MODE: with blind=("vision",) the agent lives the recorded stretch in
darkness (after a short lit warm-up). Scoring the position probe there asks
the strongest version of the question: does the core still know where it is
when it cannot see?
"""

import torch

from .training import Life


# ---------------------------------------------------------------------------
# Recording a stretch of life
# ---------------------------------------------------------------------------

@torch.no_grad()        # no learning here: the brain is frozen
def gather(brain, world, seed, blind=(), n_agents=128, warmup=64, n_steps=100):
    """Let a frozen brain live a fresh stretch of life and record it.

    warmup:  moments lived (all senses on) before recording starts, so the
             core has had time to orient itself.
    blind:   senses to switch off for the recorded part, e.g. ("vision",).

    Returns a dict of tensors, each with leading shape (n_steps, n_agents):
        "core"      the core states
        "senses"    the sense values as delivered to the brain
        "position"  true (x, y)                       <- ground truth
        "object"    true nearest object type (index)  <- ground truth
    """
    life = Life(brain, world, n_agents, seed=seed)
    life.live(warmup)
    for name in blind:
        life.set_sense(name, False)
    recording = life.live(n_steps, record=True)
    return {
        "core": recording["core"],
        "senses": recording["senses"],
        "position": recording["truth"]["position"],
        "object": recording["truth"]["nearest_object"],
    }


# ---------------------------------------------------------------------------
# The linear probe
# ---------------------------------------------------------------------------

class LinearProbe:
    """Least-squares linear read-out:  Y ~= X @ W  (with a bias column).

    This is plain linear regression, solved exactly in one step (no gradient
    descent needed). `ridge` adds a small penalty on large weights, which
    keeps the solution stable when some inputs are nearly duplicates.
    """

    def __init__(self, X, Y, ridge=1.0):
        X, Y = X.double(), Y.double()
        # Put every input feature on the same scale first.
        self.mean, self.std = X.mean(dim=0), X.std(dim=0) + 1e-6
        Z = self._prepare(X)
        A = Z.T @ Z + ridge * torch.eye(Z.shape[1], dtype=torch.float64)
        self.W = torch.linalg.solve(A, Z.T @ Y)

    def _prepare(self, X):
        Z = (X.double() - self.mean) / self.std
        ones = torch.ones(len(Z), 1, dtype=torch.float64)      # the bias column
        return torch.cat([Z, ones], dim=1)

    def __call__(self, X):
        return self._prepare(X) @ self.W


def score_probes(fit, held_out, features):
    """Fit probes on `fit`, score them on `held_out`.

    features: which recorded tensor the probe reads: "core" or "senses".
    Returns:
        position_error          mean distance (in grid cells) between the
                                probe's guess and the true position
        position_error_by_step  the same, separately for each recorded moment
        object_accuracy         fraction of moments where the probe names the
                                nearest object type correctly
    """
    n_steps = held_out[features].shape[0]
    X_fit = fit[features].flatten(0, 1)             # (steps, agents, d) -> (steps*agents, d)
    X_test = held_out[features].flatten(0, 1)

    # --- position: predict the two numbers (x, y) -------------------------
    probe = LinearProbe(X_fit, fit["position"].flatten(0, 1))
    guess = probe(X_test)
    distance = (guess - held_out["position"].flatten(0, 1)).norm(dim=1)

    # --- nearest object: predict a one-hot vector, take the biggest entry --
    n_types = int(fit["object"].max()) + 1
    one_hot = torch.nn.functional.one_hot(fit["object"].flatten(), n_types)
    probe = LinearProbe(X_fit, one_hot)
    correct = probe(X_test).argmax(dim=1) == held_out["object"].flatten()

    return {
        "position_error": distance.mean().item(),
        "position_error_by_step": distance.reshape(n_steps, -1).mean(dim=1).tolist(),
        "object_accuracy": correct.double().mean().item(),
    }


def chance_scores(fit, held_out):
    """What you get by knowing nothing: always guess the average position and
    the most common object (both taken from the fitting data)."""
    mean_position = fit["position"].flatten(0, 1).mean(dim=0)
    distance = (held_out["position"].flatten(0, 1) - mean_position).norm(dim=1)
    most_common = fit["object"].flatten().mode().values
    correct = held_out["object"].flatten() == most_common
    return {
        "position_error": distance.mean().item(),
        "position_error_by_step": [distance.mean().item()] * held_out["position"].shape[0],
        "object_accuracy": correct.double().mean().item(),
    }


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

def emergence_test(trained, untrained, world, seed, blind=()):
    """Run all probes. Returns dict: source name -> scores (see score_probes).

    The trained and untrained brains live the *same* walks (same seeds), so
    any difference between them is down to what is in the weights.
    """
    fit_seed, held_out_seed = 10_000 + seed, 20_000 + seed     # two separate stretches of life

    results = {}
    for name, brain in [("trained core", trained), ("untrained core", untrained)]:
        fit = gather(brain, world, fit_seed, blind)
        held_out = gather(brain, world, held_out_seed, blind)
        results[name] = score_probes(fit, held_out, "core")

    # `fit` / `held_out` still hold the last recording. The walks and sense
    # values are identical for both brains, so we can reuse them here.
    results["raw senses"] = score_probes(fit, held_out, "senses")
    results["chance"] = chance_scores(fit, held_out)
    return results
