"""
darkroom_test.py -- lights out: can the brain still predict touch and hearing?

THE QUESTION: if the agent carries an internal map, it should keep predicting
what it will bump into and what it will hear after the lights go out, by
tracking its own movement on that map. If it only ever reacted to what it
currently sees, its predictions should collapse the moment vision is blanked.

THE PROCEDURE: a frozen, trained brain lives the same walk twice.
    "lit"    the lights stay on the whole time (the control)
    "dark"   identical walk, but vision is blanked after `lit_steps` moments
For every moment we record how wrong the touch and hearing predictions were.

BONUS, THE "MIND'S EYE": the vision predictor keeps producing a guess in the
dark. We compare that guess with the true image, which the brain is never
shown. If the guess is good, the brain is in effect imagining the room.

THE YARDSTICKS, both computed from the same walk with no brain involved:
    "no change"   predict that the next sensation equals the current one
    "average"     always predict the average sensation
A brain with no map and no memory cannot do much better than these.

WHAT THIS TEST CANNOT SHOW ALONE: hearing still works in the dark, and
loudness depends on position, so some of the "map" could be re-read from sound
at every step instead of being remembered. The position probe in darkness
(emergence_test with blind=("vision",)) compares the core against exactly
those raw senses, which is what separates the two explanations.
"""

import torch

from .training import Life


@torch.no_grad()
def darkroom_test(brain, world, seed, blind=("vision",), n_agents=256, lit_steps=64, dark_steps=100):
    """Returns dict: sense name -> {
            "lit":  prediction error per moment, lights always on   (list)
            "dark": prediction error per moment, lights out after lit_steps
            "lit_mean", "dark_mean":  averages over the dark phase only
            "no_change", "average":   the two yardsticks (single numbers)
       }
    for every sense, plus "lit_steps" for plotting.
    """
    walk_seed = 30_000 + seed
    total = lit_steps + dark_steps
    results = {"lit_steps": lit_steps}

    # --- control: lights on throughout ------------------------------------
    life = Life(brain, world, n_agents, seed=walk_seed)
    lit = life.live(total, record=True)

    # --- same walk, lights go out part-way --------------------------------
    life = Life(brain, world, n_agents, seed=walk_seed)     # same seed = same walk
    before = life.live(lit_steps)
    for name in blind:
        life.set_sense(name, False)
    after = life.live(dark_steps)

    # Every sense is scored against what it REALLY would have reported, also
    # the blanked one. For vision in the dark that is the brain's "mind's
    # eye": its guess at an image it is not being shown.
    for name in brain.sense_shapes:
        lit_error = lit["error"][name].mean(dim=1)                       # average over agents
        dark_error = torch.cat([before["error"][name], after["error"][name]]).mean(dim=1)

        # Yardsticks, measured on the dark-phase part of the walk.
        outcome = lit["outcome"][name][lit_steps:]                       # (dark_steps, agents, ...)
        previous = lit["outcome"][name][lit_steps - 1:-1]                # the same, one moment earlier
        average = outcome.mean(dim=(0, 1), keepdim=True)

        results[name] = {
            "lit": lit_error.tolist(),
            "dark": dark_error.tolist(),
            "lit_mean": lit_error[lit_steps:].mean().item(),
            "dark_mean": dark_error[lit_steps:].mean().item(),
            "no_change": ((outcome - previous) ** 2).mean().item(),
            "average": ((outcome - average) ** 2).mean().item(),
        }
    return results
