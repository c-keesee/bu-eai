"""
checkpoints.py -- saving and loading trained brains.

A checkpoint is one file holding the brain's weights AND the recipe needed to
rebuild the brain they belong to, so a saved brain can be tested again later
without training and without remembering how it was configured.

    save_brain(path, brain, info)       info: anything worth knowing later
    brain, info = load_brain(path)

WHERE THE CHECKPOINTS ARE (see also results/CHECKPOINTS.md):
    results/v0.3/checkpoints/          the final v0.3 brains, one per condition and seed
    results/v0.3/checkpoints/pilot/    brains from the v0.3 pilots (including the "watcher")
    results/v0.2/checkpoints/          the twelve v0.2 brains
"""

import torch

from .model import Brain
from .model_v03 import BrainV3

KINDS = {"Brain": Brain, "BrainV3": BrainV3}


def save_brain(path, brain, info=None, **brain_args):
    """Save a brain. brain_args: the keyword arguments it was built with
    (besides sense_shapes and n_actions, which are read from the brain)."""
    torch.save({"kind": type(brain).__name__,
                "sense_shapes": {k: list(v) for k, v in brain.sense_shapes.items()},
                "n_actions": brain.n_actions, "brain_args": brain_args,
                "info": info or {}, "weights": brain.state_dict()}, path)


def load_brain(path):
    """Rebuild a saved brain. Returns (brain in eval mode, info dict)."""
    saved = torch.load(path, weights_only=False)
    shapes = {k: tuple(v) for k, v in saved["sense_shapes"].items()}
    brain = KINDS[saved["kind"]](shapes, saved["n_actions"], **saved["brain_args"])
    brain.load_state_dict(saved["weights"])
    return brain.eval(), saved["info"]
