"""
model.py -- the brain.

The pieces, and what each one loosely stands for:

    SensoryRegion / VisionRegion   a sensory cortex area (one per sense)
    Brain.core  (a GRU)            the association area that integrates the senses
    region.feedback                top-down connections: core -> sensory regions
    Brain.predictors               the brain's guess about its own next sensation

One "moment" of experience runs in two halves:

    perceive():   senses + top-down feedback -> regions -> core   (update the mind)
    predict():    core + chosen action       -> predictors        (guess what comes next)

                   +-----------------------------------------+
                   |      top-down feedback (previous state) |
                   v                                         |
    vision  --> [vision region ] --+                         |
    touch   --> [touch region  ] --+--> [ recurrent core ] --+--> [predictors] --> next vision
    hearing --> [hearing region] --+        ^         |                        --> next touch
                                            |         +-- + action ------------^   next hearing
                              last action --+
                              (efference copy)

IMPORTANT DESIGN RULE: the Brain class never mentions "vision", "touch", "grid"
or any specific sense. It is built from a dictionary {sense name: shape} and
loops over whatever is in there. That is what lets the same brain move to a 3D
world later: hand it different shapes and (optionally) a different vision
region, and nothing else changes.
"""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sensory regions (the encoders)
# ---------------------------------------------------------------------------

class SensoryRegion(nn.Module):
    """A generic sensory region: a small fully-connected network (an MLP).

    Input:  the raw numbers from one sense organ
            + a top-down feedback signal computed from the core's previous state.
    Output: a short "summary" vector (the latent) that is passed to the core.

    Why feedback? In a real cortex, higher areas send as many connections back
    down as they receive. The idea is that what you already believe about the
    world changes how you process what comes in next. Here: the region can
    interpret the same raw input differently depending on what the core knew a
    moment ago.

    ANY replacement region must follow the same contract:
        forward(observation, core_state) -> tensor of shape (batch, latent_size)
        and have an attribute `latent_size`.
    """

    def __init__(self, input_shape, core_size, latent_size=32, feedback_size=16, hidden_size=64):
        super().__init__()
        self.latent_size = latent_size
        n_inputs = math.prod(input_shape)             # e.g. (25,) -> 25

        # Top-down pathway: squeeze the core state (128 numbers) down to a
        # small feedback signal (16 numbers).
        self.feedback = nn.Linear(core_size, feedback_size)

        # Bottom-up pathway: raw input + feedback -> latent.
        self.net = nn.Sequential(
            nn.Linear(n_inputs + feedback_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
            nn.ReLU(),
        )

    def forward(self, observation, core_state):
        top_down = torch.tanh(self.feedback(core_state))
        flat = observation.flatten(start_dim=1)       # (batch, anything...) -> (batch, n_inputs)
        return self.net(torch.cat([flat, top_down], dim=1))


class VisionRegion(nn.Module):
    """A sensory region specialised for images: starts with a convolution.

    A convolution slides a small 3x3 filter over the image, so it detects the
    same local pattern (an edge, a coloured blob) wherever it appears. This is
    the standard building block for vision and is loosely how early visual
    cortex works (small receptive fields, repeated across the visual field).

    THIS IS THE CLASS TO REPLACE FOR 3D. For 64x64 camera images you would
    stack a few more Conv2d layers with stride 2. As long as the replacement
    takes (image, core_state) and returns (batch, latent_size), the rest of
    the brain does not need to know.
    """

    def __init__(self, input_shape, core_size, latent_size=32, feedback_size=16, n_filters=8):
        super().__init__()
        self.latent_size = latent_size
        channels, height, width = input_shape         # e.g. (3, 5, 5)

        self.conv = nn.Conv2d(channels, n_filters, kernel_size=3)   # 5x5 image -> 3x3 feature map
        n_features = n_filters * (height - 2) * (width - 2)

        self.feedback = nn.Linear(core_size, feedback_size)
        self.to_latent = nn.Linear(n_features + feedback_size, latent_size)

    def forward(self, observation, core_state):
        features = torch.relu(self.conv(observation)).flatten(start_dim=1)
        top_down = torch.tanh(self.feedback(core_state))
        return torch.relu(self.to_latent(torch.cat([features, top_down], dim=1)))


def default_region(input_shape, core_size, latent_size):
    """Pick a region type from the shape of the input.

    3 dimensions (channels, height, width) looks like an image -> VisionRegion.
    Anything else -> the generic SensoryRegion.
    """
    if len(input_shape) == 3:
        return VisionRegion(input_shape, core_size, latent_size)
    return SensoryRegion(input_shape, core_size, latent_size)


# ---------------------------------------------------------------------------
# The whole brain
# ---------------------------------------------------------------------------

class Brain(nn.Module):
    def __init__(self, sense_shapes, n_actions, core_size=128, latent_size=32, regions=None,
                 use_feedback=True, predictor_size=128, curiosity_heads=False):
        """
        sense_shapes: dict, sense name -> shape of one observation.
                      Comes straight from world.sense_shapes.
        n_actions:    number of possible actions (world.n_actions).
        core_size:    how many numbers make up the core's state ("the mind").
        regions:      optional dict, sense name -> your own region module, to
                      override the default for that sense (e.g. a bigger CNN).
        use_feedback: set False to cut the top-down connections (the regions
                      then receive zeros instead of the core state). Only for
                      the experiment "does feedback actually matter?".
        predictor_size: width of the hidden layer inside each predictor.
        curiosity_heads: add the two small "how wrong will I be?" networks
                      used for curious movement (v0.2, see curiosity() below).
        """
        super().__init__()
        self.sense_shapes = dict(sense_shapes)
        self.n_actions = n_actions
        self.core_size = core_size
        self.use_feedback = use_feedback
        regions = regions or {}

        # One sensory region per sense. nn.ModuleDict is a normal dict that
        # also tells PyTorch "the things in here have trainable weights".
        self.regions = nn.ModuleDict({
            name: regions.get(name) or default_region(shape, core_size, latent_size)
            for name, shape in self.sense_shapes.items()
        })

        # The recurrent core. "Recurrent" = its output at one moment is fed
        # back in as an input at the next moment, which is what gives it
        # memory. A GRU is a recurrent layer with learned "gates" that decide
        # how much of the old state to keep and how much to overwrite.
        #
        # Its inputs: all the region latents side by side, plus the action the
        # agent just took (the "efference copy": real brains send a copy of
        # each motor command to sensory areas so they know the body moved).
        total_latent = sum(region.latent_size for region in self.regions.values())
        self.core = nn.GRUCell(total_latent + n_actions, core_size)

        # One predictor per sense. Input: the core state + the action about to
        # be taken. Output: a guess at that sense's NEXT observation.
        self.predictors = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(core_size + n_actions, predictor_size),
                nn.ReLU(),
                nn.Linear(predictor_size, math.prod(shape)),
            )
            for name, shape in self.sense_shapes.items()
        })

        # v0.2 -- self-monitoring for curiosity. Two small networks that look
        # at (core state, action) and guess the brain's OWN upcoming error:
        #
        #   first_look:   how wrong will my prediction be if I take this action?
        #   second_look:  ...and how wrong would I STILL be about that same
        #                 place one moment later, after having looked at it?
        #
        # Their difference is "how much would one look teach me" (see
        # curiosity()). They are trained on the brain's own measured errors
        # (training.py) and read a detached copy of the core state, so they
        # cannot change what the core learns: the world model is still
        # trained by sensory prediction error alone.
        self.has_curiosity = curiosity_heads
        if curiosity_heads:
            def error_guesser():
                return nn.Sequential(nn.Linear(core_size + n_actions, 64), nn.ReLU(), nn.Linear(64, 1))
            self.first_look = error_guesser()
            self.second_look = error_guesser()

    def initial_state(self, n_agents):
        """A blank mind: all zeros. Shape (n_agents, core_size)."""
        return torch.zeros(n_agents, self.core_size)

    def perceive(self, observations, last_action, core_state):
        """Take in one moment of sensation and update the core state.

        observations: dict, sense name -> tensor (n_agents, *shape)
        last_action:  one-hot tensor (n_agents, n_actions): the move just made
        core_state:   tensor (n_agents, core_size): the state BEFORE this moment
        returns:      the new core state
        """
        # Each region sees its own sense plus top-down feedback from the
        # core's previous state.
        top_down = core_state if self.use_feedback else torch.zeros_like(core_state)
        latents = [region(observations[name], top_down) for name, region in self.regions.items()]
        core_input = torch.cat(latents + [last_action], dim=1)
        return self.core(core_input, core_state)

    def predict(self, core_state, action):
        """Guess every sense's next observation, given the action about to be taken.

        returns: dict, sense name -> tensor shaped like that sense's observations
        """
        x = torch.cat([core_state, action], dim=1)
        return {
            name: predictor(x).reshape(-1, *self.sense_shapes[name])
            for name, predictor in self.predictors.items()
        }

    # ------------------------------------------------------------------
    # v0.2: curiosity
    # ------------------------------------------------------------------

    def expected_errors(self, core_state, action):
        """The two self-monitoring guesses for (core state, action).

        Returns (first_look, second_look), each of shape (n_agents,).
        softplus keeps them positive (an error cannot be negative). The core
        state is detached: these heads read the mind but do not shape it.
        """
        x = torch.cat([core_state.detach(), action], dim=1)
        softplus = nn.functional.softplus
        return softplus(self.first_look(x)).squeeze(1), softplus(self.second_look(x)).squeeze(1)

    @torch.no_grad()
    def curiosity(self, core_state, style):
        """How interesting is each possible action? Returns (n_agents, n_actions).

        style "error":     plain prediction-error curiosity.
                           score = the error I expect on first look.
                           Known flaw (the "noisy TV problem"): pure noise is
                           always badly predicted, so it scores high forever
                           and the agent gets stuck staring at it.

        style "progress":  expected learning progress.
                           score = expected error on first look
                                 - expected error on a second look
                           An unexplored corner is surprising once and then
                           known: large first, small second -> high score.
                           TV static is just as surprising the second time:
                           the two cancel -> score near zero.
        """
        n_agents = core_state.shape[0]
        # Ask about every action at once: repeat each agent's state n_actions times.
        states = core_state.repeat_interleave(self.n_actions, dim=0)
        actions = torch.eye(self.n_actions).repeat(n_agents, 1)
        first, second = self.expected_errors(states, actions)
        score = first if style == "error" else torch.relu(first - second)
        return score.reshape(n_agents, self.n_actions)
