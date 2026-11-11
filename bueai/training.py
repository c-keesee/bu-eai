"""
training.py -- living and learning.

Two things live here:

    Life     the loop of experience: sense -> update the mind -> act ->
             predict -> see what really happened. Used for training AND by
             both tests, so they are guaranteed to run the brain identically.

    train()  lets a brain live for a while and, after every short stretch of
             life, adjusts its weights to make its predictions less wrong.

THE ONLY LEARNING SIGNAL is prediction error: (what I predicted I would sense
next - what I actually sensed next)^2. There is no reward, no label, no goal,
and no language. Nothing here ever reads world.ground_truth().
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


class Life:
    """A stream of experience: n_agents copies of the brain living in a world.

    The agents never reset. `live(n)` advances all of them by n moments and
    then you can call it again to continue from exactly where they stopped.

    A Life takes over the world object it is given (it places its agents in
    it). So do not keep two Lives going at once on the same world object;
    finish with one before starting the next, or give each its own world.
    """

    def __init__(self, brain, world, n_agents, seed=0, persistence=0.6, blackouts=None):
        """
        persistence: chance of repeating the previous action instead of picking
                     a fresh random one (so the walk has some momentum).
        blackouts:   dict, sense name -> (chance per step that a working sense
                     goes dark, chance per step that a dark sense comes back).
                     Think "night falls" for vision. None = senses always work.
        """
        self.brain, self.world = brain, world
        self.n_agents = n_agents
        self.persistence = persistence
        self.blackouts = blackouts or {}
        self.rng = np.random.default_rng(seed)

        self.observations = self._to_torch(world.reset(n_agents, seed=seed))
        self.core_state = brain.initial_state(n_agents)
        self.last_action = np.zeros(n_agents, dtype=np.int64)        # "stay"
        # sense_on[name][i] is False while agent i's sense `name` is blanked.
        self.sense_on = {name: np.ones(n_agents, dtype=bool) for name in brain.sense_shapes}

    # ------------------------------------------------------------------

    def live(self, n_steps, record=False):
        """Live for n_steps moments.

        Returns a dict:
            "error"     sense -> tensor (n_steps, n_agents): prediction error
                        for that sense at each moment (mean squared error)
            "sensed"    sense -> tensor (n_steps, n_agents): 1 if the agent
                        actually got to sense the outcome, 0 if blanked
        and, if record=True, also (for the tests):
            "core"      tensor (n_steps, n_agents, core_size)
            "senses"    tensor (n_steps, n_agents, all sense values flattened),
                        exactly as delivered to the brain (blanked = zeros)
            "outcome"   sense -> tensor (n_steps, n_agents, ...): what each
                        sense really reported one moment later
            "truth"     name -> tensor (n_steps, n_agents, ...)
        """
        names = list(self.brain.sense_shapes)
        core_log, action_log, senses_log, truth_log = [], [], [], []
        outcome = {name: [] for name in names}     # what each sense really reported next
        sensed = {name: [] for name in names}      # ...and whether the agent got to sense it

        for _ in range(n_steps):
            # 1. SENSE. A blanked sense delivers all zeros (darkness/silence).
            inputs = {name: self._blank_if_off(name, self.observations[name]) for name in names}

            # 2. UPDATE THE MIND. Regions encode the senses (with top-down
            #    feedback), the core integrates them with its memory.
            self.core_state = self.brain.perceive(
                inputs, self._one_hot(self.last_action), self.core_state)
            core_log.append(self.core_state)
            if record:
                senses_log.append(torch.cat([inputs[n].flatten(1) for n in names], dim=1))
                truth_log.append(self.world.ground_truth())

            # 3. ACT. A random walk with momentum; the brain has no say (yet).
            action = self._choose_action()
            action_log.append(self._one_hot(action))

            # 4. THE WORLD MOVES ON, and we find out what really happened.
            self.observations = self._to_torch(self.world.step(action))
            self.last_action = action
            self._update_blackouts()
            for name in names:
                outcome[name].append(self.observations[name])
                sensed[name].append(torch.from_numpy(self.sense_on[name].astype(np.float32)))

        # 5. PREDICT, then 6. COMPARE with reality.
        #    Logically each prediction belongs between steps 3 and 4: it uses
        #    only the core state and the action from that moment, before the
        #    outcome is known. Because predictions do not influence the random
        #    walk, we can compute them for all moments in one batch here, which
        #    is noticeably faster than doing it inside the loop.
        core = torch.stack(core_log)                   # (n_steps, n_agents, core_size)
        actions = torch.stack(action_log)              # (n_steps, n_agents, n_actions)
        predictions = self.brain.predict(core.flatten(0, 1), actions.flatten(0, 1))

        error = {}
        for name in names:
            reality = torch.stack(outcome[name]).flatten(0, 1)
            diff = predictions[name] - reality
            # Mean squared error over all the numbers in one observation
            # -> one value per agent per moment.
            error[name] = (diff ** 2).flatten(1).mean(dim=1).reshape(n_steps, self.n_agents)

        out = {
            "error": error,
            "sensed": {n: torch.stack(sensed[n]) for n in names},
        }
        if record:
            out["core"] = core
            out["outcome"] = {n: torch.stack(outcome[n]) for n in names}
            out["senses"] = torch.stack(senses_log)
            out["truth"] = {
                key: torch.from_numpy(np.stack([t[key] for t in truth_log]))
                for key in truth_log[0]
            }
        return out

    def set_sense(self, name, on):
        """Switch a sense on/off for all agents (used by the dark-room test)."""
        self.sense_on[name][:] = on

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _choose_action(self):
        fresh = self.rng.integers(0, self.world.n_actions, size=self.n_agents)
        repeat = self.rng.random(self.n_agents) < self.persistence
        return np.where(repeat, self.last_action, fresh)

    def _update_blackouts(self):
        for name, (p_go_dark, p_come_back) in self.blackouts.items():
            roll = self.rng.random(self.n_agents)
            on = self.sense_on[name]
            # A working sense fails with chance p_go_dark; a dark one returns
            # with chance p_come_back.
            self.sense_on[name] = np.where(on, roll >= p_go_dark, roll < p_come_back)

    def _blank_if_off(self, name, observation):
        on = torch.from_numpy(self.sense_on[name].astype(np.float32))
        # Reshape (n_agents,) -> (n_agents, 1, 1, ...) so it multiplies each
        # agent's whole observation by 1 (keep) or 0 (blank).
        return observation * on.reshape(-1, *[1] * (observation.dim() - 1))

    def _one_hot(self, action):
        return F.one_hot(torch.from_numpy(action), self.world.n_actions).float()

    @staticmethod
    def _to_torch(observations):
        return {name: torch.from_numpy(obs) for name, obs in observations.items()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    n_agents: int = 64          # parallel lives (only for speed)
    chunk_len: int = 32         # moments of life between weight updates
    n_updates: int = 1500       # total life = n_updates * chunk_len moments per agent
    learning_rate: float = 2e-3
    seed: int = 0
    # While growing up, the agent sometimes loses a sense for a while (night
    # falls, the world goes quiet). Each entry is
    #     sense: (chance per step to switch off, chance per step to come back)
    # (0.02, 0.06) = off about a quarter of the time, ~17 steps at a stretch.
    # Without this, the first darkness the brain ever meets would be the test
    # itself, and we would be measuring its confusion at a never-seen input
    # rather than whether it has a map.
    blackouts: dict = field(default_factory=lambda: {
        "vision": (0.02, 0.06),
        "hearing": (0.02, 0.06),
    })
    log_every: int = 250


def prediction_loss(experience):
    """Average prediction error per sense, counting only moments the agent
    actually sensed the outcome. (You cannot learn from a prediction about
    what you would have seen if the lights were off when it happened.)

    Returns dict sense -> scalar tensor.
    """
    losses = {}
    for name, error in experience["error"].items():
        sensed = experience["sensed"][name]
        losses[name] = (error * sensed).sum() / sensed.sum().clamp(min=1)
    return losses


def train(brain, world, config=None, verbose=True, tag=""):
    """Let the brain live and learn. Returns the loss history for plotting.

    tag: text put in front of each progress line (e.g. which seed this is).
    """
    config = config or TrainConfig()
    life = Life(brain, world, config.n_agents, seed=config.seed, blackouts=config.blackouts)
    optimizer = torch.optim.Adam(brain.parameters(), lr=config.learning_rate)

    history = {name: [] for name in brain.sense_shapes}
    for update in range(1, config.n_updates + 1):
        # Live a short stretch of life, remembering every computation so that
        # PyTorch can work out how each weight contributed to the errors.
        experience = life.live(config.chunk_len)
        losses = prediction_loss(experience)
        total = sum(losses.values())

        # Standard PyTorch learning step (same three lines as in Karpathy's
        # course): clear old gradients, backpropagate, nudge the weights.
        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), max_norm=1.0)   # safety against huge steps
        optimizer.step()

        # "Truncated backpropagation through time": the agent keeps its mind
        # (core_state) and carries on living, but detach() cuts the gradient
        # trail so the next update only looks back over the next chunk, not
        # over the agent's entire life.
        life.core_state = life.core_state.detach()

        for name, loss in losses.items():
            history[name].append(loss.item())
        if verbose and (update % config.log_every == 0 or update == 1):
            parts = "  ".join(f"{n} {history[n][-1]:.5f}" for n in history)
            print(f"  {tag}update {update:5d}/{config.n_updates}   prediction error:  {parts}", flush=True)

    return history
