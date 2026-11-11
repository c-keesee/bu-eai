"""
training.py -- living and learning.

Two things live here:

    Life     the loop of experience: sense -> update the mind -> act ->
             predict -> see what really happened. Used for training AND by
             all the tests, so they are guaranteed to run the brain identically.

    train()  lets a brain live for a while and, after every short stretch of
             life, adjusts its weights to make its predictions less wrong.

THE ONLY LEARNING SIGNAL FOR THE WORLD MODEL is prediction error: (what I
predicted I would sense next - what I actually sensed next)^2. There is no
reward, no label, no goal, and no language. Nothing here ever reads
world.ground_truth().

New in v0.2 (all switched off by default, so v0.1 runs exactly as before):
    * life_length: after this many moments an agent starts a new life in a
      brand-new map, with the same weights but a blank core state.
    * movement: "random" (as in v0.1), or one of two curious styles that
      prefer actions the brain expects to learn from (see Brain.curiosity).
"""

import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

STAY = 0                 # every world must make action 0 mean "do not move"
ERROR_SCALE = 100.0      # the self-monitoring heads work with error * 100, so their numbers are near 1


class Life:
    """A stream of experience: n_agents copies of the brain living in a world.

    `live(n)` advances all agents by n moments; call it again to continue from
    exactly where they stopped.

    A Life takes over the world object it is given (it places its agents in
    it). So do not keep two Lives going at once on the same world object;
    finish with one before starting the next, or give each its own world.
    """

    def __init__(self, brain, world, n_agents, seed=0, persistence=0.6, blackouts=None,
                 movement="random", curiosity_strength=3.0, life_length=None, memoryless=False):
        """
        persistence: chance of repeating the previous action instead of picking
                     a fresh random one (so the walk has some momentum).
        blackouts:   dict, sense name -> (chance per step that a working sense
                     goes dark, chance per step that a dark sense comes back).
                     Think "night falls" for vision. None = senses always work.
        movement:    "random"    the momentum random walk
                     "progress"  the same walk, tilted toward actions with high
                                 expected learning progress
                     "error"     the same walk, tilted toward actions with high
                                 expected prediction error (naive curiosity)
        curiosity_strength: how hard curiosity tilts the walk (0 = not at all).
        life_length: moments per life. None = one endless life (v0.1).
        memoryless:  (v0.3 control) wipe the core state before every moment,
                     so the brain can only use what it senses right now.
        """
        self.memoryless = memoryless
        self.brain, self.world = brain, world
        self.n_agents = n_agents
        self.persistence = persistence
        self.blackouts = blackouts or {}
        self.movement = movement
        self.curiosity_strength = curiosity_strength
        self.life_length = life_length
        self.rng = np.random.default_rng(seed)

        self.observations = self._to_torch(world.reset(n_agents, seed=seed))
        self.core_state = brain.initial_state(n_agents)
        # (v0.3) Brains with an episodic memory keep its contents in a separate
        # store, which lives and dies with the agent just like the core state.
        self.memory = brain.new_memory(n_agents) if getattr(brain, "has_memory", False) else None
        self.last_action = np.zeros(n_agents, dtype=np.int64)        # "stay"
        # sense_on[name][i] is False while agent i's sense `name` is blanked.
        self.sense_on = {name: np.ones(n_agents, dtype=bool) for name in brain.sense_shapes}
        # How old each agent is in its current life. Starting ages are spread
        # out so that the agents do not all change map at the same moment.
        self.age = np.zeros(n_agents, dtype=np.int64)
        if life_length:
            self.age = self.rng.integers(0, life_length, size=n_agents)

    # ------------------------------------------------------------------

    def live(self, n_steps, record=False):
        """Live for n_steps moments.

        Returns a dict:
            "error"     sense -> tensor (n_steps, n_agents): prediction error
                        for that sense at each moment (mean squared error)
            "sensed"    sense -> tensor (n_steps, n_agents): 1 if the agent
                        actually got to sense the outcome, 0 if blanked
            "actions"   tensor (n_steps, n_agents): the action taken
            "reborn"    tensor (n_steps, n_agents): True if a new life began
                        right after this moment
            "first_look", "second_look"  (only if the brain has curiosity
                        heads) tensors (n_steps, n_agents): the brain's guesses
                        at its own error, see curiosity_loss()
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
        action_index_log, reborn_log = [], []
        outcome = {name: [] for name in names}     # what each sense really reported next
        sensed = {name: [] for name in names}      # ...and whether the agent got to sense it

        for _ in range(n_steps):
            # 1. SENSE. A blanked sense delivers all zeros (darkness/silence).
            inputs = {name: self._blank_if_off(name, self.observations[name]) for name in names}

            # 2. UPDATE THE MIND. Regions encode the senses (with top-down
            #    feedback), the core integrates them with its memory.
            if self.memoryless:
                self.core_state = self.brain.initial_state(self.n_agents)
            extras = {} if self.memory is None else {"memory": self.memory}
            self.core_state = self.brain.perceive(
                inputs, self._one_hot(self.last_action), self.core_state, **extras)
            core_log.append(self.core_state)
            if record:
                senses_log.append(torch.cat([inputs[n].flatten(1) for n in names], dim=1))
                truth_log.append(self.world.ground_truth())

            # 3. ACT. Random walk with momentum, possibly tilted by curiosity.
            action = self._choose_action()
            action_log.append(self._one_hot(action))
            action_index_log.append(torch.from_numpy(action))

            # 4. THE WORLD MOVES ON, and we find out what really happened.
            self.observations = self._to_torch(self.world.step(action))
            self.last_action = action
            self._update_blackouts()
            for name in names:
                outcome[name].append(self.observations[name])
                sensed[name].append(torch.from_numpy(self.sense_on[name].astype(np.float32)))

            # (v0.2) Agents whose life is over start a new one in a new map.
            reborn = np.zeros(self.n_agents, dtype=bool)
            if self.life_length:
                self.age += 1
                reborn = self.age >= self.life_length
                if reborn.any():
                    self._begin_new_lives(np.flatnonzero(reborn))
            reborn_log.append(torch.from_numpy(reborn))

        # 5. PREDICT, then 6. COMPARE with reality.
        #    Logically each prediction belongs between steps 3 and 4: it uses
        #    only the core state and the action from that moment, before the
        #    outcome is known. Because predictions do not influence the
        #    choice of action, we can compute them for all moments in one
        #    batch here, which is noticeably faster than doing it in the loop.
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
            "actions": torch.stack(action_index_log),
            "reborn": torch.stack(reborn_log),
        }
        if self.brain.has_curiosity:
            first, second = self.brain.expected_errors(core.flatten(0, 1), actions.flatten(0, 1))
            out["first_look"] = first.reshape(n_steps, self.n_agents)
            out["second_look"] = second.reshape(n_steps, self.n_agents)
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
        n_actions = self.world.n_actions
        if self.movement == "random":
            fresh = self.rng.integers(0, n_actions, size=self.n_agents)
            repeat = self.rng.random(self.n_agents) < self.persistence
            return np.where(repeat, self.last_action, fresh)

        # Curious movement. Start from the chances the random walk would give
        # each action, then multiply each by exp(strength * curiosity score).
        # With strength 0 this is exactly the random walk, so the ONLY
        # difference between the movement styles is the curiosity tilt.
        chance = np.full((self.n_agents, n_actions), (1 - self.persistence) / n_actions)
        chance[np.arange(self.n_agents), self.last_action] += self.persistence
        score = self.brain.curiosity(self.core_state, self.movement).numpy()
        score = score - score.max(axis=1, keepdims=True)          # keeps exp() from overflowing
        weight = chance * np.exp(self.curiosity_strength * score)

        # Draw one action per agent in proportion to its weights.
        cumulative = np.cumsum(weight, axis=1)
        roll = self.rng.random(self.n_agents) * cumulative[:, -1]
        return np.minimum((roll[:, None] > cumulative).sum(axis=1), n_actions - 1)

    def _begin_new_lives(self, agents):
        """Same brain wiring (weights), blank mind, brand-new map."""
        fresh = self.world.new_life(agents)
        for name, obs in fresh.items():
            # clone(): the old tensor is still referenced as a recorded outcome
            updated = self.observations[name].clone()
            updated[agents] = torch.from_numpy(obs)
            self.observations[name] = updated
        keep = torch.ones(self.n_agents, 1)
        keep[agents] = 0.0
        self.core_state = self.core_state * keep       # wipe the reborn agents' core state
        if self.memory is not None:
            self.memory.forget(agents)                 # ...and their episodic memories
        self.last_action = self.last_action.copy()
        self.last_action[agents] = STAY
        self.age[agents] = 0
        for name in self.sense_on:
            self.sense_on[name][agents] = True

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
    # v0.2 additions (the defaults reproduce v0.1):
    movement: str = "random"            # "random", "progress" or "error" (see Life)
    curiosity_strength: float = 3.0
    life_length: int = None             # moments per life; None = one endless life
    final_learning_rate: float = None   # if set, the learning rate glides down to this value...
    decay_from: float = 0.0             # ...starting after this fraction of training (0 = from the start)
    memoryless: bool = False            # v0.3 control: no memory at all (see Life)
    clip: float = 1.0                   # largest allowed gradient size per update (None = no limit)


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


def curiosity_loss(experience):
    """Teach the two self-monitoring heads to guess the brain's own error.

    measured[t] = how wrong the prediction made at moment t turned out to be
                  (summed over the senses the agent actually got to sense).

    first_look[t]  should equal measured[t].
    second_look[t] should equal measured[t+1], but ONLY for moments where the
                   agent then chose to stay put. Staying put means looking at
                   the same place again, so measured[t+1] is "how wrong was I
                   about that place on the second look". For ordinary scenery
                   that is near zero; for TV static it is as large as the
                   first time.

    The targets are detached: the heads learn to describe the errors, they do
    not get to change them.
    """
    measured = sum(experience["error"][name] * experience["sensed"][name]
                   for name in experience["error"]).detach() * ERROR_SCALE

    first = ((experience["first_look"] - measured) ** 2).mean()

    stayed_next = experience["actions"][1:] == STAY
    same_life = ~experience["reborn"][:-1]
    usable = (stayed_next & same_life).float()
    second = (((experience["second_look"][:-1] - measured[1:]) ** 2) * usable).sum() / usable.sum().clamp(min=1)
    return {"first_look": first, "second_look": second}


def train(brain, world, config=None, verbose=True, tag="", checkpoint=None, save_path=None):
    """Let the brain live and learn. Returns the loss history for plotting.

    tag:        text put in front of each progress line (e.g. which seed this is).
    checkpoint: optional function called as checkpoint(update, brain) after
                every logged update (used by the pilot to watch progress).
    save_path:  optional file. If given, the complete state of training is
                saved there at every logged update, and if the file already
                exists training picks up from it instead of starting over.
                (For long runs on machines that may restart. A resumed run
                gives exactly the same result as an uninterrupted one.)
    """
    config = config or TrainConfig()
    life = Life(brain, world, config.n_agents, seed=config.seed, blackouts=config.blackouts,
                movement=config.movement, curiosity_strength=config.curiosity_strength,
                life_length=config.life_length, memoryless=config.memoryless)
    optimizer = torch.optim.Adam(brain.parameters(), lr=config.learning_rate)

    is_head = lambda name: name.startswith(("first_look", "second_look"))
    model_weights = [p for name, p in brain.named_parameters() if not is_head(name)]
    head_weights = [p for name, p in brain.named_parameters() if is_head(name)]

    history = {name: [] for name in brain.sense_shapes}
    if brain.has_curiosity:
        history.update({"first_look": [], "second_look": []})

    first_update, seconds_so_far, started = 1, 0.0, time.time()
    if save_path and os.path.exists(save_path):
        saved = torch.load(save_path, weights_only=False)
        brain.load_state_dict(saved["brain"])
        optimizer.load_state_dict(saved["optimizer"])
        world.__dict__.update(saved["world"])           # the maps, positions and random stream
        for key, value in saved["life"].items():        # the agents' minds, ages, senses, random stream
            setattr(life, key, value)
        history, seconds_so_far = saved["history"], saved["seconds"]
        first_update = saved["update"] + 1
        if verbose:
            print(f"  {tag}resuming from update {saved['update']}", flush=True)

    for update in range(first_update, config.n_updates + 1):
        if config.final_learning_rate:
            # Glide smoothly (geometrically) from the starting rate to the final one:
            # big steps early to learn fast, small steps late to settle.
            start = config.decay_from * config.n_updates
            progress = max(0.0, (update - 1 - start) / max(1.0, config.n_updates - 1 - start))
            rate = config.learning_rate * (config.final_learning_rate / config.learning_rate) ** progress
            for group in optimizer.param_groups:
                group["lr"] = rate

        # Live a short stretch of life, remembering every computation so that
        # PyTorch can work out how each weight contributed to the errors.
        experience = life.live(config.chunk_len)
        losses = prediction_loss(experience)
        if brain.has_curiosity:
            losses.update(curiosity_loss(experience))
        total = sum(losses.values())

        # Standard PyTorch learning step (same three lines as in Karpathy's
        # course): clear old gradients, backpropagate, nudge the weights.
        optimizer.zero_grad()
        total.backward()
        # Safety against huge steps. The self-monitoring heads are clipped
        # separately so that their gradients can never shrink the world model's.
        if config.clip:
            torch.nn.utils.clip_grad_norm_(model_weights, max_norm=config.clip)
        if head_weights:
            torch.nn.utils.clip_grad_norm_(head_weights, max_norm=1.0)
        optimizer.step()

        # "Truncated backpropagation through time": the agent keeps its mind
        # (core_state) and carries on living, but detach() cuts the gradient
        # trail so the next update only looks back over the next chunk, not
        # over the agent's entire life.
        life.core_state = life.core_state.detach()

        for name, loss in losses.items():
            history[name].append(loss.item())
        if update % config.log_every == 0 or update == 1:
            if verbose:
                parts = "  ".join(f"{n} {history[n][-1]:.5f}" for n in history)
                print(f"  {tag}update {update:5d}/{config.n_updates}   prediction error:  {parts}", flush=True)
            if checkpoint:
                checkpoint(update, brain)
            if save_path:
                keep = ("observations", "core_state", "last_action", "sense_on", "age", "rng", "memory")
                torch.save({"brain": brain.state_dict(), "optimizer": optimizer.state_dict(),
                            "world": world.__dict__, "life": {k: getattr(life, k) for k in keep},
                            "history": history, "update": update,
                            "seconds": seconds_so_far + time.time() - started}, save_path + ".tmp")
                os.replace(save_path + ".tmp", save_path)     # swap in one step: never a half-written file

    if save_path:
        history["seconds"] = [seconds_so_far + time.time() - started]
    return history
