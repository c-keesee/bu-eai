"""
model_v03.py -- the v0.3 brain.

WHY A NEW BRAIN. The v0.1/v0.2 brain (model.py) could not learn how things
move. Shown a roller crossing open floor, it bet about 40% on "it stays where
it is" and had no idea of its direction, even after 190,000 moments of life.
Its losses were almost the same as those of a copy with no memory at all.
(See the v0.3 entry in RESULTS.md for how this was tracked down.)

Two things are added here. The first is part of every v0.3 brain; the second
is one of the memory conditions being compared.

1. VISUAL SHORT-TERM MEMORY (always on).
   The old vision region squeezed each image into 32 numbers and forgot it.
   The new one keeps its own small map of recent activity: a 5x5 grid, one
   little vector per cell of the view, laid out like the view itself. It is a
   "convolutional GRU": the same small set of weights is applied at every
   cell, looking only at that cell and its neighbours.

   Why that helps: "something at this cell was one cell to the left a moment
   ago, so it will be one cell to the right next" is the same rule everywhere
   in the view. With shared weights the rule is learned once and applies at
   all 25 cells. The old brain had to discover it separately for every cell
   and direction.

   Brain analogy: early visual cortex is laid out like the retina (a
   "retinotopic map"), its neurons respond to motion, and it holds a brief
   trace of what was just seen ("iconic memory"). Sensory regions now have a
   short memory of their own; before, only the core did.

   The vision prediction also gets a LOCAL part: next to the old predictor
   that reads the core, a small convolution reads the visual map directly.
   The core says what the place is like; the map says what is moving in it.

2. EPISODIC MEMORY (optional; the "external memory" condition).
   A store of past moments that the core can write to and look things up in,
   loosely like the hippocampus next to the cortex:

     WRITE  every few moments: "what I sensed" (the regions' summaries),
            filed under an address made from the core's state at that time.
            The address code is a fixed random scrambling of the core state
            (the hippocampus is thought to do something similar to keep
            similar memories from blurring together).
     READ   every moment: the core forms a few queries ("what did I sense
            when things were like THIS?"), the store returns a blend of the
            best-matching memories, and that recollection is handed to the
            predictors now and to the core at the next moment.

   Only the reading is learned (which questions to ask, what to do with the
   answers). Nothing tells the brain what to remember or when to recall: as
   everywhere else, the only teacher is sensory prediction error.
   The store is wiped at the start of each life.

THE "MIND STATE". Life (training.py) carries one tensor per agent from moment
to moment and wipes it at each new life. For this brain that tensor is
    [ core state | visual map | last recollection ]
laid end to end. The episodic store itself is too big for that and lives in
its own object (EpisodicStore), which Life also carries and wipes.

The interface is the same as the old Brain's: initial_state(), perceive(),
predict(). So Life, train() and all the tests work with either.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import SensoryRegion


class EpisodicStore:
    """The contents of episodic memory for n_agents agents: plain tensors, no weights.

    keys[i, m]    the address memory m of agent i is filed under
    values[i, m]  what was sensed at that moment
    used[i, m]    False for slots not written yet in this life
    """

    def __init__(self, n_agents, slots, key_size, value_size, write_every):
        self.keys = torch.zeros(n_agents, slots, key_size)
        self.values = torch.zeros(n_agents, slots, value_size)
        self.used = torch.zeros(n_agents, slots, dtype=torch.bool)
        self.written = torch.zeros(n_agents, dtype=torch.long)    # memories filed so far in this life
        self.moment, self.write_every = 0, write_every

    def tick(self, keys, values):
        """One moment passes. Every `write_every` moments, each agent files
        the current moment away (once its store is full, the oldest memory is
        overwritten)."""
        if self.moment % self.write_every == 0:
            rows = torch.arange(len(self.written))
            slot = self.written % self.keys.shape[1]
            # clone() first: PyTorch still needs the OLD contents to work out
            # the gradients of lookups made earlier in this stretch of life.
            self.keys, self.values, self.used = self.keys.clone(), self.values.clone(), self.used.clone()
            self.keys[rows, slot] = keys.detach()
            self.values[rows, slot] = values.detach()
            self.used[rows, slot] = True
            self.written = self.written + 1
        self.moment += 1

    def forget(self, agents):
        """A new life begins for these agents: their store is emptied."""
        agents = torch.as_tensor(agents, dtype=torch.long)
        self.used, self.written = self.used.clone(), self.written.clone()
        self.used[agents] = False
        self.written[agents] = 0


class BrainV3(nn.Module):
    has_curiosity = False       # (the v0.2 curiosity heads are not part of v0.3)

    def __init__(self, sense_shapes, n_actions, core_units=256, filters=16, vision_latent=64,
                 latent_size=32, predictor_size=256, local_hidden=32, feedback_size=8,
                 episodic_memory=False, memory_slots=128, write_every=4, memory_heads=4,
                 key_size=64, recall_size=128):
        """
        core_units:      size of the core (the association area), as before
        filters:         numbers kept per cell of the visual map
        episodic_memory: add the hippocampus-like store (see the top of this file)
        memory_slots:    how many moments the store can hold
        write_every:     a moment is stored every this many moments
                         (memory_slots * write_every should cover one life)
        memory_heads:    how many questions the core asks the store per moment
        """
        super().__init__()
        self.sense_shapes, self.n_actions = dict(sense_shapes), n_actions
        channels, self.H, self.W = sense_shapes["vision"]
        self.core_units, self.filters = core_units, filters
        self.has_memory = episodic_memory
        self.recall_size = recall_size if episodic_memory else 0
        self.map_size = filters * self.H * self.W
        self.core_size = core_units + self.map_size + self.recall_size     # the whole mind state

        # --- Vision region with its own short-term memory (a convolutional GRU) ---
        # Each cell of the map looks at: the image there, the action just
        # taken (so it can allow for the view having shifted), a top-down
        # signal from the core, and its own and its neighbours' previous state.
        self.fb = nn.Linear(core_units, feedback_size)                      # top-down feedback, core -> vision
        seen = channels + n_actions + feedback_size + filters
        self.gates = nn.Conv2d(seen, 2 * filters, kernel_size=3, padding=1)  # how much to keep / overwrite
        self.cand = nn.Conv2d(seen, filters, kernel_size=3, padding=1)       # the proposed new contents
        self.vision_out = nn.Linear(self.map_size, vision_latent)           # summary handed to the core

        # --- The other senses: unchanged regions from model.py ---
        self.regions = nn.ModuleDict({name: SensoryRegion(shape, core_units, latent_size)
                                      for name, shape in sense_shapes.items() if name != "vision"})
        self.latent_total = vision_latent + latent_size * len(self.regions)

        # --- The core: a GRU over all the summaries, the last action and (if any) the last recollection ---
        self.core = nn.GRUCell(self.latent_total + n_actions + self.recall_size, core_units)

        # --- Predictors: as before, one per sense, reading the core (and the recollection) ---
        self.predictors = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(core_units + self.recall_size + n_actions, predictor_size), nn.ReLU(),
                                nn.Linear(predictor_size, math.prod(shape)))
            for name, shape in sense_shapes.items()})
        # ...plus the LOCAL vision predictor reading the visual map cell by cell.
        self.local = nn.Sequential(
            nn.Conv2d(filters + n_actions + feedback_size, local_hidden, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(local_hidden, channels, kernel_size=3, padding=1))

        # --- Episodic memory ---
        if episodic_memory:
            self.memory_slots, self.write_every = memory_slots, write_every
            self.heads, self.key_size = memory_heads, key_size
            self.key = nn.Linear(core_units, key_size, bias=False)          # the address code...
            self.key.weight.requires_grad_(False)                           # ...fixed, never trained
            self.query = nn.Linear(core_units, memory_heads * key_size)     # the questions (learned)
            self.sharpness = nn.Parameter(torch.tensor(math.log(8.0)))      # how picky the matching is (learned)
            self.recall_out = nn.Linear(memory_heads * self.latent_total, recall_size)

    # ------------------------------------------------------------------

    def initial_state(self, n_agents):
        """A blank mind: all zeros. Shape (n_agents, core_size)."""
        return torch.zeros(n_agents, self.core_size)

    def new_memory(self, n_agents):
        """An empty episodic store for n_agents agents (None if this brain has none)."""
        if not self.has_memory:
            return None
        return EpisodicStore(n_agents, self.memory_slots, self.key_size, self.latent_total, self.write_every)

    def _split(self, state):
        """The mind state -> (core, visual map, last recollection)."""
        core = state[:, :self.core_units]
        vmap = state[:, self.core_units:self.core_units + self.map_size].reshape(-1, self.filters, self.H, self.W)
        return core, vmap, state[:, self.core_units + self.map_size:]

    def _everywhere(self, vector):
        """(n, k) -> (n, k, H, W): hand the same vector to every cell of the map."""
        return vector[:, :, None, None].expand(-1, -1, self.H, self.W)

    def perceive(self, observations, last_action, state, memory=None):
        """One moment of sensation -> the new mind state. (Same contract as Brain.perceive.)"""
        core, vmap, recollection = self._split(state)

        # 1. The vision region updates its map (a GRU step at every cell).
        top_down = torch.tanh(self.fb(core))
        seen = torch.cat([observations["vision"], self._everywhere(last_action), self._everywhere(top_down)], dim=1)
        keep, reset = torch.sigmoid(self.gates(torch.cat([seen, vmap], dim=1))).chunk(2, dim=1)
        proposal = torch.tanh(self.cand(torch.cat([seen, reset * vmap], dim=1)))
        vmap = keep * vmap + (1 - keep) * proposal

        # 2. Every region hands its summary to the core.
        latents = [torch.relu(self.vision_out(vmap.flatten(1)))]
        latents += [region(observations[name], core) for name, region in self.regions.items()]
        latents = torch.cat(latents, dim=1)
        core = self.core(torch.cat([latents, last_action, recollection], dim=1), core)

        # 3. Episodic memory: look things up, then file this moment away
        #    (in that order, so a moment never just recalls itself).
        if self.has_memory:
            recollection = self._recall(core, memory)
            memory.tick(F.normalize(self.key(core), dim=1), latents)
        return torch.cat([core, vmap.flatten(1), recollection], dim=1)

    def _recall(self, core, memory):
        """Ask the store a few questions; return what comes back as one vector."""
        n = core.shape[0]
        questions = F.normalize(self.query(core).reshape(n, self.heads, self.key_size), dim=2)
        # How well each stored address matches each question (a dot product per pair).
        # matmul on (n, heads, k) x (n, k, slots) does this for all agents at once.
        match = torch.matmul(questions, memory.keys.transpose(1, 2)) * self.sharpness.exp()      # (n, heads, slots)
        match = match.masked_fill(~memory.used[:, None, :], -1e9)             # empty slots never match
        weights = torch.softmax(match, dim=2) * memory.used[:, None, :]       # (all zero if nothing is stored yet)
        recalled = torch.matmul(weights, memory.values)                       # a blend of the best matches
        return torch.relu(self.recall_out(recalled.flatten(1)))

    def predict(self, state, action):
        """Guess every sense's next observation, given the action about to be taken."""
        core, vmap, recollection = self._split(state)
        x = torch.cat([core, recollection, action], dim=1)
        out = {name: predictor(x).reshape(-1, *self.sense_shapes[name]) for name, predictor in self.predictors.items()}
        top_down = torch.tanh(self.fb(core))
        local = self.local(torch.cat([vmap, self._everywhere(action), self._everywhere(top_down)], dim=1))
        out["vision"] = out["vision"] + local
        return out
