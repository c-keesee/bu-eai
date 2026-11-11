"""
world.py -- the simulated world the agent lives in.

BRAIN ANALOGY: this file is *not* part of the brain. It is the outside world
(physics + the body's position in it). The brain never gets to look inside
this file's variables -- it only receives what the senses (senses.py) report.

SWAPPING THIS OUT LATER (v0.2+, MiniWorld / MuJoCo):
The brain and the training loop only ever use these five things:

    world.sense_shapes    dict: sense name -> shape of one observation
    world.n_actions       how many different actions exist
    world.reset(n)        put n agents somewhere, return first observations
    world.step(actions)   move the agents, return the new observations
    world.ground_truth()  the *true* hidden facts (position etc.). Used ONLY
                          by the tests afterwards, never shown to the brain.

Any class with those five members can replace GridWorld without touching the
brain. A 3D world would return camera images under "vision" instead of a 5x5
colour patch, and that is the only thing that changes.

"n agents": for speed we simulate many independent copies of the agent at once
("parallel lives"). They all live in the same map but never see or bump into
each other. Every array therefore has a leading dimension of size n_agents.
If you are used to C/Java: instead of `for each agent: pos = move(pos)` we
keep all positions in one NumPy array and move them all in one operation.
"""

from dataclasses import dataclass

import numpy as np

from .senses import Hearing, Touch, Vision


# ---------------------------------------------------------------------------
# What exists in the world
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectType:
    """One kind of thing in the world.

    Every object type shows up in MORE THAN ONE sense: it has a colour (vision),
    a texture (touch) and a sound (hearing). Nothing ever tells the brain that
    "red", "smooth" and "high-pitched" belong to the same object. If it figures
    that out, it figured it out by itself.
    """
    name: str
    symbol: str           # the character used for it in the ASCII map
    color: tuple          # (r, g, b), each 0..1  -> what vision reports
    texture: str          # one of TEXTURES below -> what touch reports
    sound_channel: int    # which "pitch" it emits -> what hearing reports
    loudness: float       # how loud it is right next to it
    solid: bool           # True: you bump into it. False: you can walk on it.


# Everything touch can feel. The position in this list is the texture's id.
TEXTURES = ["nothing", "rough", "smooth", "wet", "soft"]

OBJECT_TYPES = [
    #          name        sym  colour             texture   pitch  loud  solid
    ObjectType("bell",     "B", (1.0, 0.1, 0.1),   "smooth", 0,     1.0,  True),
    ObjectType("fountain", "F", (0.1, 0.3, 1.0),   "wet",    1,     0.7,  True),
    ObjectType("moss",     "m", (0.1, 0.8, 0.2),   "soft",   2,     0.4,  False),
]

# Colours of the things that are not objects.
FLOOR_COLOR = (0.25, 0.25, 0.25)
WALL_COLOR = (0.65, 0.65, 0.65)
VOID_COLOR = (0.0, 0.0, 0.0)      # outside the map (vision can peek past the edge)

# Cell codes stored in the grid array.
FLOOR, WALL = 0, 1
FIRST_OBJECT = 2                  # cell code 2 = OBJECT_TYPES[0], 3 = OBJECT_TYPES[1], ...

# The map. '#' wall, '.' floor, letters = objects (see OBJECT_TYPES).
# Edit this freely -- the only rule is that every row has the same length.
#
# It is deliberately lopsided: no mirror image or rotation of it looks the
# same, and every area has its own landmark (a wall stub in the top right, an
# L-shaped wall in the middle, a dead-end corridor in the bottom right, and
# objects scattered unevenly). If two places looked and sounded identical in
# every way, no amount of experience could tell them apart, and "where am I?"
# would have no answer for the brain to discover.
DEFAULT_MAP = """
############
#......#...#
#.B....#.m.#
#..........#
#....##....#
#.....#..F.#
#.m...#....#
#..........#
###..m...#.#
#.....B..#.#
#F.......#.#
############
"""

# Actions. Each one is a (dy, dx) step. Row 0 is the top of the map, so
# "north" means y gets smaller.
ACTION_NAMES = ["stay", "north", "east", "south", "west"]
ACTION_MOVES = np.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]])


# ---------------------------------------------------------------------------
# The world itself
# ---------------------------------------------------------------------------

class GridWorld:
    def __init__(self, ascii_map=DEFAULT_MAP, senses=None, seed=0):
        # ---- 1. turn the ASCII drawing into a 2D array of cell codes -------
        rows = [line for line in ascii_map.strip().splitlines()]
        self.height, self.width = len(rows), len(rows[0])
        symbol_to_code = {".": FLOOR, "#": WALL}
        for i, obj in enumerate(OBJECT_TYPES):
            symbol_to_code[obj.symbol] = FIRST_OBJECT + i

        self.grid = np.zeros((self.height, self.width), dtype=np.int64)
        for y, row in enumerate(rows):
            assert len(row) == self.width, "all map rows must have the same length"
            for x, ch in enumerate(row):
                self.grid[y, x] = symbol_to_code[ch]

        # ---- 2. derived facts about each cell ------------------------------
        # solid[y, x] is True where the agent cannot stand.
        self.solid = self.grid == WALL
        for i, obj in enumerate(OBJECT_TYPES):
            if obj.solid:
                self.solid |= self.grid == FIRST_OBJECT + i

        # All cells the agent can stand on, as an array of (y, x) pairs.
        self.free_cells = np.argwhere(~self.solid)

        # Where every object is: list of (y, x, object_type_index).
        self.objects = [
            (y, x, self.grid[y, x] - FIRST_OBJECT)
            for y, x in np.argwhere(self.grid >= FIRST_OBJECT)
        ]

        # ---- 3. the senses -------------------------------------------------
        # A list of sense objects (see senses.py). To give the agent a new
        # sense (smell?), write a class there and add it to this list; the
        # brain will automatically grow a region for it.
        self.senses = senses if senses is not None else [Vision(radius=2), Touch(), Hearing()]
        self.sense_shapes = {s.name: s.shape(self) for s in self.senses}
        self.n_actions = len(ACTION_NAMES)

        # SPEED TRICK (not part of the idea): the world never changes, so what
        # an agent senses depends only on which cell it stands on. We compute
        # every sense for every cell once, up front, and afterwards "sensing"
        # is just looking up a row in a table. A 3D world can't do this and
        # would render a camera image each step instead.
        self._sense_tables = {}
        for s in self.senses:
            table = np.zeros((self.height, self.width) + self.sense_shapes[s.name], dtype=np.float32)
            for y, x in self.free_cells:
                table[y, x] = s.sense(self, y, x)
            self._sense_tables[s.name] = table

        # Ground truth for the tests: for every cell, which object type is
        # nearest (straight-line distance)?
        self._nearest_object = np.zeros((self.height, self.width), dtype=np.int64)
        for y, x in self.free_cells:
            dists = [np.hypot(oy - y, ox - x) for oy, ox, _ in self.objects]
            self._nearest_object[y, x] = self.objects[int(np.argmin(dists))][2]

        self.rng = np.random.default_rng(seed)
        self.pos = None               # (n_agents, 2) array of (y, x); set by reset()

    # ------------------------------------------------------------------
    # The interface the brain / training loop uses
    # ------------------------------------------------------------------

    def reset(self, n_agents, seed=None):
        """Drop n_agents onto random free cells. Returns their first observations.

        Pass a seed to get the same starting cells every time (the tests use
        this to replay the exact same walk under different conditions).
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        picks = self.rng.integers(0, len(self.free_cells), size=n_agents)
        self.pos = self.free_cells[picks].copy()
        return self.observe()

    def step(self, actions):
        """Move every agent by its chosen action. Returns the new observations.

        actions: integer array of shape (n_agents,), values 0..n_actions-1.
        Walking into something solid simply fails: the agent stays put.
        """
        target = self.pos + ACTION_MOVES[actions]
        blocked = self.solid[target[:, 0], target[:, 1]]
        # np.where(condition, a, b): pick a where condition is True, else b.
        self.pos = np.where(blocked[:, None], self.pos, target)
        return self.observe()

    def observe(self):
        """What every agent senses right now: dict sense name -> float32 array."""
        y, x = self.pos[:, 0], self.pos[:, 1]
        return {name: table[y, x] for name, table in self._sense_tables.items()}

    def ground_truth(self):
        """The true hidden state of the world. FOR THE TESTS ONLY.

        The brain is never given any of this. The emergence test asks whether
        the brain worked these facts out on its own.
        """
        y, x = self.pos[:, 0], self.pos[:, 1]
        return {
            "position": np.stack([x, y], axis=1).astype(np.float32),   # (n_agents, 2) as (x, y)
            "nearest_object": self._nearest_object[y, x],               # (n_agents,) object type index
        }

    # ------------------------------------------------------------------
    # Small helpers used by the senses
    # ------------------------------------------------------------------

    def in_bounds(self, y, x):
        return 0 <= y < self.height and 0 <= x < self.width

    def object_at(self, y, x):
        """The ObjectType in cell (y, x), or None if it is floor / wall."""
        code = self.grid[y, x]
        return OBJECT_TYPES[code - FIRST_OBJECT] if code >= FIRST_OBJECT else None

    def render_ascii(self):
        """The map as text, handy for printing."""
        symbols = {FLOOR: ".", WALL: "#"}
        for i, obj in enumerate(OBJECT_TYPES):
            symbols[FIRST_OBJECT + i] = obj.symbol
        return "\n".join("".join(symbols[c] for c in row) for row in self.grid)
