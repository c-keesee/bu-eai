"""
many_worlds.py -- v0.2: every life happens in a different, randomly built map.

WHY: in v0.1 the agent lived in one fixed map for its whole life, so the
layout ended up memorised in the network's weights. Here no two lives share a
map. The weights can only learn *how to map a place*; the map itself has to
be built in the core state, during the life, from experience.

WHAT IS DIFFERENT FROM world.py (v0.1):
    * maps are 16x16 and generated at random (walls, one bell, one fountain,
      three moss patches, one TV), with checks that keep them usable
    * every agent has its OWN map (they still never meet)
    * new_life(agents): gives those agents a brand-new map
    * a "noisy TV": an object whose colour is random at every moment. Nothing
      about it can ever be predicted. It is there to test curiosity: a naive
      curious agent stares at it forever (see model.py / training.py).

The brain-facing interface is the same five members as GridWorld
(sense_shapes, n_actions, reset, step, ground_truth), plus new_life.

SPEED: v0.1 computed its sense tables with readable Python loops (senses.py).
We now need tens of thousands of maps per training run, so the same three
senses are recomputed here with NumPy array operations for a whole batch of
maps at once. tests/test_many_worlds.py checks that both versions agree
number for number.
"""

from collections import deque

import numpy as np

from .world import (ACTION_MOVES, ACTION_NAMES, FIRST_OBJECT, FLOOR, FLOOR_COLOR, OBJECT_TYPES,
                    TEXTURES, VOID_COLOR, WALL, WALL_COLOR, ObjectType)

# ---------------------------------------------------------------------------
# What exists in a v0.2 world: the three v0.1 objects plus the noisy TV
# ---------------------------------------------------------------------------

V2_TEXTURES = TEXTURES + ["glassy"]

TV_STATIC_COLOR = (0.5, 0.5, 0.5)      # placeholder; the real colour is re-rolled every moment
V2_OBJECT_TYPES = OBJECT_TYPES + [
    #          name  sym  colour            texture   pitch  loud  solid
    ObjectType("tv", "T", TV_STATIC_COLOR,  "glassy", None,  0.0,  True),
]
BELL, FOUNTAIN, MOSS, TV = range(4)                 # indices into V2_OBJECT_TYPES
OBJECTS_PER_MAP = [BELL, FOUNTAIN, MOSS, MOSS, MOSS, TV]   # exactly one bell, fountain and TV


# ---------------------------------------------------------------------------
# Random map generation
# ---------------------------------------------------------------------------

def _connected(walkable):
    """True if every walkable cell can be reached from every other (flood fill)."""
    cells = np.argwhere(walkable)
    start = tuple(cells[0])
    seen, queue = {start}, deque([start])
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            cell = (y + dy, x + dx)
            if walkable[cell] and cell not in seen:
                seen.add(cell)
                queue.append(cell)
    return len(seen) == len(cells)


def is_asymmetric(grid):
    """True if no mirror image or rotation of the map equals the map."""
    variants = [np.fliplr(grid), np.flipud(grid), np.rot90(grid, 1), np.rot90(grid, 2),
                np.rot90(grid, 3), grid.T, np.rot90(grid, 2).T]
    return not any(np.array_equal(grid, v) for v in variants)


def random_map(rng, size=16):
    """Build one random map: a (size, size) array of cell codes.

    Recipe: an outer wall, a few straight wall pieces dropped at random
    inside, then the objects on random free cells. The attempt is thrown away
    and redone if the result is unusable:
        * some area is walled off (not every cell reachable)
        * too little open floor
        * the map equals one of its own mirror images / rotations
    """
    interior = (size - 2) ** 2
    min_free = int(0.75 * interior)                 # keep at least three quarters open
    most_pieces = max(3, round(interior / 24))      # 8 pieces in a 16x16 map, 4 in a 12x12
    while True:
        grid = np.full((size, size), FLOOR, dtype=np.int64)
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = WALL

        for _ in range(rng.integers(most_pieces // 2 + 1, most_pieces + 1)):
            length = rng.integers(2, 6)
            y, x = rng.integers(1, size - 1, size=2)
            if rng.random() < 0.5:
                grid[y, x:min(x + length, size - 1)] = WALL        # horizontal piece
            else:
                grid[y:min(y + length, size - 1), x] = WALL        # vertical piece

        floor = np.argwhere(grid == FLOOR)
        if len(floor) < min_free:
            continue
        picks = rng.choice(len(floor), size=len(OBJECTS_PER_MAP), replace=False)
        for (y, x), kind in zip(floor[picks], OBJECTS_PER_MAP):
            grid[y, x] = FIRST_OBJECT + kind

        solid = grid == WALL
        for kind, obj in enumerate(V2_OBJECT_TYPES):
            if obj.solid:
                solid |= grid == FIRST_OBJECT + kind
        if _connected(~solid) and is_asymmetric(grid):
            return grid


def grid_to_ascii(grid):
    symbols = {FLOOR: ".", WALL: "#"}
    for i, obj in enumerate(V2_OBJECT_TYPES):
        symbols[FIRST_OBJECT + i] = obj.symbol
    return "\n".join("".join(symbols[c] for c in row) for row in grid)


# ---------------------------------------------------------------------------
# The three senses, computed for every cell of a batch of maps at once.
# Same definitions as senses.py, written with array operations.
# ---------------------------------------------------------------------------

# Per cell code: its colour, its texture id, whether it is solid.
_COLORS = np.array([FLOOR_COLOR, WALL_COLOR] + [o.color for o in V2_OBJECT_TYPES], dtype=np.float32)
_TEXTURE_ID = np.array([V2_TEXTURES.index("nothing"), V2_TEXTURES.index("rough")]
                       + [V2_TEXTURES.index(o.texture) for o in V2_OBJECT_TYPES])
_SOLID = np.array([False, True] + [o.solid for o in V2_OBJECT_TYPES])
_N_CHANNELS = 1 + max(o.sound_channel for o in V2_OBJECT_TYPES if o.sound_channel is not None)
_TOUCH_SPOTS = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]      # under, N, E, S, W (as in senses.py)


def vision_tables(grids, radius):
    """(maps, H, W) cell codes -> (maps, H, W, 3, size, size) colour patches."""
    colors = _COLORS[grids]                                       # (maps, H, W, 3)
    void = np.array(VOID_COLOR, dtype=np.float32)
    padded = np.empty((grids.shape[0], grids.shape[1] + 2 * radius, grids.shape[2] + 2 * radius, 3),
                      dtype=np.float32)
    padded[:] = void
    padded[:, radius:-radius, radius:-radius] = colors
    size = 2 * radius + 1
    # A sliding window over the padded image: one (size x size) patch per cell.
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size), axis=(1, 2))
    return np.ascontiguousarray(windows)                          # (maps, H, W, 3, size, size)


def touch_tables(grids):
    """(maps, H, W) -> (maps, H, W, 5 spots * textures) one-hot textures."""
    textures = _TEXTURE_ID[grids]
    padded = np.pad(textures, ((0, 0), (1, 1), (1, 1)), constant_values=V2_TEXTURES.index("rough"))
    H, W = grids.shape[1:]
    out = np.zeros(grids.shape + (len(_TOUCH_SPOTS), len(V2_TEXTURES)), dtype=np.float32)
    for i, (dy, dx) in enumerate(_TOUCH_SPOTS):
        shifted = padded[:, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]   # texture of the neighbour
        np.put_along_axis(out[:, :, :, i], shifted[..., None], 1.0, axis=-1)
    return out.reshape(grids.shape + (-1,))


def hearing_tables(grids, falloff=2.0):
    """(maps, H, W) -> (maps, H, W, channels) loudness per pitch."""
    n_maps, H, W = grids.shape
    ys, xs = np.mgrid[0:H, 0:W]
    out = np.zeros((n_maps, H, W, _N_CHANNELS), dtype=np.float32)
    for m, y, x in np.argwhere(grids >= FIRST_OBJECT):
        obj = V2_OBJECT_TYPES[grids[m, y, x] - FIRST_OBJECT]
        if obj.sound_channel is None:
            continue
        distance = np.hypot(ys - y, xs - x)
        out[m, :, :, obj.sound_channel] += obj.loudness / (1.0 + (distance / falloff) ** 2)
    return out


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------

class ManyWorlds:
    """n_agents agents, each alone in its own random map.

    All per-map data has a leading dimension of size n_agents: grid[i] is the
    map agent i currently lives in.
    """

    def __init__(self, size=16, vision_radius=2, seed=0):
        self.size = size
        self.radius = vision_radius
        patch = 2 * vision_radius + 1
        self.sense_shapes = {
            "vision": (3, patch, patch),
            "touch": (len(_TOUCH_SPOTS) * len(V2_TEXTURES),),
            "hearing": (_N_CHANNELS,),
        }
        self.n_actions = len(ACTION_NAMES)
        self.object_types, self.textures = V2_OBJECT_TYPES, V2_TEXTURES
        self.rng = np.random.default_rng(seed)
        self.maps_made = 0
        self.map_fingerprints = set()        # to check that test maps were never seen in training
        self.pos = None

    # ------------------------------------------------------------------
    # The interface the brain / training loop uses
    # ------------------------------------------------------------------

    def reset(self, n_agents, seed=None):
        """Give n_agents agents a fresh random map each. Returns first observations."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        S = self.size
        self.grid = np.zeros((n_agents, S, S), dtype=np.int64)
        self.solid = np.zeros((n_agents, S, S), dtype=bool)
        self.tables = {name: np.zeros((n_agents, S, S) + shape, dtype=np.float32)
                       for name, shape in self.sense_shapes.items()}
        self.landmarks = {name: np.zeros((n_agents, 2), dtype=np.int64)     # (y, x) per map
                          for name in ("bell", "fountain", "tv")}
        self.seen = np.zeros((n_agents, S, S), dtype=bool)    # cells that have been in view this life
        self.pos = np.zeros((n_agents, 2), dtype=np.int64)
        self._agents = np.arange(n_agents)
        self._build(self._agents)
        return self.observe()

    def new_life(self, agents):
        """The listed agents start a new life: each gets a brand-new map.

        Returns the first observations of the new lives (rows = those agents only).
        """
        self._build(np.asarray(agents))
        return {name: obs[agents] for name, obs in self.observe(mark_seen=False).items()}

    def step(self, actions):
        """Move every agent in its own map. Walking into something solid fails."""
        target = self.pos + ACTION_MOVES[actions]
        blocked = self.solid[self._agents, target[:, 0], target[:, 1]]
        self.pos = np.where(blocked[:, None], self.pos, target)
        return self.observe()

    def observe(self, mark_seen=True):
        y, x = self.pos[:, 0], self.pos[:, 1]
        obs = {name: table[self._agents, y, x] for name, table in self.tables.items()}

        # The noisy TV: wherever it is inside an agent's view, its pixel gets a
        # fresh random colour (each of R, G, B is 0 or 1 by coin flip).
        tv = self.landmarks["tv"]
        row, col = tv[:, 0] - y + self.radius, tv[:, 1] - x + self.radius
        patch = 2 * self.radius + 1
        in_view = (row >= 0) & (row < patch) & (col >= 0) & (col < patch)
        colors = self.rng.integers(0, 2, size=(len(y), 3)).astype(np.float32)
        obs["vision"][in_view, :, row[in_view], col[in_view]] = colors[in_view]
        self._tv_in_view = in_view

        if mark_seen:        # bookkeeping for the tests only: which cells have been looked at
            offsets = np.arange(-self.radius, self.radius + 1)
            ys = np.clip(y[:, None, None] + offsets[None, :, None], 0, self.size - 1)
            xs = np.clip(x[:, None, None] + offsets[None, None, :], 0, self.size - 1)
            self.seen[self._agents[:, None, None], ys, xs] = True
        return obs

    def ground_truth(self):
        """True hidden facts, FOR THE TESTS ONLY (never shown to the brain)."""
        walkable = ~self.solid
        as_xy = lambda yx: yx[:, ::-1].astype(np.float32)          # (y, x) -> (x, y)
        return {
            "position": as_xy(self.pos),
            "bell": as_xy(self.landmarks["bell"]),
            "fountain": as_xy(self.landmarks["fountain"]),
            "tv_in_view": self._tv_in_view.copy(),
            # share of the map's walkable cells that have been in view so far this life
            "coverage": ((self.seen & walkable).sum(axis=(1, 2)) / walkable.sum(axis=(1, 2))).astype(np.float32),
        }

    # ------------------------------------------------------------------

    def _build(self, agents):
        """Make new maps for these agents and drop each agent on a random free cell."""
        for i in agents:
            while True:
                grid = random_map(self.rng, self.size)
                tables = {"vision": vision_tables(grid[None], self.radius)[0],
                          "touch": touch_tables(grid[None])[0],
                          "hearing": hearing_tables(grid[None])[0]}
                solid = _SOLID[grid]
                free = np.argwhere(~solid)
                # Last usability check: no two free cells may feel exactly the
                # same in every sense, or position would be unknowable there.
                snapshot = np.concatenate([t[free[:, 0], free[:, 1]].reshape(len(free), -1)
                                           for t in tables.values()], axis=1)
                if len(np.unique(snapshot.round(4), axis=0)) == len(free):
                    break
            self.grid[i], self.solid[i] = grid, solid
            for name, table in tables.items():
                self.tables[name][i] = table
            for name, kind in (("bell", BELL), ("fountain", FOUNTAIN), ("tv", TV)):
                self.landmarks[name][i] = np.argwhere(grid == FIRST_OBJECT + kind)[0]
            self.pos[i] = free[self.rng.integers(len(free))]
            self.seen[i] = False
            self.map_fingerprints.add(grid.astype(np.uint8).tobytes())
            self.maps_made += 1

    def render_ascii(self, agent=0):
        return grid_to_ascii(self.grid[agent])
