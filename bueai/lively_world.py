"""
lively_world.py -- v0.3: a world where things move, hide and react.

WHY: common sense about objects ("things still exist when hidden", "things do
not pass through walls", "lifeless things only move when pushed") cannot be
learned in a world where nothing ever moves. v0.1 and v0.2 were such worlds.

WHAT IS NEW (everything else is as in many_worlds.py: a new random 16x16 map
for every life, one map per agent, the same three senses):

    BALLS    lifeless. A ball moves only when something moves into it: the
             agent, a roller, or another rolling ball. Once moving it keeps
             rolling in a straight line until something stops it (there is no
             friction: "a thing in motion stays in motion").
    ROLLERS  self-moving, like small animals. They travel in straight lines,
             turn when blocked, and sometimes rest and set off again on their
             own. A roller that runs into a ball pushes it.
    SCREENS  covered floor. Balls and rollers pass underneath and cannot be
             seen or felt while there. The agent cannot enter. This is how
             things go out of sight and come back, as in the baby studies.

ONE MOMENT OF PHYSICS (advance()), in this order:
    1. the agent moves (pushing a ball if it walks into one)
    2. each ball that is rolling moves one cell, or stops if blocked
    3. each roller moves one cell, turns, rests or wakes up
Nothing ever shares a cell: the agent, balls and rollers are all solid.

SPEED: as in many_worlds.py every array has a leading dimension "one row per
agent", and the physics is written with NumPy operations over all agents at
once. The only Python loop is over the handful of objects in a map.

THE TESTS need to break the rules on purpose (a ball that vanishes, a wall
that lets a ball through). For that the world keeps a few per-agent switches
that are never touched in normal life: alive, invisible, stuck and
blocks_objects. See commonsense_test.py.
"""

import numpy as np

from .many_worlds import _connected, is_asymmetric
from .world import ACTION_MOVES, ACTION_NAMES, FLOOR_COLOR, VOID_COLOR, WALL_COLOR

# ---------------------------------------------------------------------------
# What exists
# ---------------------------------------------------------------------------

# Things that never move, stored as a code per map cell.
FLOOR, WALL, BELL, FOUNTAIN, MOSS, SCREEN = range(6)
#                 floor         wall        bell             fountain         moss             screen
CELL_COLORS = [FLOOR_COLOR, WALL_COLOR, (1.0, 0.1, 0.1), (0.1, 0.3, 1.0), (0.1, 0.8, 0.2), (0.95, 0.9, 0.7)]
CELL_TEXTURES = ["nothing", "rough", "smooth", "wet", "soft", "cloth"]
CELL_SYMBOLS = ".#BFmS"
BLOCKS_AGENT = np.array([False, True, True, True, False, True])       # the agent cannot enter a screen
BLOCKS_OBJECTS = np.array([False, True, True, True, False, False])    # balls and rollers can
# Sounds, as in v0.1: cell code -> (pitch channel, loudness). Movers are silent,
# otherwise hearing would give away where a hidden ball is.
SOUNDS = {BELL: (0, 1.0), FOUNTAIN: (1, 0.7), MOSS: (2, 0.4)}
LANDMARKS_PER_MAP = [BELL, FOUNTAIN, MOSS, MOSS, MOSS]

# Things that move.
BALL, ROLLER = 0, 1
MOVER_COLORS = np.array([(1.0, 0.8, 0.0), (0.9, 0.2, 0.9)], dtype=np.float32)   # yellow, magenta
MOVER_TEXTURES = ["hard", "furry"]
MOVER_SYMBOLS = "or"

TEXTURES = ["nothing", "rough", "smooth", "wet", "soft", "cloth", "hard", "furry"]
TOUCH_SPOTS = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]       # under, N, E, S, W
STAY = 0                                                       # action 0; also "not moving" for objects

_COLORS = np.array(CELL_COLORS, dtype=np.float32)
_TEXTURE_ID = np.array([TEXTURES.index(t) for t in CELL_TEXTURES])
_MOVER_TEXTURE_ID = [TEXTURES.index(t) for t in MOVER_TEXTURES]


# ---------------------------------------------------------------------------
# Random maps
# ---------------------------------------------------------------------------

def lively_map(rng, size=16):
    """One random map: outer wall, a few wall pieces, a few screens, landmarks.

    Thrown away and redone if some floor is walled off, there is too little
    open floor, or the map equals one of its own mirror images / rotations.
    """
    interior = (size - 2) ** 2
    while True:
        grid = np.full((size, size), FLOOR, dtype=np.int64)
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = WALL

        def drop_pieces(code, count, longest):
            for _ in range(count):
                length = rng.integers(1, longest + 1)
                y, x = rng.integers(1, size - 1, size=2)
                piece = grid[y, x:x + length] if rng.random() < 0.5 else grid[y:y + length, x]
                piece[piece == FLOOR] = code            # only ever replaces open floor

        drop_pieces(WALL, rng.integers(3, 7), longest=5)
        drop_pieces(SCREEN, rng.integers(4, 8), longest=3)

        floor = np.argwhere(grid == FLOOR)
        if len(floor) < 0.7 * interior:
            continue
        picks = rng.choice(len(floor), size=len(LANDMARKS_PER_MAP), replace=False)
        grid[floor[picks, 0], floor[picks, 1]] = LANDMARKS_PER_MAP
        if _connected(~BLOCKS_AGENT[grid]) and is_asymmetric(grid):
            return grid


# ---------------------------------------------------------------------------
# The senses of everything that never moves, worked out once per map.
# (Same definitions as senses.py. Balls and rollers are added in observe().)
# ---------------------------------------------------------------------------

def vision_tables(grids, radius):
    """(maps, H, W) cell codes -> (maps, H, W, 3, size, size) colour patches."""
    padded = np.empty((grids.shape[0], grids.shape[1] + 2 * radius, grids.shape[2] + 2 * radius, 3),
                      dtype=np.float32)
    padded[:] = np.array(VOID_COLOR, dtype=np.float32)
    padded[:, radius:-radius, radius:-radius] = _COLORS[grids]
    size = 2 * radius + 1
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size), axis=(1, 2))
    return np.ascontiguousarray(windows)


def touch_tables(grids):
    """(maps, H, W) -> (maps, H, W, 5 spots * textures) one-hot textures."""
    padded = np.pad(_TEXTURE_ID[grids], ((0, 0), (1, 1), (1, 1)), constant_values=TEXTURES.index("rough"))
    H, W = grids.shape[1:]
    out = np.zeros(grids.shape + (len(TOUCH_SPOTS), len(TEXTURES)), dtype=np.float32)
    for i, (dy, dx) in enumerate(TOUCH_SPOTS):
        neighbour = padded[:, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        np.put_along_axis(out[:, :, :, i], neighbour[..., None], 1.0, axis=-1)
    return out.reshape(grids.shape + (-1,))


def hearing_tables(grids, falloff=2.0):
    """(maps, H, W) -> (maps, H, W, 3 pitches) loudness, fading with distance."""
    n_maps, H, W = grids.shape
    ys, xs = np.mgrid[0:H, 0:W]
    out = np.zeros((n_maps, H, W, 3), dtype=np.float32)
    for code, (channel, loudness) in SOUNDS.items():
        for m, y, x in np.argwhere(grids == code):
            out[m, :, :, channel] += loudness / (1.0 + (np.hypot(ys - y, xs - x) / falloff) ** 2)
    return out


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------

class LivelyWorld:
    """n_agents agents, each alone in its own random map with balls and rollers.

    Per-agent arrays (leading dimension n_agents):
        grid       (S, S)        the cell codes of the map
        pos        (2,)          the agent's (y, x)
        obj_pos    (n_objects, 2)   each mover's (y, x)
        obj_dir    (n_objects,)     its heading: 0 = still, 1..4 = N, E, S, W
                                    (the same numbers as the agent's actions)
    kind[j] says whether mover j is a BALL or a ROLLER (the same in every map).
    """

    def __init__(self, size=16, vision_radius=2, seed=0, n_balls=2, n_rollers=2,
                 p_rest=0.03, p_start=0.15):
        """
        p_rest:  chance per moment that a moving roller stops for a rest
        p_start: chance per moment that a resting roller sets off again
        """
        self.size, self.radius = size, vision_radius
        patch = 2 * vision_radius + 1
        self.sense_shapes = {"vision": (3, patch, patch),
                             "touch": (len(TOUCH_SPOTS) * len(TEXTURES),),
                             "hearing": (3,)}
        self.n_actions = len(ACTION_NAMES)
        self.kind = np.array([BALL] * n_balls + [ROLLER] * n_rollers)
        self.n_objects = len(self.kind)
        self.p_rest, self.p_start = p_rest, p_start
        self.rng = np.random.default_rng(seed)
        self.maps_made = 0
        # Tests only: number of variants of each scene held in this world. The
        # variants of one scene must get the same dice rolls (see _uniform).
        self.tile = 1
        self.pos = None

    # ------------------------------------------------------------------
    # The interface the brain / training loop uses (same as ManyWorlds)
    # ------------------------------------------------------------------

    def reset(self, n_agents, seed=None):
        """Give n_agents agents a fresh random map each. Returns first observations."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._allocate(n_agents)
        self._build(self._agents)
        return self.observe()

    def new_life(self, agents):
        """The listed agents start a new life in a brand-new map.
        Returns their first observations (rows = those agents only)."""
        self._build(np.asarray(agents))
        return {name: obs[agents] for name, obs in self.observe(mark_seen=False).items()}

    def step(self, actions):
        """One moment: physics, then what every agent senses."""
        self.advance(actions)
        return self.observe()

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def advance(self, actions):
        """Move the agent, then the balls, then the rollers. No sensing here."""
        A = self._agents
        actions = np.asarray(actions)
        self.moved[:] = False                     # nothing moves twice in one moment

        # SPEED: `taken` marks every cell a mover cannot enter right now: walls,
        # bells and fountains, other movers, and the agent. Looking a cell up
        # in it is much cheaper than comparing against every mover's position.
        # It is kept up to date as things move (see _shift).
        taken = self.blocks_objects.copy()
        taken[A, self.pos[:, 0], self.pos[:, 1]] = True
        for j in range(self.n_objects):
            rows = A[self.alive[:, j]]
            taken[rows, self.obj_pos[rows, j, 0], self.obj_pos[rows, j, 1]] = True
        self.taken = taken

        # 1. The agent. Walking into a ball pushes it (if the ball has room).
        target = self.pos + ACTION_MOVES[actions]
        wants = actions != STAY
        ty, tx = target[:, 0], target[:, 1]
        pushed = self._bump(target, actions, wants & taken[A, ty, tx])
        walks = wants & ~self.blocks_agent[A, ty, tx] & (~taken[A, ty, tx] | pushed)
        self._shift(A[walks], self.pos[walks], target[walks])
        self.pos = np.where(walks[:, None], target, self.pos)

        # 2. and 3. The movers, one at a time.
        for j in range(self.n_objects):
            if self.kind[j] == BALL:
                self._roll_ball(j)
            else:
                self._move_roller(j)

    def _shift(self, rows, old, new):
        """Record in `taken` that something moved from cell `old` to cell `new`."""
        self.taken[rows, old[:, 0], old[:, 1]] = False
        self.taken[rows, new[:, 0], new[:, 1]] = True

    def _bump(self, cells, direction, candidates):
        """Something moving in `direction` has found the cell ahead taken.

        If what is there is a ball, and the cell behind the ball is free, the
        ball is knocked one cell on and keeps rolling that way. Returns, per
        agent, whether that happened (so the cell ahead is now free). Rows
        where `candidates` is False are ignored.
        """
        if not candidates.any():
            return candidates
        A = self._agents
        same = (self.obj_pos == cells[:, None, :]).all(axis=2) & self.alive      # which mover is there?
        which = same.argmax(axis=1)
        beyond = self.obj_pos[A, which] + ACTION_MOVES[direction]
        ok = (candidates & same.any(axis=1) & (self.kind[which] == BALL) & ~self.stuck[A, which]
              & ~self.taken[A, beyond[:, 0], beyond[:, 1]])
        rows, balls = A[ok], which[ok]
        self._shift(rows, self.obj_pos[rows, balls], beyond[ok])
        self.obj_pos[rows, balls] = beyond[ok]
        self.obj_dir[rows, balls] = direction[ok]
        self.moved[rows, balls] = True
        return ok

    def _roll_ball(self, j):
        """A rolling ball moves one cell on; if blocked it stops for good.
        If what blocks it is another ball, that ball is knocked on instead."""
        A = self._agents
        heading = self.obj_dir[:, j].copy()
        rolling = self.alive[:, j] & (heading != STAY) & ~self.moved[:, j]
        if not rolling.any():
            return
        ahead = self.obj_pos[:, j] + ACTION_MOVES[heading]
        blocked = self.taken[A, ahead[:, 0], ahead[:, 1]]
        goes = rolling & ~blocked
        self._bump(ahead, heading, rolling & blocked)
        self._shift(A[goes], self.obj_pos[goes, j], ahead[goes])
        self.obj_pos[goes, j] = ahead[goes]
        self.obj_dir[rolling & ~goes, j] = STAY

    def _move_roller(self, j):
        """A roller keeps going straight; turns when blocked; sometimes rests."""
        A = self._agents
        heading = self.obj_dir[:, j].copy()
        here = self.obj_pos[:, j]
        dice, turn_dice = self._uniform(), self._uniform(4)
        alive = self.alive[:, j]
        under_screen = self.grid[A, here[:, 0], here[:, 1]] == SCREEN
        rests = alive & (heading != STAY) & ~under_screen & (dice < self.p_rest)
        wakes = alive & (heading == STAY) & (dice < self.p_start)
        moving = alive & (heading != STAY) & ~rests & ~self.moved[:, j]

        ahead = here + ACTION_MOVES[heading]
        ay, ax = ahead[:, 0], ahead[:, 1]
        self._bump(ahead, heading, moving & self.taken[A, ay, ax])     # a ball in the way gets pushed
        goes = moving & ~self.taken[A, ay, ax]
        self._shift(A[goes], here[goes], ahead[goes])
        self.obj_pos[goes, j] = ahead[goes]

        # Blocked or waking up: pick a new heading among the free neighbours
        # (it sets off at the next moment). No free neighbour: stay put.
        self.obj_dir[rests, j] = STAY
        rows = A[(moving & ~goes) | wakes]
        if len(rows):
            y, x = self.obj_pos[rows, j, 0], self.obj_pos[rows, j, 1]
            free = np.stack([~self.taken[rows, y + dy, x + dx] for dy, dx in ACTION_MOVES[1:]], axis=1)
            choice = (turn_dice[rows] * free).argmax(axis=1) + 1
            self.obj_dir[rows, j] = np.where(free.any(axis=1), choice, STAY)

    def _uniform(self, *shape):
        """Random numbers in [0, 1), one row per agent.
        (In the tests, the variants of one scene share their dice rolls.)"""
        n = len(self._agents) // self.tile
        return np.tile(self.rng.random((n,) + shape), (self.tile,) + (1,) * len(shape))

    # ------------------------------------------------------------------
    # Sensing
    # ------------------------------------------------------------------

    def observe(self, mark_seen=True):
        """What each agent senses right now: the fixed scenery from the
        per-map tables, with the balls and rollers painted on top."""
        A = self._agents
        y, x = self.pos[:, 0], self.pos[:, 1]
        obs = {name: table[A, y, x] for name, table in self.tables.items()}
        touch = obs["touch"].reshape(len(A), len(TOUCH_SPOTS), len(TEXTURES))     # a view, not a copy
        patch = 2 * self.radius + 1

        for j in range(self.n_objects):
            oy, ox = self.obj_pos[:, j, 0], self.obj_pos[:, j, 1]
            # Under a screen a mover can be neither seen nor felt.
            shown = self.alive[:, j] & ~self.invisible[:, j] & (self.grid[A, oy, ox] != SCREEN)
            row, col = oy - y + self.radius, ox - x + self.radius
            in_view = shown & (row >= 0) & (row < patch) & (col >= 0) & (col < patch)
            obs["vision"][in_view, :, row[in_view], col[in_view]] = MOVER_COLORS[self.kind[j]]
            dy, dx = oy - y, ox - x
            if (shown & (np.abs(dy) + np.abs(dx) == 1)).any():        # is any agent right next to it?
                for spot, (sy, sx) in enumerate(TOUCH_SPOTS):
                    touching = shown & (dy == sy) & (dx == sx)
                    touch[touching, spot] = 0.0
                    touch[touching, spot, _MOVER_TEXTURE_ID[self.kind[j]]] = 1.0

        if mark_seen:        # bookkeeping for the tests only: which cells have been looked at
            offsets = np.arange(-self.radius, self.radius + 1)
            ys = np.clip(y[:, None, None] + offsets[None, :, None], 0, self.size - 1)
            xs = np.clip(x[:, None, None] + offsets[None, None, :], 0, self.size - 1)
            self.seen[A[:, None, None], ys, xs] = True
        return obs

    def ground_truth(self):
        """True hidden facts, FOR THE TESTS ONLY (never shown to the brain)."""
        walkable = ~self.blocks_agent
        as_xy = lambda yx: yx[..., ::-1].astype(np.float32)             # (y, x) -> (x, y)
        return {
            "position": as_xy(self.pos),
            "bell": as_xy(self.landmarks["bell"]),
            "fountain": as_xy(self.landmarks["fountain"]),
            "ball": as_xy(self.obj_pos[:, 0]),                          # the first ball
            "coverage": ((self.seen & walkable).sum(axis=(1, 2)) / walkable.sum(axis=(1, 2))).astype(np.float32),
        }

    # ------------------------------------------------------------------
    # Building maps
    # ------------------------------------------------------------------

    def _allocate(self, n_agents):
        S, n = self.size, self.n_objects
        self._agents = np.arange(n_agents)
        self.grid = np.zeros((n_agents, S, S), dtype=np.int64)
        self.blocks_agent = np.zeros((n_agents, S, S), dtype=bool)
        self.blocks_objects = np.zeros((n_agents, S, S), dtype=bool)
        self.tables = {name: np.zeros((n_agents, S, S) + shape, dtype=np.float32)
                       for name, shape in self.sense_shapes.items()}
        self.landmarks = {name: np.zeros((n_agents, 2), dtype=np.int64) for name in ("bell", "fountain")}
        self.seen = np.zeros((n_agents, S, S), dtype=bool)
        self.pos = np.zeros((n_agents, 2), dtype=np.int64)
        self.obj_pos = np.zeros((n_agents, n, 2), dtype=np.int64)
        self.obj_dir = np.zeros((n_agents, n), dtype=np.int64)
        self.moved = np.zeros((n_agents, n), dtype=bool)
        # Switches used only by the tests to stage impossible events:
        self.alive = np.ones((n_agents, n), dtype=bool)         # False: this mover does not exist
        self.invisible = np.zeros((n_agents, n), dtype=bool)    # True: exists but cannot be sensed
        self.stuck = np.zeros((n_agents, n), dtype=bool)        # True: a ball that refuses to be pushed

    def set_maps(self, agents, grids):
        """Install these maps for these agents and work out their sense tables."""
        self.grid[agents] = grids
        self.blocks_agent[agents] = BLOCKS_AGENT[grids]
        self.blocks_objects[agents] = BLOCKS_OBJECTS[grids]
        self.tables["vision"][agents] = vision_tables(grids, self.radius)
        self.tables["touch"][agents] = touch_tables(grids)
        self.tables["hearing"][agents] = hearing_tables(grids)
        self.seen[agents] = False
        self.alive[agents], self.invisible[agents], self.stuck[agents] = True, False, False
        self.obj_dir[agents] = STAY

    def _build(self, agents):
        """New random maps for these agents; agent and movers on random free floor."""
        grids = np.stack([lively_map(self.rng, self.size) for _ in agents])
        self.set_maps(agents, grids)
        for i, grid in zip(agents, grids):
            for name, code in (("bell", BELL), ("fountain", FOUNTAIN)):
                self.landmarks[name][i] = np.argwhere(grid == code)[0]
            floor = np.argwhere(grid == FLOOR)
            picks = floor[self.rng.choice(len(floor), size=self.n_objects + 1, replace=False)]
            self.pos[i], self.obj_pos[i] = picks[0], picks[1:]
        self.maps_made += len(agents)

    def render_ascii(self, agent=0):
        """A text picture of one agent's map ('@' is the agent)."""
        rows = [[CELL_SYMBOLS[c] for c in row] for row in self.grid[agent]]
        for j in range(self.n_objects):
            if self.alive[agent, j]:
                y, x = self.obj_pos[agent, j]
                under = rows[y][x] == "S"
                rows[y][x] = MOVER_SYMBOLS[self.kind[j]].upper() if under else MOVER_SYMBOLS[self.kind[j]]
        y, x = self.pos[agent]
        rows[y][x] = "@"
        return "\n".join("".join(row) for row in rows)
