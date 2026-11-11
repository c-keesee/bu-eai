"""
commonsense_test.py -- v0.3: common-sense tests, done the way psychologists
test babies.

THE METHOD ("violation of expectation"). A baby cannot tell you what it
expects, but it looks longer at things that surprise it. So you show it a
possible event and an impossible one and compare the looking times.

Our brain has a built-in surprise meter: its prediction error. At every
moment it predicts what it will sense next; how wrong that turns out to be is
how surprised it was. So each test here stages a short scene in a few
versions, some possible and some impossible, and compares the surprise.

HOW A SCENE IS STAGED. The agent stands in the middle of a cleared patch of
an ordinary random map and watches (or, in one test, pushes). The first
moments are quiet so its mind can settle. Then the event happens. All
versions of one scene use the same map, the same spot and the same dice, and
are identical up to the moment the event differs.

POSSIBLE versions are just the simulator running. IMPOSSIBLE versions break a
rule on purpose with one "magic" edit of the world (a roller deleted while it
is under a screen, a wall that lets a ball through, ...).

MATCHED SETS (borrowed from the IntPhys benchmark). Wherever it can be done,
a test has four versions arranged so that every outcome appears once as
possible and once as impossible. Example, object permanence:

    roller goes in,  roller comes out      possible
    roller goes in,  nothing comes out     IMPOSSIBLE
    nothing goes in, nothing comes out     possible
    nothing goes in, roller comes out      IMPOSSIBLE

A brain that is simply startled whenever a roller appears is "surprised" by
one possible and one impossible version, and scores 50%. Only a brain that
connects what went in with what should come out can do better.

THE SCORE. Each version has a weight: +1 if impossible, -1 if possible. For
each scene we add up weight x surprise. The scene is passed if the sum is
above zero (more surprise at the impossible versions). The score is the share
of scenes passed; 50% is chance. Next to it we report HOW MUCH more surprising
the impossible versions were, as a ratio (1.0 = no difference). Both matter: a
brain can be "more surprised" in every scene by a hair's breadth.

We also score each MATCHED PAIR on its own: an impossible version against the
possible version that ends the same way ("nothing comes out" after a roller
went in, against "nothing comes out" when nothing went in). What is on screen
is the same; only the history differs. This shows which half of a principle
the brain has.

BASELINES, given exactly the same scenes:
    * the same network before training
    * "expect no change": a brain-free yardstick whose prediction is simply
      the last thing it saw. It shows how much of a result could come from
      the raw amount of change on screen.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .lively_world import BALL, FLOOR, BELL, FOUNTAIN, ROLLER, SCREEN, STAY, WALL, LivelyWorld, lively_map

EAST = (1, 0)        # scenes are written for "things travel east"; each scene is then turned at random
HABITUATION = 8      # quiet moments before anything happens
STAGE_RADIUS = 3     # the cleared patch is 7x7 cells around the agent


def at(x):
    """Observation number at which a mover that sets off from x = -3 at the
    end of the quiet period is at position x (one cell per moment)."""
    return HABITUATION + 3 + x


@dataclass
class Version:
    """One version of a scene. Positions are (dx, dy) from the agent; east is +x, north is -y."""
    label: str
    possible: bool
    scenery: list = field(default_factory=list)    # (dx, dy, cell code) stamped onto the cleared patch
    movers: list = field(default_factory=list)     # (BALL or ROLLER, dx, dy); all start at rest
    actions: dict = field(default_factory=dict)    # moment -> direction the agent moves (default: stay)
    # Things done to the world just before a given observation is taken:
    #   (observation number, what, mover kind, ...)
    events: list = field(default_factory=list)
    dark: tuple = None                             # (first, last) observation numbers with vision off
    stuck: bool = False                            # magic: the ball cannot be pushed
    ghost_walls: list = field(default_factory=list)   # magic: (dx, dy) walls that movers pass through
    weight: int = None                             # default: -1 if possible, +1 if impossible

    def __post_init__(self):
        if self.weight is None:
            self.weight = -1 if self.possible else 1


@dataclass
class Test:
    name: str
    principle: str        # which piece of common sense it is about
    question: str         # one line for the report
    versions: list
    window: tuple         # (first, last) observation numbers over which surprise is added up
    # Matched pairs: (impossible version, possible version that ENDS THE SAME WAY).
    # Within a pair what is seen during the window is the same (or nearly); only
    # what happened before differs. The strictest comparison we have.
    pairs: list = field(default_factory=lambda: [(1, 2), (3, 0)])


# Events that are just the simulator being told when to start ("launch": a
# mover out of view, or a roller, sets off). Everything else is magic.
LEGAL_EVENTS = {"launch"}


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------

def _launch(kind):
    return (HABITUATION, "launch", kind, EAST)


def permanence_screen(length):
    """A roller rolls under a screen. Does the brain expect it to come out?"""
    screens = [(x, -1, SCREEN) for x in ((0,) if length == 1 else (-1, 0, 1))]
    exit_x = 1 if length == 1 else 2
    roller = [(ROLLER, -3, -1)]
    return Test(
        f"permanence_screen_{length}", "object permanence",
        f"roller passes under a screen, hidden for {length} moment{'s' if length > 1 else ''}",
        [Version("goes in, comes out", True, screens, roller, events=[_launch(ROLLER)]),
         Version("goes in, never comes out", False, screens, roller,
                 events=[_launch(ROLLER), (at(exit_x), "remove", ROLLER)]),
         Version("nothing goes in, nothing comes out", True, screens),
         Version("nothing goes in, one comes out", False, screens,
                 events=[(at(exit_x), "spawn", ROLLER, exit_x, -1, EAST)])],
        window=(at(exit_x), at(exit_x) + 1))


def permanence_dark(dark_moments):
    """A ball sits there. The lights go out for a while. Is it still there after?"""
    first, last = HABITUATION, HABITUATION + dark_moments - 1
    middle = HABITUATION + dark_moments // 2
    ball = [(BALL, 1, -1)]
    return Test(
        f"permanence_dark_{dark_moments}", "object permanence",
        f"ball in view, lights out for {dark_moments} moments",
        [Version("ball there, still there", True, movers=ball, dark=(first, last)),
         Version("ball there, gone", False, movers=ball, dark=(first, last), events=[(middle, "remove", BALL)]),
         Version("no ball, still none", True, dark=(first, last)),
         Version("no ball, one appeared", False, dark=(first, last),
                 events=[(middle, "spawn", BALL, 1, -1, None)])],
        window=(last + 1, last + 1))


def solidity():
    """A ball rolls at a wall. It should stop; it should not go through."""
    wall = [(0, -1, WALL)]
    ball = [(BALL, -3, -1)]
    go = [_launch(BALL)]
    return Test(
        "solidity", "solidity", "ball rolls towards a wall (or towards nothing)",
        [Version("wall: ball stops", True, wall, ball, events=go),
         Version("wall: ball goes through", False, wall, ball, events=go, ghost_walls=[(0, -1)]),
         Version("no wall: ball keeps rolling", True, [], ball, events=go),
         Version("no wall: ball stops dead", False, [], ball, events=go + [(at(-1), "halt", BALL)])],
        window=(at(0), at(1)))


def causality_watch():
    """Watching: a roller runs into a resting ball (or rolls past it)."""
    ball = (BALL, 0, -1)
    hit, miss = [ball, (ROLLER, -3, -1)], [ball, (ROLLER, -3, -2)]
    go = [_launch(ROLLER)]
    return Test(
        "causality_watch", "causality", "watching a roller hit a ball, or roll past it",
        [Version("hit: ball moves", True, movers=hit, events=go),
         Version("hit: ball does not move", False, movers=hit, events=go, stuck=True),
         Version("no hit: ball stays", True, movers=miss, events=go),
         Version("no hit: ball moves anyway", False, movers=miss,
                 events=go + [(at(0), "shove", BALL, 1, -1, EAST)])],
        window=(at(0), at(1)))


def causality_act():
    """Acting: the agent itself pushes the ball (or stands still)."""
    ball = [(BALL, 1, 0)]
    push = {HABITUATION: EAST}
    return Test(
        "causality_act", "causality", "the agent pushes a ball itself, or stands still",
        [Version("I push: ball moves", True, movers=ball, actions=push),
         Version("I push: ball does not move", False, movers=ball, actions=push, stuck=True),
         Version("I do nothing: ball stays", True, movers=ball),
         Version("I do nothing: ball moves anyway", False, movers=ball,
                 events=[(HABITUATION + 1, "shove", BALL, 2, 0, EAST)])],
        window=(HABITUATION + 1, HABITUATION + 2))


def self_starting():
    """A resting thing sets off by itself. Normal for a roller, impossible for a ball.
    The weights ask: is a self-starting ball MORE surprising than a self-starting roller?"""
    return Test(
        "self_starting", "causality", "a resting ball / roller sets off by itself",
        [Version("ball sets off by itself", False, movers=[(BALL, 0, -1)],
                 events=[(HABITUATION, "self_start", BALL, EAST)]),
         Version("ball stays", True, movers=[(BALL, 0, -1)]),
         Version("roller sets off by itself", True, movers=[(ROLLER, 0, -1)], events=[_launch(ROLLER)]),
         Version("roller stays", True, movers=[(ROLLER, 0, -1)], weight=1)],
        window=(HABITUATION + 1, HABITUATION + 2), pairs=[(0, 2)])


def continuity_jump():
    """A roller in plain view vanishes and reappears somewhere else."""
    roller = [(ROLLER, -3, -1)]
    return Test(
        "continuity_jump", "continuity", "roller in plain view jumps to a far-away cell",
        [Version("rolls on normally", True, movers=roller, events=[_launch(ROLLER)]),
         Version("jumps to a far cell", False, movers=roller,
                 events=[_launch(ROLLER), (at(0), "teleport", ROLLER, 2, 1)])],
        window=(at(0), at(1)), pairs=[(1, 0)])


def continuity_screens():
    """Two screens with a gap. A roller must be seen crossing the gap."""
    screens = [(-1, -1, SCREEN), (1, -1, SCREEN)]
    roller = [(ROLLER, -3, -1)]
    return Test(
        "continuity_screens", "continuity", "roller passes two screens with a gap between them",
        [Version("seen crossing the gap", True, screens, roller, events=[_launch(ROLLER)]),
         Version("skips the gap", False, screens, roller,
                 events=[_launch(ROLLER), (at(0), "hide", ROLLER), (at(1), "show", ROLLER)])],
        window=(at(0), at(2)), pairs=[(1, 0)])


def all_tests():
    return [permanence_screen(1), permanence_screen(3), permanence_dark(5), permanence_dark(30),
            solidity(), causality_watch(), causality_act(), self_starting(),
            continuity_jump(), continuity_screens()]


# ---------------------------------------------------------------------------
# Staging: build the maps and apply the scripted events
# ---------------------------------------------------------------------------

class Stage:
    """n_scenes scenes x all versions of one test, in one LivelyWorld.

    Agent number (v * n_scenes + s) plays version v of scene s.
    Each scene has its own random map, spot and orientation.
    """

    def __init__(self, test, n_scenes, seed, size=16):
        self.test, self.n = test, n_scenes
        rng = np.random.default_rng(seed)
        V = len(test.versions)
        world = LivelyWorld(size=size, seed=seed, p_rest=0.0, p_start=0.0)   # nothing happens unscripted
        world._allocate(V * n_scenes)
        world.tile = V                                # versions of a scene share their dice
        self.world = world
        self.slot = {kind: int(np.argmax(world.kind == kind)) for kind in (BALL, ROLLER)}

        # A random map per scene with a cleared 7x7 patch somewhere inside it.
        low, high = STAGE_RADIUS + 1, size - STAGE_RADIUS - 2
        base, centre = [], []
        while len(base) < n_scenes:
            grid = lively_map(rng, size)
            cy, cx = rng.integers(low, high + 1, size=2)
            patch = grid[cy - STAGE_RADIUS:cy + STAGE_RADIUS + 1, cx - STAGE_RADIUS:cx + STAGE_RADIUS + 1]
            if np.isin(patch, (BELL, FOUNTAIN)).any():
                continue                              # do not delete a landmark; try another map
            patch[:] = FLOOR
            base.append(grid)
            centre.append((cy, cx))
        self.centre = np.array(centre)
        self.turns = rng.integers(0, 4, size=n_scenes)        # quarter turns
        self.mirror = rng.random(n_scenes) < 0.5

        grids = np.stack(base)
        scenes = np.arange(n_scenes)
        for v, version in enumerate(test.versions):
            agents = v * n_scenes + scenes
            own = grids.copy()
            for dx, dy, code in version.scenery:
                cell = self.cells(dx, dy)
                own[scenes, cell[:, 0], cell[:, 1]] = code
            world.set_maps(agents, own)
            world.pos[agents] = self.centre
            world.alive[agents] = False
            for kind, dx, dy in version.movers:
                world.alive[agents, self.slot[kind]] = True
                world.obj_pos[agents, self.slot[kind]] = self.cells(dx, dy)
            world.stuck[agents, self.slot[BALL]] = version.stuck
            for dx, dy in version.ghost_walls:
                cell = self.cells(dx, dy)
                world.blocks_objects[agents, cell[:, 0], cell[:, 1]] = False

    # Turning a scene: (dx, dy) as written -> the real offset in each scene's map.
    def _turned(self, dx, dy):
        dx, dy = np.full(self.n, dx), np.where(self.mirror, -dy, dy)
        for quarter in range(1, 4):
            turn = self.turns >= quarter
            dx, dy = np.where(turn, -dy, dx), np.where(turn, dx, dy)      # a quarter turn clockwise
        return dx, dy

    def cells(self, dx, dy):
        """(dx, dy) from the agent -> absolute (y, x), one row per scene."""
        dx, dy = self._turned(dx, dy)
        return np.stack([self.centre[:, 0] + dy, self.centre[:, 1] + dx], axis=1)

    def heading(self, direction):
        """A direction as written -> the action/heading number in each scene (1..4 = N, E, S, W)."""
        if direction is None:
            return np.zeros(self.n, dtype=np.int64)
        dx, dy = self._turned(*direction)
        return np.where(dy == -1, 1, np.where(dx == 1, 2, np.where(dy == 1, 3, 4)))

    def actions(self, moment):
        """The scripted action of every agent at this moment."""
        out = np.zeros(len(self.test.versions) * self.n, dtype=np.int64)
        for v, version in enumerate(self.test.versions):
            if moment in version.actions:
                out[v * self.n:(v + 1) * self.n] = self.heading(version.actions[moment])
        return out

    def lit(self, observation):
        """1.0 where vision works at this observation, 0.0 where it is dark."""
        out = np.ones(len(self.test.versions) * self.n, dtype=np.float32)
        for v, version in enumerate(self.test.versions):
            if version.dark and version.dark[0] <= observation <= version.dark[1]:
                out[v * self.n:(v + 1) * self.n] = 0.0
        return torch.from_numpy(out)

    def apply_events(self, observation):
        """Do everything scripted for just before this observation is taken."""
        w = self.world
        for v, version in enumerate(self.test.versions):
            agents = v * self.n + np.arange(self.n)
            for when, what, kind, *args in version.events:
                if when != observation:
                    continue
                j = self.slot[kind]
                if what in ("launch", "self_start"):    # set off (moves from the next moment on);
                    # "self_start" is the same thing done to a ball in plain view, which is magic
                    w.obj_dir[agents, j] = self.heading(args[0])
                elif what == "halt":                    # magic: stop dead
                    w.obj_dir[agents, j] = STAY
                elif what == "remove":                  # magic: cease to exist
                    w.alive[agents, j] = False
                elif what in ("spawn", "shove"):        # magic: appear / be moved by nothing
                    w.alive[agents, j] = True
                    w.obj_pos[agents, j] = self.cells(args[0], args[1])
                    w.obj_dir[agents, j] = self.heading(args[2])
                elif what == "teleport":                # magic: be somewhere else
                    w.obj_pos[agents, j] = self.cells(args[0], args[1])
                elif what in ("hide", "show"):          # magic: exist but be unseeable for a moment
                    w.invisible[agents, j] = what == "hide"
                else:
                    raise ValueError(what)


# ---------------------------------------------------------------------------
# Running a test
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_test(brain, test, n_scenes=200, seed=0, memoryless=False):
    """Show every version of n_scenes scenes to a frozen brain.

    Returns surprise for the brain and for the "expect no change" yardstick,
    each an array (moments, versions, scenes). Entry [t] is the surprise at
    observation t + 1 (the prediction was made at moment t).
    """
    stage = Stage(test, n_scenes, seed)
    world = stage.world
    n_agents, n_moments = len(test.versions) * n_scenes, test.window[1]
    one_hot = lambda a: F.one_hot(torch.from_numpy(a), world.n_actions).float()
    to_torch = lambda obs: {name: torch.from_numpy(o) for name, o in obs.items()}

    obs = to_torch(world.observe())
    core = brain.initial_state(n_agents)
    last_action = np.zeros(n_agents, dtype=np.int64)
    last_seen = {name: o.clone() for name, o in obs.items()}
    surprise, yardstick = [], []
    for t in range(n_moments):
        # Sense (vision may be dark), update the mind, act, predict.
        lit = stage.lit(t)
        inputs = dict(obs, vision=obs["vision"] * lit.reshape(-1, 1, 1, 1))
        seen = lit.reshape(-1, 1, 1, 1) > 0
        last_seen = dict(obs, vision=torch.where(seen, obs["vision"], last_seen["vision"]))
        if memoryless:
            core = brain.initial_state(n_agents)
        core = brain.perceive(inputs, one_hot(last_action), core)
        action = stage.actions(t)
        prediction = brain.predict(core, one_hot(action))

        # The world moves on (with any scripted magic), and we see what happened.
        world.advance(action)
        stage.apply_events(t + 1)
        obs = to_torch(world.observe())
        error = lambda guess: sum(((guess[name] - obs[name]) ** 2).flatten(1).mean(dim=1) for name in obs)
        surprise.append(error(prediction))
        yardstick.append(error(last_seen))
        last_action = action

    shape = (n_moments, len(test.versions), n_scenes)
    return torch.stack(surprise).numpy().reshape(shape), torch.stack(yardstick).numpy().reshape(shape)


def score(test, surprise):
    """Turn surprise (moments, versions, scenes) into the test's numbers."""
    first, last = test.window
    in_window = surprise[first - 1:last].sum(axis=0)                    # (versions, scenes)
    weights = np.array([v.weight for v in test.versions])[:, None]
    contrast = (weights * in_window).sum(axis=0)                        # per scene; > 0 = pass
    possible = np.array([v.possible for v in test.versions])
    rate = lambda difference: float((difference > 0).mean() + 0.5 * (difference == 0).mean())   # ties count half
    mean = in_window.mean(axis=1)                                       # per version
    ratio = lambda up, down: float(up / max(down, 1e-12))
    return {
        "pass_rate": rate(contrast),
        "pair_rates": [rate(in_window[i] - in_window[p]) for i, p in test.pairs],
        # HOW MUCH more surprising: mean surprise at the +1 versions divided by
        # mean surprise at the -1 versions. 1.0 = no difference at all. A pass
        # rate can be high while this is 1.01, which would mean "consistently,
        # but by a hair": not what we are looking for.
        "ratio": ratio(mean[weights[:, 0] > 0].sum(), mean[weights[:, 0] < 0].sum()),
        "pair_ratios": [ratio(mean[i], mean[p]) for i, p in test.pairs],
        "surprise": [float(x) for x in in_window.mean(axis=1)],         # mean per version, in the window
        "possible_mean": float(in_window[possible].mean()),
        "impossible_mean": float(in_window[~possible].mean()),
        "course": surprise.mean(axis=2).T.tolist(),                     # (versions, moments) for plotting
    }


def commonsense_tests(brain, untrained=None, n_scenes=200, seed=0, memoryless=False):
    """Run every test. Returns {test name: {"trained": ..., "untrained": ..., "no change": ..., info}}."""
    out = {}
    for test in all_tests():
        trained, yardstick = run_test(brain, test, n_scenes, seed, memoryless)
        result = {
            "principle": test.principle, "question": test.question, "window": list(test.window),
            "versions": [v.label for v in test.versions],
            "pairs": [[test.versions[i].label, test.versions[p].label] for i, p in test.pairs],
            "possible": [v.possible for v in test.versions],
            "trained": score(test, trained),
            "no change": score(test, yardstick),
        }
        if untrained is not None:
            result["untrained"] = score(test, run_test(untrained, test, n_scenes, seed, memoryless)[0])
        out[test.name] = result
    return out
