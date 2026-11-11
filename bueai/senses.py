"""
senses.py -- the agent's sense organs.

BRAIN ANALOGY: these are the eye, the skin and the ear -- the *organs*, not
the brain regions that process them (those are in model.py). A sense organ
turns "the state of the world around my body" into a list of numbers. That
list of numbers is the only thing the brain ever receives.

Each sense is a small class with:
    name                 e.g. "vision"
    shape(world)         the shape of one observation, e.g. (3, 5, 5)
    sense(world, y, x)   what an agent standing on cell (y, x) perceives

The code here is written with plain loops for one agent at one position so it
is easy to read. (The world calls it once per cell at start-up and caches the
answers, so it does not need to be fast.)

SWAPPING LATER: in a 3D world you would not use these classes at all -- the
simulator hands you a camera image, contact sensors, and so on. The brain does
not care where the numbers come from, only what shape they have.
"""

import numpy as np


class Vision:
    """A small square patch of colours centred on the agent.

    With radius=2 the agent sees a 5x5 patch: 2 cells in every direction.
    The patch is "egocentric": the agent is always in the middle, so the same
    wall looks different depending on where the agent stands.

    Simplifications in v0.1: the agent has no facing direction (the patch is
    never rotated) and walls do not block the view.

    Output shape: (3, size, size) = (colour channel, row, column), the layout
    PyTorch's image layers expect.
    """
    name = "vision"

    def __init__(self, radius=2):
        self.radius = radius

    def shape(self, world):
        size = 2 * self.radius + 1
        return (3, size, size)

    def sense(self, world, y, x):
        # Imported here (not at the top) because world.py imports this file,
        # and two files importing each other at the top would be a cycle.
        from .world import FLOOR_COLOR, VOID_COLOR, WALL, WALL_COLOR

        size = 2 * self.radius + 1
        patch = np.zeros((3, size, size), dtype=np.float32)
        for row in range(size):
            for col in range(size):
                # World coordinates of this patch cell.
                wy, wx = y + row - self.radius, x + col - self.radius
                if not world.in_bounds(wy, wx):
                    color = VOID_COLOR
                elif world.object_at(wy, wx) is not None:
                    color = world.object_at(wy, wx).color
                elif world.grid[wy, wx] == WALL:
                    color = WALL_COLOR
                else:
                    color = FLOOR_COLOR
                patch[:, row, col] = color
        return patch


class Touch:
    """What the agent feels under its feet and on its four sides.

    Five touch spots: [underfoot, north, east, south, west].
    Each spot reports ONE texture as a "one-hot" vector: a list with a 1 at the
    texture's index and 0 everywhere else. E.g. with textures
    [nothing, rough, smooth, wet, soft], "rough" is [0, 1, 0, 0, 0].

    Output shape: (5 spots * number of textures,) = (25,)
    """
    name = "touch"
    SPOTS = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]    # (dy, dx) per spot

    def shape(self, world):
        from .world import TEXTURES
        return (len(self.SPOTS) * len(TEXTURES),)

    def sense(self, world, y, x):
        from .world import TEXTURES, WALL

        out = np.zeros((len(self.SPOTS), len(TEXTURES)), dtype=np.float32)
        for i, (dy, dx) in enumerate(self.SPOTS):
            wy, wx = y + dy, x + dx
            if world.object_at(wy, wx) is not None:
                texture = world.object_at(wy, wx).texture
            elif world.grid[wy, wx] == WALL:
                texture = "rough"
            else:
                texture = "nothing"
            out[i, TEXTURES.index(texture)] = 1.0
        return out.reshape(-1)     # flatten (5, 5) -> (25,)


class Hearing:
    """How loud each "pitch" is at the agent's position.

    Every object type emits sound on its own channel (think: its own pitch).
    Loudness falls off with distance, and sounds on the same channel add up.
    The ear cannot tell direction, only loudness per pitch.

        intensity = loudness / (1 + (distance / falloff)^2)

    so the sound is at full loudness at distance 0, half at distance `falloff`,
    and fades smoothly beyond. Walls do not block sound in v0.1.

    Output shape: (number of sound channels,) = (3,)
    """
    name = "hearing"

    def __init__(self, falloff=2.0):
        self.falloff = falloff

    def shape(self, world):
        from .world import OBJECT_TYPES
        n_channels = 1 + max(obj.sound_channel for obj in OBJECT_TYPES)
        return (n_channels,)

    def sense(self, world, y, x):
        from .world import OBJECT_TYPES

        out = np.zeros(self.shape(world), dtype=np.float32)
        for oy, ox, type_index in world.objects:
            obj = OBJECT_TYPES[type_index]
            distance = np.hypot(oy - y, ox - x)        # straight-line distance
            out[obj.sound_channel] += obj.loudness / (1.0 + (distance / self.falloff) ** 2)
        return out
