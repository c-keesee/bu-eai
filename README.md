# BU-EAI — Bottom-Up Emergent AI (v0.1)

A first working prototype of the BU-EAI thesis: instead of training a model
top-down on a static dataset, grow a mind from accumulated simulated
experience.

An agent wanders a small 2D world with three senses. Its brain is built from
separate sensory regions that feed a shared recurrent core, and the core feeds
back into the regions. It has **no language, no reward, no labels and no
goal**. The only thing it ever learns is to predict what it will sense next.

Afterwards we freeze it and ask: did it work out things nobody told it, such
as where it is?

**Short answer for v0.1: yes, for position.** With vision switched off, a
linear read-out of the trained core locates the agent to within 0.40 grid
cells, against 1.35 for the same network before training and 2.42 for the raw
senses. The result is the same in all 3 seeds. Several other things I tested
did *not* show emergence, and those are listed just as plainly below.

## Run it

```bash
pip install -r requirements.txt
```

```bash
python run_experiment.py
```

That trains 3 seeds side by side (one CPU core each), runs both tests, prints
the tables below and writes three plots plus `results.json` into `results/`.
It took 2.5 minutes on a 4-core cloud machine. Results are deterministic: the
same seeds give the same numbers.

Other ways to run it:

```bash
python run_experiment.py --quick
```

```bash
python run_experiment.py --seeds 0 1 2 3 4 --updates 3000
```

```bash
python run_experiment.py --no-feedback --out results_no_feedback
```

```bash
python -m pytest tests/ -q
```

`--quick` is a 20-second smoke test (1 seed, short life; its numbers mean
nothing). `--no-feedback` cuts the top-down connections. The last command runs
the unit tests for the building blocks.

## The files

| File | What it is | Brain analogy |
|---|---|---|
| `bueai/world.py` | The grid world: map, objects, movement, ground truth | The outside world. Not part of the brain. |
| `bueai/senses.py` | `Vision`, `Touch`, `Hearing` | Sense organs: eye, skin, ear |
| `bueai/model.py` | `SensoryRegion`, `VisionRegion`, `Brain` | Sensory cortex areas, association area, top-down pathways |
| `bueai/training.py` | `Life` (the loop of experience) and `train()` | Living, and learning from surprise |
| `bueai/emergence_test.py` | Linear probes on the frozen core | A neuroscientist recording from the brain |
| `bueai/darkroom_test.py` | Lights-out prediction test | A blindfold experiment |
| `bueai/plots.py` | The three figures | |
| `run_experiment.py` | Runs everything, prints the report | |
| `tests/test_sanity.py` | Unit tests | |

## The world

```
############        #  wall       grey,  feels rough, silent
#......#...#        .  floor      dark
#.B....#.m.#        B  bell       red,   feels smooth, high pitch, loud,  solid
#..........#        F  fountain   blue,  feels wet,    mid pitch,  medium, solid
#....##....#        m  moss       green, feels soft,   low pitch,  quiet, walkable
#.....#..F.#
#.m...#....#
#..........#
###..m...#.#
#.....B..#.#
#F.......#.#
############
```

Every object type shows up in all three senses, and nothing tells the brain
that "red", "smooth" and "high-pitched" belong together. The map is
deliberately lopsided (no mirror image or rotation of it matches itself) and
every area has its own landmark, so "where am I?" has an answer.

The senses:

- **Vision**: a 5x5 patch of RGB colours centred on the agent (75 numbers).
- **Touch**: the texture under the agent and on its four sides (25 numbers).
- **Hearing**: loudness of each of the 3 pitches, fading with distance (3 numbers). No direction.

Actions are stay / north / east / south / west, chosen at random with some
momentum.

## The brain

```
               +-----------------------------------------+
               |      top-down feedback (previous state) |
               v                                         |
vision  --> [vision region ] --+                         |
touch   --> [touch region  ] --+--> [ recurrent core ] --+--> [predictors] --> next vision
hearing --> [hearing region] --+        ^         |                        --> next touch
                                        |         +-- + action ------------^   next hearing
                          last action --+
```

One moment of experience, as coded in `Life.live()`:

1. **Sense.** Each sense organ reports its numbers.
2. **Update the mind.** Each sensory region encodes its own sense, together with the core's previous state (the top-down feedback). The core, a GRU, combines the three encodings with its own memory and the last action taken.
3. **Act.** A random move.
4. **Predict.** From the core state and the chosen action, one predictor per sense guesses the next sensation.
5. **Compare.** The world moves; the squared difference between guess and reality is the prediction error.

`train()` lowers that error by gradient descent. That is the entire learning
signal. `world.ground_truth()` (true position and so on) is only ever read by
the tests.

## Results (seeds 0, 1, 2)

Each cell is the mean, followed by the value from each seed. Every agent lived
48,000 moments.

### Prediction got better

| Sense | Error at start of life | Error at end |
|---|---|---|
| vision | 0.0536 | 0.0046 |
| touch | 0.0796 | 0.0041 |
| hearing | 0.0330 | 0.0002 |

![training loss](results/training_loss.png)

The curves were still falling when training stopped, so a longer life would
likely improve every number below.

### Emergence test: where am I?

Position error in grid cells, lower is better. A linear probe is fitted on one
stretch of life and scored on a separate stretch it has never seen.

| Probe reads | Lights on | In darkness | Dark and silent |
|---|---|---|---|
| **trained core** | 0.34 [0.36 0.33 0.32] | **0.40** [0.42 0.39 0.38] | **0.61** [0.67 0.64 0.52] |
| untrained core | 1.02 [1.05 0.95 1.05] | 1.35 [1.31 1.39 1.35] | 2.28 [2.22 2.26 2.35] |
| raw senses | 0.33 [0.30 0.34 0.35] | 2.42 [2.44 2.39 2.43] | 2.95 [2.89 2.97 2.99] |
| chance | 4.27 | 4.27 | 4.27 |

"In darkness" blanks vision. "Dark and silent" blanks vision and hearing, so
only touch and the agent's own movements are left.

![probe results](results/probe_results.png)

How to read this:

- **Lights on shows nothing.** The raw senses, with no brain at all, find
  position as well as the trained core does (0.33 vs 0.34). The world is small
  and a 5x5 view plus three loudness values is nearly a fingerprint of each
  cell. Position is handed to the agent for free here, so this column cannot
  tell us whether anything emerged.
- **Darkness is where it shows.** With vision off, the raw senses fall to 2.42
  cells and the untrained network to 1.35. The trained core stays at 0.40. It
  beats both baselines in 3 of 3 seeds, by a wide margin.
- **Dark and silent is the hardest case.** Touch mostly says "nothing here".
  The trained core still knows where it is to 0.61 cells, and that error barely
  grows over 100 steps (top-right panel of the dark-room figure). To do that it
  has to add up its own movements on a remembered layout. Nobody programmed
  that; it is the main thing that emerged.

### Emergence test: which object is nearest?

Accuracy, higher is better.

| Probe reads | Lights on | In darkness | Dark and silent |
|---|---|---|---|
| trained core | 1.00 | 0.99 | **0.91** [0.91 0.91 0.93] |
| untrained core | 0.99 | 0.97 | 0.75 [0.77 0.77 0.72] |
| raw senses | 1.00 | 0.93 | 0.63 [0.62 0.63 0.63] |
| chance | 0.42 | 0.42 | 0.42 |

This probe is **mostly uninformative**. While hearing works, loudness gives the
answer away and every baseline scores 0.93 or higher. Only when both vision and
hearing are off does the trained core separate from the baselines (0.91 vs
0.75).

### Dark-room test

Mean squared prediction error over the 100 moments after vision is blanked.

| Predicting | Lights on | Lights out | Guess "no change" | Guess the average |
|---|---|---|---|---|
| touch | 0.0030 | 0.0068 | 0.0377 | 0.0785 |
| hearing | 0.0001 | 0.0003 | 0.0029 | 0.0279 |
| vision (imagined) | 0.0045 | 0.0119 | 0.0254 | 0.0458 |

![dark room](results/darkroom.png)

- Touch and hearing predictions get about 2 to 3 times worse when the lights
  go out, then **stay flat** for 100 steps. They remain 5 to 10 times better
  than guessing "nothing will change".
- "Vision (imagined)" is the brain's guess at the image it would see, scored
  against the true image it is not being shown. It is about twice as good as
  the "no change" guess. In a loose sense the brain keeps picturing the room.

### Does the top-down feedback matter?

I ran the whole experiment again with the feedback connections cut
(`--no-feedback`).

| Position error | Lights on | In darkness | Dark and silent |
|---|---|---|---|
| with feedback | 0.34 | 0.40 [0.42 0.39 0.38] | 0.61 |
| feedback cut | 0.36 | 0.46 [0.48 0.49 0.42] | 0.63 |

**At this scale, hardly.** There is a small edge in darkness, but with 3 seeds
whose ranges touch, I would not claim it. The recurrent core already carries
memory forward by itself, and the inputs are too simple for a region to need
context to interpret them. This part of the thesis is not yet supported or
refuted; see v0.2.

## What these results do not show

- **The map is in the weights, not learned on the fly.** There is one fixed
  world and the agent lived in it for 48,000 moments (times 64 parallel
  copies). The weights memorised the layout; the core state tracks position on
  it. v0.1 says nothing about how fast the brain could learn a *new* room.
- **The agent practised being blind.** During training, vision and hearing
  each switch off about a quarter of the time ("nights"). So coping with
  darkness is trained behaviour, not a surprise. What was never given is any
  coordinate or position label; that part is emergent.
- **Untrained networks are a strong baseline.** A random recurrent network
  already gets 1.02 cells with the lights on. Comparing only against "chance"
  (4.27) would have made the result look far more impressive than it is.
- **A linear probe finding position does not mean the brain "uses" position.**
  It shows the information is present in an easily readable form.
- **Three seeds on one map.** Consistent, but a small sample.

## Decisions I made where the idea was open

| Topic | Decision | Why |
|---|---|---|
| One world or many | One fixed map | "Absolute position" needs a fixed world |
| Episodes | None: one continuous life, learning online, no replay | Closest to "accumulated experience" |
| Parallel lives | 64 copies of the agent at once | CPU speed only |
| Heading | None; vision is centred on the agent but never rotated | Keeps v0.1 small |
| Motor copy | The core is told the last action it took | Without it the brain cannot track its own movement. Brains do this too ("efference copy"). |
| Blackouts in training | Vision and hearing each go off ~25% of the time | Otherwise the dark-room test measures confusion at a never-seen input |
| Movement | Random with momentum | Pure coin-flips barely leave the starting area |
| Loss | Mean squared error for all senses | One uniform meaning of "prediction error" |
| Untrained baseline | An exact copy of the same network before training | The fairest "same network untrained" |

## Moving to 3D later (v0.2 / v0.3)

The brain never mentions "grid", "vision" or any specific sense. It is built
from `world.sense_shapes`, a dictionary of names and shapes, and loops over
whatever is in it. To move to MiniWorld or MuJoCo:

1. **Write a new world class** with the same five members as `GridWorld`:
   `sense_shapes`, `n_actions`, `reset(n_agents, seed)`, `step(actions)`,
   `ground_truth()`. `reset` and `step` return a dict of NumPy arrays with one
   row per agent.
2. **Swap the vision region** for a deeper CNN. Any module with
   `forward(image, core_state)` returning `(batch, latent_size)` and a
   `latent_size` attribute works:

   ```python
   brain = Brain(world.sense_shapes, world.n_actions,
                 regions={"vision": MyBiggerCNN(...)})
   ```

`Life`, `train()`, both tests and the plots then run unchanged.
`tests/test_sanity.py` has a test that builds a brain with a 9x9 image and a
made-up "smell" sense to confirm no brain code needs to change.

Two things will need real work in 3D: continuous actions (the predictors take
a one-hot action today), and predicting raw pixels with squared error, which
gets blurry. Predicting in the region's latent space is the usual fix.

## Suggested v0.2 experiments

1. **Make position hard to see.** Use a larger map (24x24) or a 3x3 view, so
   one glance no longer identifies the cell. Then the lights-on probe becomes
   a real test too.
2. **Many worlds.** Generate a new random map for each life (or every few
   thousand moments). Then the map cannot live in the weights and the brain
   must build it within a lifetime, which is much closer to the Fluctlight idea.
3. **Give feedback a job.** Add noise or ambiguity to the senses (two objects
   that look alike but sound different), where knowing the context should help
   a region interpret its input. Rerun the `--no-feedback` comparison there.
4. **Cross-sense binding test.** Probe whether hearing a bell in the dark makes
   the brain predict *red* nearby. v0.1 has the ingredients but does not test it.
5. **Let the agent choose its actions**, for example by seeking out places
   where its predictions are worst (curiosity), instead of walking at random.
6. **Train longer and with more seeds** (`--updates 3000 --seeds 0 1 2 3 4`).
