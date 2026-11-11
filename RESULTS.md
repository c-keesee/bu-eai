# BU-EAI results log

One entry per version, newest first. Every number here was produced by the
code in this repository with fixed seeds, and each cell is the mean followed by
the value from each seed `[seed 0, seed 1, seed 2]`.

- [v0.3: common sense, tested like a baby](#v03--common-sense-tested-like-a-baby-2026-11-11)
- [v0.2: many worlds](#v02--many-worlds-2026-11-11)
- [v0.1: one fixed world](#v01--one-fixed-world)

---

## v0.3 — common sense, tested like a baby (2026-11-11)

**Goal:** can the agent learn basic common sense about objects, tested the way
psychologists test babies? And does a memory help?

Reproduce with `python run_v03.py` (about 2 hours on 4 CPU cores), then
`python memory_check.py`. Raw output: `results/v0.3/report.txt`,
`results/v0.3/memory_check.txt`, `results/v0.3/runs/*.json`. The twelve trained
brains are saved in `results/v0.3/checkpoints/`; `python run_v03.py --retest`
runs every test again without training.

### Verdict

| Question | Answer | Evidence (12 brains: 4 conditions x 3 seeds) |
|---|---|---|
| Solidity: is a roller going through a wall surprising? | **Yes.** | 100% of scenes in all 12 brains; 2.1 to 2.8 times the surprise. Untrained: 0%. |
| Continuity: is a roller that jumps in plain view surprising? | **Emerging.** | 8 of 12 brains pass 75% or more of scenes. The other 4 had not learned roller motion yet when training stopped. |
| Causality: do lifeless balls and self-moving rollers differ? | **Yes, modestly.** | A ball setting off by itself is 1.8 to 1.9 times as surprising as a roller doing so. 100% in all 12. |
| Causality: "things move when I push them"? | **Half.** | Very surprised when a pushed ball does not budge (2.3 to 3.2x). But still surprised when its own push works. |
| Permanence over a short gap? | **Weakly.** | Lights out for 5 moments: 1.2 to 2.0x. One moment under a screen: 1.1 to 1.2x. |
| Permanence over a long gap? | **No.** | 3 moments under a screen: 1.00x. 30 dark moments: 1.01 to 1.09x. |
| Continuity behind two screens? | **No, the reverse.** | 0% in all 12 brains. It is 8 times MORE surprised when the roller comes out into the gap. |
| Solidity for balls (a rare event)? | **No.** | It expects a rolling ball to stop. Balls roll in under 1% of moments. |
| Does an added memory help? | **Not shown.** | No memory condition beats the plain brain on any test. The external store is used, but not for these tests. |

The honest one-line summary: **a few pieces of common sense about things it
can see have appeared. Nothing about things it cannot see has, and adding
memory did not change that.**

One thing had to be fixed first. The v0.2 brain could not learn how things
move at all, so the base brain changed (see "The brain had to change" below).

### What changed from v0.2

| | v0.2 | v0.3 |
|---|---|---|
| World | static: walls, bell, fountain, moss, TV | things move: balls, rollers, screens (no TV) |
| Brain | regions -> core | vision region with its own short-term memory -> core; local vision predictor |
| Movement | random walk + two curiosity styles | random walk only |
| Test | probes: where am I, where is the bell | violation-of-expectation tests with prediction error as surprise |
| Compared | movement styles, feedback | memory: none added / remembering pays / external store / both |
| Experience per agent | 192,000 moments | 96,000 moments |

### The world

- **Balls** are lifeless. A ball moves only when something moves into it (the
  agent, a roller, another rolling ball). Then it rolls in a straight line
  until something stops it. There is no friction.
- **Rollers** move by themselves: straight lines, a turn when blocked, an
  occasional rest, and they set off again on their own. A roller that runs
  into a ball pushes it.
- **Screens** are covered floor. Balls and rollers pass underneath and cannot
  be seen or felt while there. The agent cannot enter.
- 3 balls and 3 rollers per map. Otherwise as v0.2: a new random 16x16 map for
  every life of 512 moments, a 5x5 view, three senses, a blank mind per life.

### The tests

A baby cannot say what it expects, but it looks longer at things that surprise
it. Our brain's surprise is its prediction error. Each test stages a short
scene in several versions, some possible and some impossible, and compares the
surprise. The agent stands in a cleared patch of an ordinary random map; all
versions share the map, the spot and the dice, and are identical until the
event. Possible versions are just the simulator running. Impossible versions
break one rule with one scripted edit.

| Test | Possible | Impossible |
|---|---|---|
| Permanence, screen (1 or 3 moments hidden) | roller goes in and comes out; nothing goes in, nothing comes out | goes in, never comes out; nothing goes in, one comes out |
| Permanence, lights out (5 or 30 moments) | ball there before and after; no ball before or after | ball gone; ball appeared |
| Solidity, roller | turns away at a wall; rolls on where there is none | goes through the wall; turns away from thin air |
| Solidity, ball (rare event) | stops at a wall; rolls on where there is none | goes through; stops dead in the open |
| Causality, watching | roller hits ball and it moves; roller passes and it stays | hit but does not move; moves without being hit |
| Causality, acting | I push and it moves; I do nothing and it stays | I push and it does not move; it moves untouched |
| Causality, ball vs roller | a roller sets off by itself | a ball sets off by itself |
| Continuity, plain view | roller rolls on | roller jumps to a far cell |
| Continuity, two screens | roller is seen crossing the gap between them | goes under one and comes out of the other without crossing the gap |

**Two numbers per test.** How OFTEN the impossible versions were the more
surprising ones (share of scenes; 50% is chance), and by HOW MUCH (surprise at
the impossible versions divided by surprise at the possible ones; 1.0 is no
difference). Both are needed. In the first pilot one test read 87% while the
ratio was 1.00: "more surprised" in most scenes, by a hair.

**Borrowed from the IntPhys benchmark:** matched sets. Wherever possible a test
has four versions arranged so that each outcome appears once as possible and
once as impossible, so a brain that is merely startled by "a roller appearing"
scores 50%. Each impossible version is also scored against the possible
version that ends the same way (a matched pair).

**Baselines on exactly the same scenes:** the same network untrained, and
"expect no change" (a brain-free rule that predicts the last thing it saw).

**Checks that the tests themselves work:**
- A brain with its memory wiped every moment scores exactly 50% on every
  hidden-object pair.
- "Expect no change" ties on every pair whose versions look identical on screen.
- A positive control passes. The "watcher" (a brain that only watches, never
  moves, never in the dark) learned roller motion fully and scored 100% with
  76 times the surprise on the plain-view jump test, where the v0.2 brain
  scored 0%.

### The brain had to change

**What went wrong.** The Stage 1 pilot put the v0.2 brain in the new world for
190,000 moments. It never learned how things move. Watching a roller cross
open floor, it bet about 40% on "it stays where it is" and spread the rest
over the neighbouring cells with no sense of direction. Its prediction error
was nearly the same as that of a copy with no memory at all (vision 0.0090
against 0.0094).

**What was ruled out** (one seed each):

| Suspect | Finding |
|---|---|
| The world is unpredictable | No. In the recorded training data a moving roller continues straight unless something visibly blocks it. |
| Vision pathway too narrow | No. Four times wider made no difference. |
| Too few events | No. A denser world (4 balls, 4 rollers) made no difference. |
| A faster learning rate is enough | No. The v0.2 brain at learning rate 0.003 still did not learn motion in 3,000 updates. |
| Motion cannot be learned at all | No. With the agent held still and no blackouts it is learned in 1,000 to 1,500 updates. |

**The fix: a short-term memory inside the vision region.** The region keeps
its own 5x5 map of recent activity, one small vector per cell of the view,
updated by the same small set of weights at every cell (a convolutional GRU).
A second, local vision predictor reads that map directly. The rule "it was one
cell to the left, so next it is one cell to the right" is then learned once
and applies at all 25 cells; the old brain had to discover it separately for
every cell and direction. Together with a faster learning rate (0.003, no
gradient clipping), motion is learned in the real setting by about 2,000 to
3,000 updates.

This is a change to the thesis design, not a tuning detail: **sensory regions
now have a short memory of their own; before, only the core did.** It is
loosely what early visual cortex is like (laid out like the retina, sensitive
to motion, holding a brief trace of what was just seen). It also builds in an
assumption the old brain did not have: that the same rules hold everywhere in
the view. The motion sense that results is less "from nothing" than v0.2's
position sense was.

### The memory conditions

| Condition | What it is |
|---|---|
| **Core only** | The v0.3 brain. Its only long memory is the core's recurrent state. |
| **Remembering pays** | The same brain, raised with darkness half of the time (in stretches of about 33 moments, against a quarter of the time and 17 moments) and lives twice as long in each map. |
| **External memory** | The brain plus a hippocampus-like store. Every 4 moments it files away what it sensed, under an address made from the core's state (a fixed random code). Every moment the core asks the store four questions and gets back a blend of the best-matching memories, which goes to the predictors and to the core. Only the asking is learned. The store holds 128 moments and is wiped at each new life. |
| **Both** | The store, in the world where remembering pays. |

### Choices I made, and why

| Setting | Choice | Why |
|---|---|---|
| Hiding | screens, not line of sight | Simple, controllable geometry; the baby studies use screens too |
| Two kinds of mover | balls and rollers | If everything could start by itself, "moves without being pushed" would not be impossible |
| Friction | none: balls roll until blocked | Makes "a ball stops dead for no reason" a real violation. Cost: balls end up against walls, so pushes get rarer later in a life |
| View | 5x5 for every condition | Watching something go under a screen and come out needs a strip 5 cells wide; a 3x3 view cannot show it |
| Visual memory size | 16 numbers per cell | 8 was tried to save time and learned motion far more slowly |
| Learning rate | 0.003, gliding to 0.001 over the last quarter; no gradient clipping | The combination that learned motion in the pilots |
| Training length | 3,000 updates (96,000 moments per agent) | The time budget. Every brain was still improving when it stopped |
| Curiosity, feedback comparison, noisy TV | dropped | To keep the run near 2 hours |

Training took 25 to 27 minutes per brain without the store and 39 to 40 with
it. The twelve brains took about 1 hour 55 minutes in all.

### Results

Each cell: scenes passed, then how many times more surprising the impossible
versions were. Mean of 3 seeds; per-seed pass rates in brackets where they differ.

| Test | Core only | Remembering pays | External memory | Both | Untrained | Expect no change |
|---|---|---|---|---|---|---|
| Solidity: roller meets wall | 100%, 2.57x | 100%, 2.14x | 100%, 2.79x | 100%, 2.40x | 0%, 0.98x | 0%, 0.92x |
| Continuity: jump in plain view | 89%, 2.35x [100 90 76] | 44%, 1.26x [100 26 6] | 70%, 2.66x [100 100 10] | 53%, 1.64x [76 75 10] | 0%, 0.95x | 0%, 0.23x |
| Causality: ball vs roller setting off | 100%, 1.85x | 100%, 1.89x | 100%, 1.78x | 100%, 1.86x | 56%, 1.00x | 100%, 1.04x |
| Permanence: lights out 5 moments | 100%, 1.98x | 100%, 1.30x | 100%, 1.40x | 100%, 1.19x | 49%, 1.00x | 100% (see note) |
| Causality: watching a collision | 100%, 1.31x | 98%, 1.21x | 100%, 1.28x | 96%, 1.19x | 42%, 1.00x | 0%, 0.82x |
| Causality: pushing it myself | 100%, 1.26x | 100%, 1.25x | 100%, 1.27x | 100%, 1.26x | 0%, 0.98x | 100%, 1.16x |
| Permanence: screen, 1 moment | 100%, 1.19x | 100%, 1.13x | 100%, 1.16x | 100%, 1.12x | 67%, 1.00x | 50%, 1.00x |
| Permanence: lights out 30 moments | 100%, 1.09x | 100%, 1.01x | 90%, 1.02x | 100%, 1.01x | 51%, 1.00x | 100% (see note) |
| Permanence: screen, 3 moments | 68%, 1.00x | 33%, 1.00x | 27%, 1.00x | 30%, 1.00x | 41%, 1.00x | 50%, 1.00x |
| Solidity: ball meets wall (rare) | 0%, 0.94x | 14%, 0.94x | 12%, 0.94x | 21%, 0.95x | 0%, 0.98x | 0%, 0.94x |
| Continuity: two screens and a gap | 0%, 0.12x | 0%, 0.14x | 0%, 0.13x | 0%, 0.14x | 0%, 0.98x | 0%, 0.08x |

Note: in the lights-out tests "expect no change" predicts the last image it
saw, which for a resting ball is the right answer. It is a ceiling there, not
a control.

![common-sense scores](results/v0.3/commonsense_scores.png)

**What is where**, in a never-seen map after 350 to 400 moments (probe error in
grid cells, lower is better; as in v0.2 one probe for all maps, scored on maps
it was not fitted on):

| | Core only | Remembering pays | External memory | Both | Untrained | Chance |
|---|---|---|---|---|---|---|
| Where am I? | 1.57 | 1.56 | 1.65 | 1.72 | 3.61 | 5.82 |
| Where is the bell? | 4.93 | 4.94 | 4.96 | 5.09 | 5.39 | 5.34 |
| Where is the first ball? | 7.74 | 7.74 | 7.74 | 7.79 | 7.76 | 7.65 |

### Reading the results, test by test

**Solidity (rollers): yes.** All 12 brains, every scene. The matched pairs say
the same: a roller going through a wall is 3.1 to 5.0 times as surprising as
one rolling on where there is no wall, and a roller turning away from thin air
is 1.5 to 1.75 times as surprising as one turning away from a wall. So the
brain connects the turn with the wall.

**Continuity in plain view: emerging, and it depends on when motion "clicks".**
The surprise ratio starts at 0.2 (the brain expects nothing to move, so normal
rolling surprises it more than a jump), stays there for 1,500 to 2,500
updates, then climbs steeply. Where a brain is on that climb when training
stops decides its score:

| Seed | Core only | Remembering pays | External memory | Both |
|---|---|---|---|---|
| 0 | 3.8x | 2.7x | 2.7x | 1.9x |
| 1 | 1.6x | 0.6x | 4.7x | 2.5x |
| 2 | 1.6x | 0.4x | 0.6x | 0.4x |

Eight brains are above 1; four have not got there yet. All twelve were still
climbing.

![growing up](results/v0.3/growing_up.png)

**Causality.**
- *Ball vs roller:* a ball setting off by itself gives 91 units of surprise, a
  roller doing the same 52. A resting roller is also expected to maybe move
  (9 units when it stays, against 2 for a ball). The brain treats the two
  kinds differently, as it should.
- *Pushing it myself:* "I push and the ball does not move" is 2.3 to 3.2 times
  as surprising as "I do nothing and it stays", so the brain expects pushing to
  do something. But "I push and it moves" still gives 64 units, nearly as much
  as "it moves untouched" (77). It has not learned what a push does to the
  ball. The agent pushes a ball only about twice per life.
- *Watching a collision:* the first pair looks strong (2.3 to 2.6x) but most
  of that is the blocked roller turning in a direction nobody can predict. The
  second pair is 1.02 to 1.11x. Weak.

**Permanence: short gaps only.** After 5 dark moments the brain is 1.5 to 2.7
times as surprised to find the ball gone as to find an empty spot still empty.
After 30 moments that has faded to almost nothing. Under a screen for one
moment there is a small effect; for three moments, none.

**The two-screens result: it expects things under screens to be gone.** This
test came out backwards in all 12 brains and got MORE backwards with training
(the ratio fell from 1.0 to about 0.13). When the roller goes under the first
screen and comes out into the gap, as it must, the brain is surprised (about
100 units). When it never appears in the gap, the brain is content (13 units).
The one-moment screen test shows the same thing from the other side: a roller
coming out is almost as surprising when one went in (27 units) as when none
did (32 units). In the brain's predictions, a thing that goes under a screen
simply ceases to be. What it has learned about screens is "things vanish
here", which is true of what it sees and false of the world. This is the
clearest picture of what is missing: nothing in this brain yet carries an
object forward while it is out of sight.

**Solidity for balls: no.** It expects a rolling ball to stop (129 units of
surprise when the ball keeps rolling where there is no wall, 1 unit when it
stops dead). Balls roll in under 1% of moments, and the brain has filed them
under "things that do not move".

### Did memory help? The head-start finding

**No condition beat the plain brain.** On every test the core-only brain is
level with or ahead of the three memory conditions, and the differences
between seeds inside a condition are larger than the differences between
conditions.

**An early lead that turned out not to be memory.** The first Stage 2 pilot
(one seed, a smaller visual memory, 4,000 updates) seemed to show a clear
advantage for the external store: roller solidity 2.6x against 1.5x for core
only, and the plain-view jump at 75% against 0%. But the store brains had
simply learned roller motion earlier in training. When the visual memory was
restored to its validated size, the plain brain learned motion on time and the
lead disappeared (same seed: core only 3.7x, external memory 2.9x). With three
seeds the picture is the table above: when motion clicks varies from seed to
seed by more than any condition changes it. A single-seed comparison at one
point in training would have reported a memory effect that was a head start.

**The store is used, just not for these tests.** `memory_check.py` empties the
store before every moment and measures how much worse prediction gets:

| Brain | Vision error | Hearing error |
|---|---|---|
| External memory (3 seeds) | +5%, +13%, +10% | +24%, +44%, +15% |
| Both (3 seeds) | +3%, +5%, +2% | +5%, +7%, +8% |

(Late in a life, moments 300 to 400.) Hearing depends only on where the agent
is, so what the store seems to supply is a sense of "what this place is like".
It does not show up as knowing where the bell is (4.96 against 4.93 for core
only) or as expecting hidden things to come back.

**"Remembering pays" did not pay.** With darkness half the time the brain gets
half the visual experience. At this training length that cost shows (slowest
to learn motion; lights-out permanence 1.3x against 2.0x for core only) and no
benefit does.

**Why memory did not help, as far as this shows.** The failures are not
"forgot where it was"; they are "never represented it as still existing". A
store of past sensations cannot fix that: what was sensed when the roller was
under the screen is a screen. And the tests that would need a long memory
(30 dark moments, where the bell is) may simply need more training before any
memory can show its worth; the short-gap effects were still growing.

### What v0.3 does not show

- **Training was cut short.** Every curve was still rising at 3,000 updates.
  These numbers are a snapshot of brains part-way through learning.
- **Three seeds, and seeds matter a lot here.** Enough for the effects all 12
  brains share (roller solidity, ball vs roller, the two-screens reversal).
  Not enough to rank the memory conditions.
- **Some pass rates owe something to the raw amount of change on screen.**
  "Expect no change" scores 100% on "pushing it myself" (1.16x) and on "ball
  vs roller" (1.04x). The trained ratios are higher (1.26x, 1.85x) and the
  matched pairs are cleaner evidence, but for "pushing it myself" the margin
  is small.
- **The new vision region builds in an assumption** (the same rules at every
  cell of the view). The old brain was not given that and could not learn
  motion without it.
- **Scenes are staged at the very start of a life**, with the agent standing
  still. That is unusual for an agent raised on a random walk.
- **A toy world.** One object is one cell. This says nothing about how hard
  the same tests are in 3D video, where PLATO, IntPhys and V-JEPA work.

### Checks

- 43 unit tests pass (`python -m pytest tests/ -q`), including: the physics
  rules; nothing ever shares a cell; every possible version uses no scripted
  edits and every impossible one does; versions of a scene are identical
  through the quiet period; the store files, overwrites and empties correctly;
  only the reading of the store is trained; saved brains load; the saved
  watcher brain still predicts that rollers keep rolling.
- Runs are deterministic and resumable. Seed 0 of the final run was trained in
  a separate pilot and seeds 1 and 2 later, with the same code and settings.
- v0.1 and v0.2 code paths are unchanged.

### Options for v0.4

**Continue these brains, or start fresh?** Every brain was still improving, so
more training is the cheapest next step. Measured cost: 8.4 minutes per 1,000
updates for a brain without the store, 13.3 with it (four training at once).

| Plan | What runs | Time on 4 cores |
|---|---|---|
| Continue all 12 saved brains by 3,000 updates (to 6,000) | 12 brains x 3,000 | about 1 h 40 min |
| Start 12 fresh brains and train to 6,000 | 12 brains x 6,000 | about 3 h 20 min |
| Continue all 12 by 6,000 updates (to 9,000) | 12 brains x 6,000 | about 3 h 20 min |
| Start 12 fresh brains and train to 9,000 | 12 brains x 9,000 | about 5 h |
| Continue only core-only and external memory (6 brains) to 6,000 | 6 brains x 3,000 | about 55 min |

Continuing saves exactly the time already spent (about 1 h 40 min per 3,000
updates). What to know before choosing it:
- The checkpoints hold the weights, not the optimiser's running averages or
  the agents' lives in progress. Continued training starts those afresh, so it
  is not identical to one long run. Expect a brief wobble, not a setback.
- The learning rate had glided down to 0.001. Continue there, or go back up
  to 0.003; that is a choice to pilot.
- Continuing only works while the brain's shape and the senses stay the same.
  Changing the WORLD is fine (for example, making balls roll more often): the
  brains just carry on in it. Changing the BRAIN (a new module) means the new
  part starts untrained next to trained parts, or a fresh start.

**Other things worth doing, in the order I would do them:**
1. **Train longer** (above). It may by itself move the long-gap tests.
2. **Give the brain a way to keep a hidden thing "alive".** The two-screens
   result says this is the real gap. Options: let the visual map keep a trace
   under screens (it currently overwrites it with what it sees, which is the
   screen); or predict a little further ahead than one moment, so that "it
   will come out the other side" becomes something worth knowing.
3. **Make balls roll more often**, so ball physics stops being a rare event.
4. **More seeds** (5) for anything that compares conditions.
5. **The v0.2 leftovers:** a fair noisy-TV test, giving top-down feedback a
   job, and one continuous life with unannounced moves.

---

## v0.2 — many worlds (2026-11-11)

**Goal:** the brain should build a map of a *new* place within its lifetime,
instead of memorising one fixed map in its weights.

Reproduce with `python run_v02.py` (about 90 minutes on 4 CPU cores), then
`python tv_check.py`. Raw output: `results/v0.2/report.txt`,
`results/v0.2/tv_check.txt`, `results/v0.2/runs/*.json`.

### Verdict

| Question | Answer | Evidence |
|---|---|---|
| Does it work out **where it is** in a place it has never seen? | **Yes, clearly.** | Position error falls from 4.55 to 1.19 cells over 400 moments. Untrained network: 3.68. Raw senses: 4.43. 3 of 3 seeds. |
| Does that survive the lights going out? | **Yes.** | 1.33 cells in the dark after exploring, against 4.38 untrained and 4.77 raw senses. |
| Does it remember **what is where** in the new place? | **Barely.** | "Where is the bell?" is 5.10 cells against 5.37 for chance. Dark predictions are only 6 to 9% better after a long look around. |
| Does top-down feedback matter in the harder setting? | **Not shown.** | Small edge late in life, in 2 of 3 seeds only. |
| Does curiosity help? | **Yes, for exploring.** | Both curious styles see the map faster and get their bearings sooner. |
| Does plain-error curiosity get stuck on the noisy TV? | **No. The demonstration we planned did not work.** | Neither style gets stuck. The reason is measured below. |

The honest one-line summary: **the brain learned to find itself in new places,
but not to remember them.** It builds a "you are here" dot, not yet a map with
things on it.

### What changed from v0.1

| | v0.1 | v0.2 |
|---|---|---|
| World | one fixed 12x12 map | a new random 16x16 map for every life |
| Life | one continuous life | lives of 512 moments; each new life is a new map and a blank core state |
| Objects | bell, fountain, moss | the same, plus one **noisy TV** per map |
| Movement | random walk | random walk, learning-progress curiosity, plain-error curiosity |
| Core | 128 units | 256 units |
| Experience per agent | 48,000 moments | 192,000 moments, across about 48,000 maps per brain |
| Main test | probes on the one known map | probes on never-seen maps, against time since arrival |

Each map has 1 bell, 1 fountain, 3 moss patches and 1 TV, is fully connected,
has no mirror or rotation symmetry, and no two walkable cells in it give
identical sensations. The TV shows a fresh random colour every moment, is
silent, and feels "glassy".

### Choices I made, and why

**What "position" means.** Position is (x, y) measured inside the arena, whose
outer wall is the same size in every map. That makes position *knowable* but
not *visible*: the agent has to find walls and keep count of its own steps from
there. **One** linear probe is shared by all maps. It is fitted on one set of
new maps and scored on a different set, so it cannot memorise any map. It can
only work if the brain uses the same code for position everywhere.

A second probe asks **"where is the bell?"** Position can be worked out by a
general skill stored in the weights (wall-finding plus step-counting). The
bell's location cannot: it is different in every map. So this probe is the
direct test of whether the core holds a fact about *this particular place*.

**Making position hard to see.** I enlarged the map to 16x16 and kept the 5x5
view. I checked this was enough before training anything: on new maps the raw
senses give 4.4 cells of error (chance about 6.0), against 0.33 in v0.1. One
glance no longer tells you where you are. A 3x3 view was not needed.

**Brain size and lifetime.** These come from a pilot on one seed:

| Setting | Choice | Why |
|---|---|---|
| Core size | 256 | 384 was slightly better per update but slower; not better per minute |
| Parallel agents | 128 | 256 gave no gain per update and ran 1.7x slower |
| Learning window | 32 moments | 64 and 128 gave no gain |
| Updates | 6,000 | Position error in new maps was 1.50 at 3,000 updates and 1.08 at 6,000 |
| Learning rate | 0.001, gliding to 0.0003 over the second half | 0.002 was worse (1.65 to 1.73 at 3,000 updates) |
| Life length | 512 moments | A random walker sees about 83% of a map in 400 moments |

Training took 27.5 minutes per brain for random movement and 31 minutes for
the curious styles. Twelve brains (4 conditions x 3 seeds), four at a time,
came to about 90 minutes.

**Blank mind per life.** At each rebirth the core state is zeroed, so the agent
is told, in effect, "this is a new place". That is a simplification. The v0.3
plan removes it (see the end of this entry).

**Curiosity method.** The noisy-TV problem is that anything random is always
badly predicted, so a curiosity that seeks prediction error stares at it
forever. I chose **expected learning progress**:

- Two small extra networks read the core (without being able to change it).
  One guesses "how wrong will my prediction be if I do this?" (first look). The
  other guesses "how wrong would I be on an immediate second look at the same
  place?" (second look). The second one learns from moments where the agent
  happened to stay put.
- The curiosity score for an action is first look minus second look. A new
  corner is surprising once and then known: big score. TV static is as
  surprising the second time: the two cancel.
- The agent's walk is the same random walk as before, with each action's chance
  multiplied by exp(3 x score). With the multiplier off it is exactly the
  random walk, so the movement styles differ in nothing else.

I rejected the common alternative, disagreement between several copies of the
model. Disagreement measures what the *weights* have not learned yet. In this
project the unexplored part of a random map is unknowable to the weights
forever, exactly like TV static, so disagreement could not tell the two apart.

**Plain-error curiosity** uses the first-look guess alone as the score. It was
added as the method that *should* fall for the TV.

### Test 1: finding its bearings in a never-seen map

256 agents, each dropped into its own new map, lights on, random movement.
None of the 1,536 test maps per condition had appeared in training (checked by
fingerprint). Error in grid cells, lower is better.

**Where am I?**

| Probe reads | Moments 1-10 | 40-60 | 90-110 | 190-210 | 350-400 |
|---|---|---|---|---|---|
| **trained core** | 4.55 | 2.18 | 1.59 | 1.23 | **1.19** [1.07 1.41 1.07] |
| untrained core | 4.33 | 3.73 | 3.56 | 3.72 | 3.68 [3.60 3.71 3.72] |
| raw senses | 4.57 | 4.44 | 4.29 | 4.40 | 4.43 [4.39 4.46 4.44] |
| chance | 5.39 | 6.01 | 5.88 | 6.08 | 6.02 |

![new map learning](results/v0.2/new_map_learning.png)

- On arrival the trained brain is as lost as everything else (4.55). Nothing
  about the place is in its weights.
- Its error then halves within about 50 moments and reaches about 1.2 cells by
  moment 200. The untrained network and the raw senses stay flat, because they
  cannot accumulate anything.
- This is the result v0.1 could not give: the position sense is built **during
  the life**, in a place the weights have never seen.

**Where is the bell?**

| Probe reads | Moments 1-10 | 40-60 | 90-110 | 190-210 | 350-400 |
|---|---|---|---|---|---|
| trained core | 5.44 | 5.26 | 5.13 | 5.12 | 5.10 [5.19 5.02 5.10] |
| untrained core | 5.39 | 5.37 | 5.37 | 5.37 | 5.36 |
| raw senses | 5.41 | 5.41 | 5.40 | 5.40 | 5.39 |
| chance | 5.37 | 5.37 | 5.37 | 5.37 | 5.37 |

The trained core is better than chance in all 3 seeds, but only by 5%. For
practical purposes the core does **not** hold where the bell is. (A linear
probe might miss a memory stored in some more tangled form, but the dark-room
predictions below point the same way.)

### Test 2: dark room in a new place

The agent explores a new map with the lights on for either 8 moments (almost
nothing) or 300 moments, then vision is blanked for 100 moments.

| Where am I, in the dark? (chance 6.01) | Position error |
|---|---|
| **trained core, explored 300** | **1.33** [1.26 1.53 1.19] |
| trained core, explored 8 | 3.06 [3.06 3.20 2.92] |
| untrained core, explored 300 | 4.38 |
| raw senses, explored 300 | 4.77 |

| Prediction error in the dark | Explored 8 | Explored 300 | Guess "no change" | Guess the average |
|---|---|---|---|---|
| touch | 0.0084 | 0.0079 | 0.0204 | 0.0529 |
| hearing | 0.0004 | 0.0004 | 0.0013 | 0.0198 |
| vision (imagined) | 0.0192 | 0.0175 | 0.0210 | 0.0390 |

![dark room in a new place](results/v0.2/darkroom_new_place.png)

- **Position holds in the dark.** After 300 moments of looking around, the
  brain keeps its position to 1.33 cells for 100 dark moments with no drift.
- **It can localise by touch and hearing alone.** The agents that saw only 8
  moments start the dark period lost (about 4.1) and still improve to about 2.4
  while blind (dashed line, top-left panel). Vision makes it faster, not
  possible.
- **Looking around first helps prediction only a little.** Touch is 6% better
  and imagined vision 9% better after exploring; hearing is the same. If the
  brain had stored the layout, the gap would be large. Imagined vision is only
  modestly better than guessing "no change" (0.0175 against 0.0210).

### Test 3: does top-down feedback matter here?

Same seeds, same maps, same walks; feedback connections on or cut.

| Where am I? | Moments 40-60 | 90-110 | 190-210 | 350-400 | In the dark, after 300 |
|---|---|---|---|---|---|
| with feedback | 2.18 | 1.59 | 1.23 [1.06 1.52 1.11] | 1.19 [1.07 1.41 1.07] | 1.33 [1.26 1.53 1.19] |
| feedback cut | 2.12 | 1.62 | 1.39 [1.16 1.48 1.54] | 1.34 [1.18 1.35 1.49] | 1.50 [1.34 1.48 1.66] |

![feedback](results/v0.2/feedback.png)

**Not shown.** The first 100 moments are identical. Later the feedback brain is
ahead by about 0.15 cells on average, but only in 2 of 3 seeds, and in seed 1
the order flips. Prediction error during training is indistinguishable, and
the bell probe is the same (5.10 against 5.09). The hint is in the right
direction and slightly larger than in v0.1, but three seeds cannot carry it.
The likely reason is unchanged: these senses are clean and unambiguous, so a
region gains little from being told the context.

### Test 4: three ways of moving

Each brain here was trained with one movement style and is tested with that
same style.

| | Random walk | Learning progress | Plain error |
|---|---|---|---|
| Share of map seen after 100 moments | 0.48 | 0.57 | 0.61 |
| Share of map seen after 400 moments | 0.83 | 0.87 | 0.91 |
| Position error, moments 90-110 | 1.59 [1.47 1.80 1.49] | 1.27 [1.19 1.23 1.38] | 1.31 [1.28 1.37 1.29] |
| Position error, moments 350-400 | 1.19 [1.07 1.41 1.07] | 1.10 [1.15 0.97 1.17] | 1.10 [1.10 1.13 1.07] |
| Bell error, moments 350-400 (chance 5.37) | 5.10 | 4.55 [4.46 4.72 4.45] | 4.80 [5.06 4.75 4.58] |
| Share of time with the TV in view | 0.099 | 0.138 | 0.123 |

![movement styles](results/v0.2/movement_styles.png)

- **Curiosity explores faster.** Both curious styles cover more of the map at
  every point, in all seeds.
- **Curious brains get their bearings sooner, not better.** At moment 100 they
  are ahead in 3 of 3 seeds. By moment 400 the three styles overlap.
- **Curious brains hold a little more about the bell** (4.55 for learning
  progress, 15% better than chance). Treat this with care: a curious agent may
  simply linger near objects, in which case its own position gives the bell
  away. This run cannot separate the two.
- **The curiosity only works for the brain that grew up with it.** Steering a
  random-walk-trained brain with its own curiosity scores (Test 4b in
  `report.txt`) made things worse: 72% of the map seen instead of 83%, and
  position error 1.83 instead of 1.19. That brain had never experienced that
  way of moving.
- Training losses are not comparable between styles. The curious brains show
  higher prediction error because they seek out the hard-to-predict moments.

### The noisy TV: what we planned to show, and what happened

The plan was to show plain-error curiosity stuck at the TV while learning
progress walks away. **That did not happen.** Plain error spent 12.3% of its
time with the TV in view, learning progress 13.8%, the random walk 9.9%.
Neither is stuck, and learning progress is slightly *higher*.

`tv_check.py` takes this apart (units: squared error x 100, summed over senses):

| | Random walk | Plain error | Learning progress |
|---|---|---|---|
| Real error, TV in view | 2.28 | 2.79 | 2.57 |
| Real error, no TV | 0.56 | 0.74 | 0.72 |
| ...no TV, moments 1-50 | 0.59 | 0.75 | 0.71 |
| ...no TV, moments 200-400 | 0.55 | 0.73 | 0.72 |
| Error score (first look), TV in view | 1.19 | 1.42 | 1.46 |
| Error score (first look), no TV | 0.75 | 0.73 | 0.73 |
| Second-look guess, TV in view | 0.61 | 1.17 | 1.17 |
| Second-look guess, no TV | 0.32 | 0.49 | 0.45 |
| Progress score, TV in view | 0.64 | 0.38 | 0.39 |
| Progress score, no TV | 0.46 | 0.27 | 0.31 |
| TV stays in view next moment | 0.904 | 0.889 | 0.916 |
| TV comes into view | 0.010 | 0.015 | 0.013 |

Reading it in three steps:

1. **The TV is a real trap in the world.** Prediction error is about 4 times
   higher when the TV is in the picture.
2. **The scores behave as designed.** With the TV in view the plain-error
   score doubles (0.73 to 1.42). The second-look head has learned that the TV
   stays unpredictable (0.49 to 1.17), so the learning-progress score rises
   much less (0.31 to 0.39). The cancelling trick works on the signal.
3. **The behaviour does not follow.** Once the TV is in view, the plain-error
   agent keeps it in view 88.9% of the time, which is *less* than the random
   walk (90.4%). It does not hold on to the TV. Curious agents see the TV more
   only because they cover more ground and so run into it more often (0.015
   and 0.013 against 0.010 per moment).

Why the trap never closes, as far as this data shows:

- **The rest of the world never gets boring.** A noisy TV traps an agent when
  everything else has become predictable. Here, error on ordinary scenery is
  the same late in a life as at the start (0.75 then 0.73 for the plain-error
  brain). Because the brain does not remember what is where, every step into
  scenery is fresh surprise, and moving stays as attractive as the TV. **The
  missing TV trap and the weak bell memory are the same finding.**
- **The TV is small.** It is 1 of the 25 cells in view, and it only pays while
  the agent is within 2 cells.
- **The curiosity looks one step ahead.** Staying next to a TV for a long time
  takes a plan ("hover here"). A one-step score cannot express that.

So the claim I can support is narrow: the learning-progress *signal* ignores
TV noise that the plain-error signal responds to. The claim "learning progress
avoids a trap that plain error falls into" is **untested**, because in this
world there was no trap to fall into.

### What v0.2 does not show

- **No stored map of contents.** Self-location is solid; landmark memory is
  marginal. The thesis needs both.
- **The agent is told when the place changes** (blank core state at each new
  life). Noticing a move without being told is untested.
- **Position may be a learned routine, not a "map".** Finding walls and
  counting steps is a general skill that lives in the weights. What is built
  within a life is the *estimate*, which is real but more modest than "a map of
  the place".
- **The curiosity comparison mixes two things**: each curious brain differs
  from the random one in both its training experience and its test movement.
- **Three seeds.** Enough for the large effects (position), not for the small
  ones (feedback, final position error between styles).

### Checks

- 21 unit tests pass (`python -m pytest tests/ -q`), including: the fast batched
  senses match the readable v0.1 ones exactly; every map is different; the TV
  is the only thing that flickers; curiosity strength 0 equals the random walk;
  the curiosity networks send no gradient into the world model.
- Runs are deterministic. A run that is interrupted and resumed gives
  bit-identical weights to one that is not.
- v0.1 still reproduces its committed numbers exactly after the v0.2 changes.

### Plan for v0.3

1. **Continuous life with unannounced moves** (decided). One unbroken life. At
   unpredictable moments the agent is placed in a different map with no reset
   and no signal. It has to notice from its own prediction errors that the
   place has changed, drop the old estimate and rebuild. Measure how long it
   takes to notice and to recover.
2. **Get "what is where" to emerge.** This is the main gap. Things to try, in
   the order I would try them:
   - Make remembering pay: a 3x3 view and longer blackouts, so that predicting
     well requires recalling what was seen earlier.
   - Longer lives with more return visits to the same spot.
   - A memory the core can write to and look things up in (store "core state,
     what I sensed" pairs; recall the nearest ones). That is roughly the job
     the hippocampus does next to the cortex. A longer learning window alone
     did not help in the pilot.
3. **A fair noisy-TV test**, after 2. A bigger TV (several cells, or one that
   lights up its surroundings), longer lives, and curiosity that values
   surprise several steps ahead. Only once the rest of the map can become
   boring is there a trap for plain error to fall into.
4. **Give feedback a job**: noisy or ambiguous senses (two objects that look
   the same and sound different), then repeat the feedback comparison with
   more seeds.
5. **Separate the curiosity confound**: test every brain under every movement
   style, and report the bell probe with the agent's distance to the bell
   controlled.

---

## v0.1 — one fixed world

One fixed 12x12 map, one continuous life of 48,000 moments per agent (64
agents), random movement, 128-unit core. Reproduce with
`python run_experiment.py` (2.5 minutes on 4 cores). Raw numbers:
`results/v0.1/results.json`.

### Verdict

**Position emerged; most other things tested did not show emergence.** With
vision off, a linear read-out of the trained core finds the agent to 0.40
cells, against 1.35 for the untrained network and 2.42 for the raw senses, in
3 of 3 seeds.

### Prediction got better

| Sense | Error at start of life | Error at end |
|---|---|---|
| vision | 0.0536 | 0.0046 |
| touch | 0.0796 | 0.0041 |
| hearing | 0.0330 | 0.0002 |

![training loss](results/v0.1/training_loss.png)

### Emergence test: where am I?

Position error in grid cells. The probe is fitted on one stretch of life and
scored on a separate one.

| Probe reads | Lights on | In darkness | Dark and silent |
|---|---|---|---|
| **trained core** | 0.34 [0.36 0.33 0.32] | **0.40** [0.42 0.39 0.38] | **0.61** [0.67 0.64 0.52] |
| untrained core | 1.02 [1.05 0.95 1.05] | 1.35 [1.31 1.39 1.35] | 2.28 [2.22 2.26 2.35] |
| raw senses | 0.33 [0.30 0.34 0.35] | 2.42 [2.44 2.39 2.43] | 2.95 [2.89 2.97 2.99] |
| chance | 4.27 | 4.27 | 4.27 |

![probe results](results/v0.1/probe_results.png)

- **Lights on shows nothing.** The raw senses do as well as the trained core
  (0.33 against 0.34): in a small world a 5x5 view is nearly a fingerprint of
  the cell. (v0.2 fixes this.)
- **Darkness is where it shows.** The trained core stays at 0.40 while the
  baselines fall to 1.35 and 2.42.
- **Dark and silent** (touch only): still 0.61, and flat over 100 steps. The
  brain is adding up its own movements on a remembered layout.

### Emergence test: which object is nearest?

| Probe reads | Lights on | In darkness | Dark and silent |
|---|---|---|---|
| trained core | 1.00 | 0.99 | **0.91** [0.91 0.91 0.93] |
| untrained core | 0.99 | 0.97 | 0.75 [0.77 0.77 0.72] |
| raw senses | 1.00 | 0.93 | 0.63 [0.62 0.63 0.63] |
| chance | 0.42 | 0.42 | 0.42 |

Mostly uninformative: while hearing works, loudness gives the answer away.

### Dark-room test

| Predicting | Lights on | Lights out | Guess "no change" | Guess the average |
|---|---|---|---|---|
| touch | 0.0030 | 0.0068 | 0.0377 | 0.0785 |
| hearing | 0.0001 | 0.0003 | 0.0029 | 0.0279 |
| vision (imagined) | 0.0045 | 0.0119 | 0.0254 | 0.0458 |

![dark room](results/v0.1/darkroom.png)

Predictions get 2 to 3 times worse when the lights go out, then stay flat, and
remain far better than guessing "no change".

### Top-down feedback

| Position error | Lights on | In darkness | Dark and silent |
|---|---|---|---|
| with feedback | 0.34 | 0.40 [0.42 0.39 0.38] | 0.61 |
| feedback cut | 0.36 | 0.46 [0.48 0.49 0.42] | 0.63 |

Hardly any difference at this scale.

### What v0.1 does not show

- **The map is in the weights.** One fixed world, memorised over 48,000
  moments. Nothing here says how fast a new place could be learned. (This is
  what v0.2 set out to test.)
- **The agent practised being blind**: vision and hearing each switch off about
  a quarter of the time in training.
- **Untrained networks are a strong baseline** (1.02 cells with the lights on).
- A linear probe finding position shows the information is present, not that
  the brain "uses" it.
- Three seeds on one map.

### Decisions made in v0.1

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
