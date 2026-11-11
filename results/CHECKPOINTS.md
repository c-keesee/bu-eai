# Saved brains

Every file here is a trained brain plus the recipe to rebuild it. Load one with:

```python
from bueai.checkpoints import load_brain
brain, info = load_brain("results/v0.3/checkpoints/pilot/watcher.pt")
print(info["note"])
```

| Folder | What is in it |
|---|---|
| `results/v0.3/checkpoints/` | The final v0.3 brains: `{condition}_seed{seed}.pt` |
| `results/v0.3/checkpoints/pilot/` | Brains from the v0.3 pilots (one seed each, see below) |
| `results/v0.2/checkpoints/` | The twelve v0.2 brains: `{random, progress, error, no_feedback}_seed{0,1,2}.pt` |

v0.1 brains were never saved; `python run_experiment.py` retrains them in 2.5 minutes.

Folders named `weights/` are scratch space for training in progress (so an
interrupted run can resume). They are not kept in git.

## v0.3 pilot brains

All: seed 0, lively world with 3 balls and 3 rollers unless noted, 128 agents, lives of 512 moments.

| File | Brain | Training | What it showed |
|---|---|---|---|
| `v02brain.pt` | v0.2 brain | 6,000 updates, learning rate 0.001 | Did not learn how things move |
| `v02brain_memoryless.pt` | v0.2 brain, core wiped every moment | same | The check: 50% on every hidden-object pair |
| `v02brain_dense_world.pt` | v0.2 brain | same, 4 balls and 4 rollers | A denser world did not help |
| `v02brain_fast_learning.pt` | v0.2 brain | 3,000 updates, learning rate 0.003, no clipping | A faster learning rate alone did not help |
| `watcher.pt` | visual short-term memory | 2,500 updates, agent never moves, no blackouts | Positive control: learned roller motion fully |
| `visual_memory.pt` | visual short-term memory | 3,000 updates, real setting | Learned roller motion; first real common-sense effects |
| `visual_memory_no_blackouts.pt` | visual short-term memory | same, no blackouts | Same motion learning; no memory through darkness |
