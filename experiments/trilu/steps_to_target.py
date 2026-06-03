"""Compute steps-to-target (first eval step where val loss <= target) post-hoc from
logged val curves. The benchmark-aligned metric: how fast a variant reaches a given
loss, rather than its loss at a fixed step. Reports per-seed and the mean.
"""
import json, sys
import numpy as np

target = float(sys.argv[1])
for f in sys.argv[2:]:
    d = json.load(open(f))
    k = list(d)[0]
    steps = []
    for r in d[k]:
        hit = next((s for s, v in r["val_loss"] if v <= target), None)
        steps.append(hit)
    reached = [s for s in steps if s is not None]
    name = f.split("results_")[-1].replace(".json", "")
    if len(reached) == len(steps) and steps:
        print(f"  {name:28s} steps@{target}: mean={np.mean(reached):.0f}  per-seed={steps}")
    else:
        print(f"  {name:28s} steps@{target}: {len(reached)}/{len(steps)} seeds reached  per-seed={steps}")
