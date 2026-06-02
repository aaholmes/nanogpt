"""Paired-significance analysis for Phase 1 TriLU sweep.

Reports final-step val loss per variant (mean +/- std across seeds), then
runs paired t-tests between variants. Pairing is on seed index: all variants
share the same per-seed initialization/data draw, so a paired test removes the
(large, correlated) seed effect and is far more powerful than an unpaired test.
"""

import argparse
import json
from itertools import combinations

import numpy as np
from scipy import stats


def final_losses(runs):
    """Return array of final-step val loss, one per seed (ordered by seed)."""
    return np.array([r["val_loss"][-1][1] for r in runs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/trilu/results_overnight.json")
    args = ap.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    variants = list(results)
    finals = {v: final_losses(results[v]) for v in variants}
    n = min(len(a) for a in finals.values())
    # Truncate all to common seed count for valid pairing.
    finals = {v: a[:n] for v, a in finals.items()}

    print(f"Final-step val loss (n={n} seeds), lower is better\n")
    ranked = sorted(variants, key=lambda v: finals[v].mean())
    for v in ranked:
        a = finals[v]
        print(f"  {v:12s}  mean={a.mean():.4f}  std={a.std(ddof=1):.4f}  "
              f"min={a.min():.4f}  max={a.max():.4f}")

    print("\nPaired t-tests (every pair; pairing on seed index)\n")
    print(f"  {'A':12s} {'B':12s} {'meanA':>8s} {'meanB':>8s} {'delta':>8s} {'t':>7s} {'p':>9s}  sig")
    for a, b in combinations(ranked, 2):
        da, db = finals[a], finals[b]
        diff = da - db
        t, p = stats.ttest_rel(da, db)
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        print(f"  {a:12s} {b:12s} {da.mean():8.4f} {db.mean():8.4f} "
              f"{diff.mean():+8.4f} {t:7.2f} {p:9.5f}  {sig}")

    # Headline contrasts: gated vs standard.
    print("\nGrouped contrast: gated vs standard MLP\n")
    gated = ["swiglu", "geglu", "triglu"]
    standard = ["relu2", "gelu", "trilu_asym"]
    gated = [v for v in gated if v in finals]
    standard = [v for v in standard if v in finals]
    if gated and standard:
        gmean = np.concatenate([finals[v] for v in gated]).mean()
        smean = np.concatenate([finals[v] for v in standard]).mean()
        # Per-seed mean within each group, then paired across seeds.
        gper = np.mean([finals[v] for v in gated], axis=0)
        sper = np.mean([finals[v] for v in standard], axis=0)
        t, p = stats.ttest_rel(gper, sper)
        print(f"  gated mean    = {gmean:.4f}  ({', '.join(gated)})")
        print(f"  standard mean = {smean:.4f}  ({', '.join(standard)})")
        print(f"  delta (gated - standard) = {gmean - smean:+.4f}")
        print(f"  paired t (per-seed group means): t={t:.2f}  p={p:.6f}"
              f"  {'**' if p < 0.01 else ('*' if p < 0.05 else '')}")


if __name__ == "__main__":
    main()
