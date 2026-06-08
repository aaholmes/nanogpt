"""
Phase 2 analysis: checks whether activation rankings are stable across the
tested LR grid, not just at each activation's individual best LR.

The central question: could a coarse LR grid cause a misleading ranking?
If activation A beats B at every tested LR, the result is robust within
that range — an LR artifact would need the crossover to fall in a specific
narrow window outside the grid. If the ranking flips at any grid point,
the result is LR-sensitive and should be treated with more caution.

Usage:
    python experiments/trilu/analyze_phase2.py          # all available data
    python experiments/trilu/analyze_phase2.py --sweep  # sweep only
    python experiments/trilu/analyze_phase2.py --main   # main results only
"""

import argparse
import json
import sys
from pathlib import Path

DIR = Path("experiments/trilu")

# ── LR grid used in the sweep ─────────────────────────────────────────────────
NORMUON_LRS = ["0.01", "0.02", "0.04"]
MUON_LRS    = ["0.01", "0.02", "0.04"]
ADAMW_LRS   = ["3e-4", "1e-3", "3e-3"]

# JSON result-key suffix for each optimizer (normuon has no suffix = default)
OPT_SUFFIX = {"normuon": "", "muon": "_muon", "adamw": "_adamw"}

# Display names
ACT_DISPLAY = {
    "relu2":      "relu2     ",
    "sniqu":      "sniqu     ",
    "reglu":      "reglu     ",
    "xglu":       "xglu      ",
    "relu2_s1t0": "relu2_s1  ",
}
ACTS = list(ACT_DISPLAY)

OPTS = [
    ("normuon", NORMUON_LRS, lambda lr: DIR / f"phase2_sweep_normuon_lr{lr.replace('.','')}.json"),
    ("muon",    MUON_LRS,    lambda lr: DIR / f"phase2_sweep_muon_lr{lr.replace('.','')}.json"),
    ("adamw",   ADAMW_LRS,   lambda lr: DIR / f"phase2_sweep_adamw_lr{lr}.json"),
]


def load_sweep_loss(filepath, act_key, opt):
    """Mean final loss for act_key from a sweep file, or None if missing."""
    if not filepath.exists():
        return None
    d = json.load(open(filepath))
    suffix = OPT_SUFFIX[opt]
    # Try exact key first, then prefix match (handles optimizer suffix in key)
    candidates = [k for k in d if k == act_key + suffix
                  or (not suffix and k == act_key)]
    if not candidates:
        # Fallback: prefix match (e.g. "relu2_muon" matches act_key="relu2" + suffix="_muon")
        candidates = [k for k in d if k.startswith(act_key + suffix)
                      or k.startswith(act_key + "_" + opt)]
    if not candidates:
        return None
    key = candidates[0]
    seeds = d[key]
    if not seeds:
        return None
    return sum(s["final_loss"] for s in seeds) / len(seeds)


def load_main_loss(opt, act_key):
    """Mean final loss from the main experiment results, or None if missing."""
    path = DIR / f"results_phase2_{opt}.json"
    if not path.exists():
        return None
    d = json.load(open(path))
    suffix = OPT_SUFFIX[opt]
    key = act_key + suffix
    if key not in d or not d[key]:
        return None
    seeds = d[key]
    return sum(s["final_loss"] for s in seeds) / len(seeds), len(seeds)


def load_best_lrs():
    path = DIR / "phase2_best_lrs.json"
    if not path.exists():
        return None
    return json.load(open(path))


def wins_at_all_lrs(matrix, act_a, act_b):
    """Does act_a beat act_b at every LR where both have data?"""
    comparisons = []
    for lr, loss_a, loss_b in matrix:
        if loss_a is not None and loss_b is not None:
            comparisons.append(loss_a < loss_b)
    if not comparisons:
        return None
    if all(comparisons):
        return "always"
    if not any(comparisons):
        return "never"
    return "mixed"


def fmt(val):
    return f"{val:.4f}" if val is not None else "  ---  "


def print_sweep_section(opt, lrs, file_fn):
    print(f"\n{'='*68}")
    print(f"  {opt.upper()} — sweep (LR stability check)")
    print(f"{'='*68}")

    # Build matrix: act → {lr: loss}
    matrix = {act: {} for act in ACTS}
    any_data = False
    for lr in lrs:
        for act in ACTS:
            loss = load_sweep_loss(file_fn(lr), act, opt)
            matrix[act][lr] = loss
            if loss is not None:
                any_data = True

    if not any_data:
        print("  (no sweep data yet)")
        return

    # Header
    lr_cols = "  ".join(f"lr={lr:>6}" for lr in lrs)
    print(f"\n  {'activation':12s}  {lr_cols}  {'best_lr':>8}")
    print(f"  {'-'*12}  {'  '.join(['-'*10]*len(lrs))}  {'-'*8}")

    best_lrs = load_best_lrs()
    for act in ACTS:
        row = matrix[act]
        losses = [row.get(lr) for lr in lrs]
        # Best LR: lowest loss among available
        valid = [(lr, l) for lr, l in zip(lrs, losses) if l is not None]
        best = min(valid, key=lambda x: x[1])[0] if valid else "?"
        # Override with phase2_best_lrs.json if available
        if best_lrs and opt in best_lrs and act in best_lrs[opt]:
            best = str(best_lrs[opt][act])
        cols = "  ".join(f"{fmt(l):>10}" for l in losses)
        print(f"  {ACT_DISPLAY[act]}  {cols}  {best:>8}")

    # Pairwise ranking stability
    print(f"\n  Pairwise ranking stability across LR grid:")
    key_pairs = [("relu2", "sniqu"), ("relu2", "reglu"), ("relu2", "xglu"),
                 ("relu2", "relu2_s1t0"), ("sniqu", "reglu")]
    for a, b in key_pairs:
        comparisons = []
        for lr in lrs:
            la, lb = matrix[a].get(lr), matrix[b].get(lr)
            if la is not None and lb is not None:
                delta = lb - la  # positive = a wins
                comparisons.append((lr, delta))
        if not comparisons:
            continue
        stable = all(d > 0 for _, d in comparisons) or all(d < 0 for _, d in comparisons)
        winner = ACT_DISPLAY[a].strip() if comparisons[0][1] > 0 else ACT_DISPLAY[b].strip()
        details = "  ".join(f"lr={lr}: {'+' if d>0 else ''}{d:.4f}" for lr, d in comparisons)
        flag = "✓ STABLE" if stable else "⚠ FLIPS"
        print(f"    {ACT_DISPLAY[a].strip():10} vs {ACT_DISPLAY[b].strip():10}  {flag}  ({details})")


def print_main_section(opt):
    print(f"\n  {opt.upper()} — main results (per-activation best LR, seeds OUTER)")
    print(f"  {'-'*50}")
    best_lrs = load_best_lrs()
    any_data = False
    rows = []
    for act in ACTS:
        result = load_main_loss(opt, act)
        lr = "?"
        if best_lrs and opt in best_lrs and act in best_lrs[opt]:
            lr = str(best_lrs[opt][act])
        if result is not None:
            mean, n = result
            rows.append((act, mean, n, lr))
            any_data = True
        else:
            rows.append((act, None, 0, lr))

    if not any_data:
        print("  (no main results yet)")
        return

    # Sort by loss
    rows_with_data = [(a, m, n, lr) for a, m, n, lr in rows if m is not None]
    rows_with_data.sort(key=lambda x: x[1])

    print(f"\n  {'rank':4}  {'activation':12}  {'mean_loss':>10}  {'n':>4}  {'best_lr':>8}")
    print(f"  {'----':4}  {'-'*12}  {'-'*10}  {'----':>4}  {'-'*8}")
    for rank, (act, mean, n, lr) in enumerate(rows_with_data, 1):
        print(f"  {rank:<4}  {ACT_DISPLAY[act]}  {mean:>10.4f}  {n:>4}  {lr:>8}")

    # Key gaps
    if len(rows_with_data) >= 2:
        print(f"\n  Key gaps (vs relu2 baseline):")
        relu2_loss = next((m for a, m, n, lr in rows_with_data if a == "relu2"), None)
        if relu2_loss:
            for act, mean, n, lr in rows_with_data:
                if act != "relu2":
                    delta = mean - relu2_loss
                    sign = "+" if delta > 0 else ""
                    print(f"    {ACT_DISPLAY[act].strip():12}  {sign}{delta:.4f}  "
                          f"({'worse' if delta > 0 else 'better'} than relu2)")


def print_phase1_context():
    """Show phase1 results for comparison."""
    phase1_files = {
        "muon":    [("relu2", DIR / "results_muon_relu2.json"),
                    ("sniqu", DIR / "results_muon_relu2.json"),   # placeholder
                    ("reglu", DIR / "results_muon_reglu.json"),
                    ("xglu",  DIR / "results_muon_xglu.json")],
        "normuon": [("relu2",  DIR / "normuon_study/results.json"),
                    ("sniqu",  DIR / "normuon_study/results.json")],
        "adamw":   [("relu2",  DIR / "adamw124/main_relu2.json"),
                    ("sniqu",  DIR / "adamw124/main_sniqu.json"),
                    ("reglu",  DIR / "adamw124/main_reglu.json"),
                    ("xglu",   DIR / "adamw124/main_xglu.json")],
    }

    print(f"\n{'='*68}")
    print(f"  PHASE 1 CONTEXT (for comparison with phase2)")
    print(f"{'='*68}")

    # NorMuon: special file structure
    nm_path = DIR / "normuon_study/results.json"
    if nm_path.exists():
        d = json.load(open(nm_path))
        print(f"\n  NorMuon (phase1, n=9 seeds, fixed muon_lr=0.02, NO per-act LR sweep):")
        for key in d:
            act = key.replace("_normuon", "")
            seeds = d[key]
            losses = [s["val_loss"][-1][1] for s in seeds]
            mean = sum(losses) / len(losses)
            print(f"    {act:12}  mean={mean:.4f}  n={len(losses)}")
        # delta
        keys = list(d.keys())
        if len(keys) >= 2:
            r = [s["val_loss"][-1][1] for s in d[keys[0]]]
            s_vals = [s["val_loss"][-1][1] for s in d[keys[1]]]
            n = min(len(r), len(s_vals))
            delta = sum(s_vals[i] - r[i] for i in range(n)) / n
            print(f"    → sniqu vs relu2: Δ={delta:+.4f} "
                  f"({'sniqu worse' if delta > 0 else 'sniqu better'}, n={n})")
        print(f"    ⚠ No per-activation LR sweep — LR precision concern applies")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="show only sweep data")
    ap.add_argument("--main",  action="store_true", help="show only main results")
    ap.add_argument("--phase1", action="store_true", help="show phase1 context only")
    args = ap.parse_args()

    show_all = not (args.sweep or args.main or args.phase1)

    if show_all or args.phase1:
        print_phase1_context()

    for opt, lrs, file_fn in OPTS:
        if show_all or args.sweep:
            print_sweep_section(opt, lrs, file_fn)
        if show_all or args.main:
            print_main_section(opt)

    print()


if __name__ == "__main__":
    main()
