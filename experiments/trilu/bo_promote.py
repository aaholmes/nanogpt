"""
Multi-fidelity promotion: take the winners of the 30-min iso-compute screen and
re-evaluate them at a longer budget, to (a) confirm the best config in a deeper
regime and (b) test whether the 30-min ranking is BUDGET-STABLE.

Selection from the screen study:
  * top-3 configs by 30-min loss (the contenders), plus
  * 1 mid-ranked config as a CONTROL — if it leapfrogs a top config at the longer
    budget, the short screen is an unreliable predictor and we'd need longer
    budgets throughout. If the order holds, the screen is validated.

Each promoted config runs PROMOTE_SEEDS seeds at PROMOTE_BUDGET. Reports the
30-min vs long-budget loss for each, and whether the ranking is preserved.
"""

import argparse
import json
from pathlib import Path

import optuna

import bo_budget_search as bb   # reuse run_one + study coordinates

PROMOTE_BUDGET = 3600   # seconds per seed-run (1 hr; override via --budget)
PROMOTE_SEEDS  = 2


def trial_cfg(t):
    md = t.params["model_dim"]
    nh = next(v for k, v in t.params.items() if k.startswith("num_heads_"))
    return dict(model_dim=md, num_heads=nh, muon_lr=t.params["muon_lr"],
                adam_lr=t.params["adam_lr"], screen_loss=t.value,
                screen_seeds=t.user_attrs.get("n_seeds"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=PROMOTE_BUDGET)
    ap.add_argument("--seeds", type=int, default=PROMOTE_SEEDS)
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    study = optuna.load_study(study_name=bb.STUDY_NAME, storage=bb.STUDY_DB)
    fin = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value)
    if len(fin) < 2:
        print("screen has too few finished trials to promote"); return

    # top-N contenders + 1 mid-ranked control (distinct)
    selected = fin[:args.top]
    mid = fin[len(fin) // 2]
    if mid not in selected:
        selected = selected + [mid]
    control_idx = len(selected) - 1 if mid not in fin[:args.top] else None

    print(f"Promoting {len(selected)} configs to {args.budget:.0f}s x {args.seeds} seeds")
    print(f"(top-{args.top} contenders" +
          (f" + 1 mid-rank control)" if control_idx is not None else ")") + "\n")

    results = []
    for i, t in enumerate(selected):
        c = trial_cfg(t)
        tag = "CONTROL" if i == control_idx else f"top{i+1}"
        print(f"--- promoting [{tag}] dim={c['model_dim']} heads={c['num_heads']} "
              f"muon={c['muon_lr']:.4f} adam={c['adam_lr']:.4f} "
              f"(30min loss={c['screen_loss']:.4f}) ---")
        losses = []
        for s in range(args.seeds):
            L = bb.run_one(c["model_dim"], c["num_heads"], c["muon_lr"], c["adam_lr"],
                           args.budget, 900 + i, s)   # trial ids 900+ avoid clashes
            if L is not None:
                losses.append(L)
                print(f"    seed {s}: {L:.4f}")
        mean = sum(losses) / len(losses) if losses else float("nan")
        results.append(dict(tag=tag, **c, long_loss=mean, long_seeds=len(losses)))

    # Report + budget-stability verdict
    print("\n=== Promotion results: 30-min screen vs longer budget ===")
    print(f"  {'tag':8}  {'dim':>5} {'heads':>6}  {'30min':>8}  {'long':>8}  {'Δ':>8}")
    for r in results:
        print(f"  {r['tag']:8}  {r['model_dim']:>5} {str(r['num_heads'])+'h':>6}  "
              f"{r['screen_loss']:>8.4f}  {r['long_loss']:>8.4f}  {r['long_loss']-r['screen_loss']:>+8.4f}")

    screen_order = [r['tag'] for r in sorted(results, key=lambda r: r['screen_loss'])]
    long_order   = [r['tag'] for r in sorted(results, key=lambda r: r['long_loss'])]
    print(f"\n  screen ranking: {' < '.join(screen_order)}")
    print(f"  long   ranking: {' < '.join(long_order)}")
    stable = screen_order == long_order
    print(f"\n  BUDGET-STABLE (ranking preserved): {stable}")
    if not stable:
        print("  ⚠ the 30-min screen did NOT predict the longer-budget ranking —")
        print("    short-budget screening is unreliable here; longer budgets needed.")
    else:
        print("  ✓ the 30-min screen predicts the longer-budget ranking — screen validated.")

    out = Path("experiments/trilu/bo_promotion_results.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
