"""
Iso-compute architecture BO (Optuna TPE): minimize the validation loss reached
in a fixed wall-clock budget, using the speedrun's faithful softcapped CE.

Objective: final val loss after BUDGET_SECONDS of training (NorMuon, relu2,
softcap=23). Lower is better. This is the fair speed/quality tradeoff — a config
wins only by reaching lower loss in the same wall-clock, so fast-but-shallow and
slow-but-deep models compete on equal footing (the speedrun's real currency).

Search space (4-dim): model_dim, num_heads (conditional), muon_lr, adam_lr.

Adaptive seeding (racing): each config first runs ONE seed. Only configs that are
competitive with the best-so-far earn confirmation seeds — extra compute is spent
where the seed noise actually affects the decision, not on clearly-bad configs.
  * gate 1: if the 1-seed loss is within MARGIN1 of the best, add a 2nd seed.
  * gate 2: if the 2-seed mean is within MARGIN2 (tighter) of the best, add a 3rd.
The first trial always runs 2 seeds to establish a baseline scale.

SQLite-backed and resumable. The record config (768x6, muon_lr=0.02, adam_lr=0.008)
is enqueued as trial 0 to anchor the comparison.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import optuna

import bo_search as b4   # reuse MODEL_DIMS, valid_head_counts

DIR = b4.DIR
STUDY_DB = f"sqlite:///{DIR}/bo_budget_study.db"
STUDY_NAME = "phase2_arch_iso_compute"

BUDGET_SECONDS = 3600   # wall-clock per seed-run (overridable via --budget)
STEP_CAP       = 200000 # large, so time is the binding constraint
MARGIN1 = 0.05          # within this of best -> add a 2nd seed
MARGIN2 = 0.02          # 2-seed mean within this of best -> add a 3rd seed
MAX_SEEDS = 3


def run_one(model_dim, num_heads, muon_lr, adam_lr, budget, trial_number, seed):
    """One seed-run; returns final val loss (or None on OOM/crash)."""
    head_dim = model_dim // num_heads
    out_path = DIR / f"bo_budget_t{trial_number:03d}_s{seed}.json"
    cmd = [
        sys.executable, "experiments/trilu/phase2.py",
        "--config", "speedrun", "--compile", "--batch-size", "8",
        "--paired-heads", "--ce-chunk", "4096", "--optimizer", "normuon",
        "--model-dim", str(model_dim), "--num-heads", str(num_heads),
        "--head-dim", str(head_dim),
        "--muon-lr", f"{muon_lr:.6f}", "--adam-lr", f"{adam_lr:.6f}",
        "--activations", "relu2",
        "--steps", str(STEP_CAP), "--max-seconds", str(budget),
        "--seeds", "1", "--seed-start", str(seed), "--out", str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True,
                       timeout=budget + 600)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        tail = (e.stderr or "")[-300:] if hasattr(e, "stderr") else ""
        print(f"    t{trial_number} s{seed} FAILED ({type(e).__name__}): {tail}")
        return None
    d = json.load(open(out_path))
    return next(iter(d.values()))[0]["final_loss"]


def best_so_far(study):
    vals = [t.value for t in study.trials if t.value is not None]
    return min(vals) if vals else float("inf")


def make_objective(budget):
    def objective(trial):
        model_dim = trial.suggest_categorical("model_dim", b4.MODEL_DIMS)
        num_heads = trial.suggest_categorical(f"num_heads_{model_dim}",
                                              b4.valid_head_counts(model_dim))
        muon_lr = trial.suggest_float("muon_lr", 0.005, 0.05, log=True)
        adam_lr = trial.suggest_float("adam_lr", 0.002, 0.02, log=True)

        best = best_so_far(trial.study)
        n_done = len([t for t in trial.study.trials if t.value is not None])

        losses = []
        l0 = run_one(model_dim, num_heads, muon_lr, adam_lr, budget, trial.number, 0)
        if l0 is None:
            raise optuna.TrialPruned()
        losses.append(l0)

        # Adaptive seeding: confirm only competitive configs (or the first trial).
        force_two = (n_done == 0)
        if force_two or l0 <= best + MARGIN1:
            l1 = run_one(model_dim, num_heads, muon_lr, adam_lr, budget, trial.number, 1)
            if l1 is not None:
                losses.append(l1)
                if (sum(losses) / len(losses)) <= best + MARGIN2:
                    l2 = run_one(model_dim, num_heads, muon_lr, adam_lr, budget, trial.number, 2)
                    if l2 is not None:
                        losses.append(l2)

        mean = sum(losses) / len(losses)
        trial.set_user_attr("n_seeds", len(losses))
        trial.set_user_attr("losses", losses)
        trial.set_user_attr("head_dim", model_dim // num_heads)
        print(f"  #{trial.number}: dim={model_dim} heads={num_heads}x{model_dim//num_heads} "
              f"muon={muon_lr:.4f} adam={adam_lr:.4f} -> mean={mean:.4f} "
              f"(n_seeds={len(losses)}, best={best:.4f})")
        return mean
    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--budget", type=float, default=BUDGET_SECONDS,
                    help="wall-clock seconds per seed-run (default 3600 = 1 hr)")
    args = ap.parse_args()

    DIR.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=0, multivariate=True, group=True)
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=STUDY_DB,
        direction="minimize", sampler=sampler, load_if_exists=True,
    )
    # Anchor: enqueue the record config first (idempotent — only if nothing done yet)
    if not study.trials:
        study.enqueue_trial({"model_dim": 768, "num_heads_768": 6,
                             "muon_lr": 0.02, "adam_lr": 0.008})

    done = len([t for t in study.trials if t.state.is_finished()])
    print(f"Study '{STUDY_NAME}': {done} trials done, +{args.n_trials} requested")
    print(f"Objective: min val loss in {args.budget:.0f}s budget (softcap, NorMuon, relu2)")
    print(f"Adaptive seeding: 1 seed, +confirm if within {MARGIN1}/{MARGIN2} of best\n")

    study.optimize(make_objective(args.budget), n_trials=args.n_trials)

    print("\n=== Best ===")
    bt = study.best_trial
    nh = next((v for k, v in bt.params.items() if k.startswith("num_heads_")), "?")
    print(f"  loss={bt.value:.4f}  dim={bt.params['model_dim']} heads={nh}x{bt.user_attrs.get('head_dim')} "
          f"muon={bt.params['muon_lr']:.4f} adam={bt.params['adam_lr']:.4f} "
          f"(n_seeds={bt.user_attrs.get('n_seeds')})")

    print("\n=== All trials by loss ===")
    for t in sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value):
        md = t.params.get("model_dim")
        nh = next((v for k, v in t.params.items() if k.startswith("num_heads_")), "?")
        print(f"  loss={t.value:.4f}  dim={md} heads={nh}x{t.user_attrs.get('head_dim','?')} "
              f"muon={t.params.get('muon_lr',0):.4f} adam={t.params.get('adam_lr',0):.4f} "
              f"n_seeds={t.user_attrs.get('n_seeds','?')}")


if __name__ == "__main__":
    main()
