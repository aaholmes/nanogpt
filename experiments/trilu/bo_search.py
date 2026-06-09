"""
4-dim Bayesian optimization over phase2 architecture + NorMuon learning rates,
minimizing wall-clock time to reach a target validation loss.

Search space (NorMuon optimizer only):
    model_dim   ∈ {512, 640, 768, 896, 1024}     residual width
    num_heads   ∈ valid divisors (head_dim = model_dim/num_heads, %4 == 0)
    muon_lr     ∈ [0.005, 0.05]   log-uniform   (matrix / NorMuon path)
    adam_lr     ∈ [0.002, 0.02]   log-uniform   (AdamW path: embeddings, banks,
                                                 scalars, gates — active under
                                                 NorMuon for all non-matrix params)

Objective: seconds to first cross TARGET_LOSS, averaged over SEEDS. Configs that
never reach it within STEP_CAP are penalised (worst time + loss-gap term) so the
surrogate still gets a gradient toward the feasible region.

Each trial runs phase2.py as a subprocess (isolates OOM / crashes → penalty).
Study is SQLite-backed and resumable: re-running continues from completed trials.

Usage:
    python experiments/trilu/bo_search.py --n-trials 40
    python experiments/trilu/bo_search.py --n-trials 40   # resumes same study
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import optuna

DIR = Path("experiments/trilu")
STUDY_DB = f"sqlite:///{DIR}/bo_study.db"
STUDY_NAME = "phase2_arch_normuon_4d"

TARGET_LOSS = 10.20     # baseline 6x128 crosses this ~step 720
STEP_CAP    = 2000      # max steps per eval
SEEDS       = 2         # seeds averaged per trial (noise control)

MODEL_DIMS = [512, 640, 768, 896, 1024]


def valid_head_counts(model_dim):
    """Head counts h such that head_dim = model_dim/h is an integer divisible by 4
    and head_dim lands in a sane range [32, 256]."""
    out = []
    for h in range(2, 33):
        if model_dim % h != 0:
            continue
        hd = model_dim // h
        if hd % 4 == 0 and 32 <= hd <= 256:
            out.append(h)
    return out


def run_eval(model_dim, num_heads, muon_lr, adam_lr, trial_number):
    """Run phase2 as a subprocess; return (mean_time_to_target, reached, mean_final_loss)."""
    head_dim = model_dim // num_heads
    out_path = DIR / f"bo_trial_{trial_number:03d}.json"
    cmd = [
        sys.executable, "experiments/trilu/phase2.py",
        "--config", "speedrun", "--compile", "--batch-size", "8",
        "--paired-heads", "--ce-chunk", "4096",
        "--optimizer", "normuon",
        "--model-dim", str(model_dim),
        "--num-heads", str(num_heads),
        "--head-dim", str(head_dim),
        "--muon-lr", f"{muon_lr:.6f}",
        "--adam-lr", f"{adam_lr:.6f}",
        "--activations", "relu2",
        "--steps", str(STEP_CAP),
        "--seeds", str(SEEDS),
        "--target-loss", str(TARGET_LOSS),
        "--stop-at-target",
        "--out", str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=7200)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        tail = (e.stderr or "")[-400:] if hasattr(e, "stderr") else ""
        print(f"  trial {trial_number} FAILED ({type(e).__name__}): {tail}")
        return None, False, None

    d = json.load(open(out_path))
    # phase2 writes one key (relu2 + arch suffix); grab its seed list
    seeds = next(iter(d.values()))

    times, reached_flags, finals = [], [], []
    for s in seeds:
        finals.append(s["final_loss"])
        # first eval checkpoint crossing the target
        crossing = next(((step, t) for (step, loss), (_, t)
                         in zip(s["val_loss"], s["wallclock"]) if loss <= TARGET_LOSS), None)
        if crossing is not None:
            times.append(crossing[1])
            reached_flags.append(True)
        else:
            times.append(s["wallclock"][-1][1])   # full run time
            reached_flags.append(False)

    mean_final = sum(finals) / len(finals)
    if all(reached_flags):
        return sum(times) / len(times), True, mean_final
    # Censored: penalise. Worst observed time + soft loss-gap term to guide search.
    worst_time = max(times)
    gap = max(0.0, mean_final - TARGET_LOSS)
    return worst_time + 1000.0 * gap, False, mean_final


def _objective_from_seeds(seeds):
    """Compute the same time-to-target objective from a list of seed logs."""
    times, reached = [], []
    finals = []
    for s in seeds:
        finals.append(s["final_loss"])
        crossing = next(((step, t) for (step, loss), (_, t)
                         in zip(s["val_loss"], s["wallclock"]) if loss <= TARGET_LOSS), None)
        if crossing is not None:
            times.append(crossing[1]); reached.append(True)
        else:
            times.append(s["wallclock"][-1][1]); reached.append(False)
    mean_final = sum(finals) / len(finals)
    if all(reached):
        return sum(times) / len(times), True, mean_final
    gap = max(0.0, mean_final - TARGET_LOSS)
    return max(times) + 1000.0 * gap, False, mean_final


def seed_from_head_sweep(study):
    """Inject the head-sweep results (model_dim=768, muon_lr=0.02, adam_lr=8e-3,
    heads ∈ {4,6,8,12}) as completed trials so TPE warm-starts from them."""
    path = DIR / "results_phase2_head_sweep.json"
    if not path.exists():
        print("  (no head-sweep file to seed from)")
        return
    d = json.load(open(path))
    dists = {
        "model_dim":     optuna.distributions.CategoricalDistribution(MODEL_DIMS),
        "num_heads_768": optuna.distributions.CategoricalDistribution(valid_head_counts(768)),
        "muon_lr":       optuna.distributions.FloatDistribution(0.005, 0.05, log=True),
        "adam_lr":       optuna.distributions.FloatDistribution(0.002, 0.02, log=True),
    }
    existing = {tuple(sorted(t.params.items())) for t in study.trials}
    n_added = 0
    for key, seeds in d.items():
        if not seeds:
            continue
        # key looks like "relu2_h{H}x{D}"; recover H from the model config in the log
        cfg = seeds[0]["config"]
        if cfg.get("model_dim") != 768:
            continue
        H = cfg["num_heads"]
        if H not in valid_head_counts(768):
            continue
        params = {"model_dim": 768, "num_heads_768": H, "muon_lr": 0.02, "adam_lr": 0.008}
        if tuple(sorted(params.items())) in existing:
            continue  # already seeded (idempotent on resume)
        obj, reached, final = _objective_from_seeds(seeds)
        study.add_trial(optuna.trial.create_trial(
            params=params, distributions=dists, value=obj,
            user_attrs={"head_dim": 768 // H, "reached_target": reached,
                        "final_loss": final, "seeded_from": "head_sweep"},
        ))
        n_added += 1
        print(f"  seeded: heads={H}x{768//H}  obj={obj:.1f}s  reached={reached}")
    print(f"  seeded {n_added} trials from head sweep")


def objective(trial):
    model_dim = trial.suggest_categorical("model_dim", MODEL_DIMS)
    heads_opts = valid_head_counts(model_dim)
    # define-by-run: head choices are conditional on model_dim
    num_heads = trial.suggest_categorical(f"num_heads_{model_dim}", heads_opts)
    muon_lr   = trial.suggest_float("muon_lr", 0.005, 0.05, log=True)
    adam_lr   = trial.suggest_float("adam_lr", 0.002, 0.02, log=True)

    obj, reached, final = run_eval(model_dim, num_heads, muon_lr, adam_lr, trial.number)
    if obj is None:
        # Hard failure (OOM/crash): prune so it doesn't poison the surrogate
        raise optuna.TrialPruned()

    head_dim = model_dim // num_heads
    trial.set_user_attr("head_dim", head_dim)
    trial.set_user_attr("reached_target", reached)
    trial.set_user_attr("final_loss", final)
    print(f"  trial {trial.number}: dim={model_dim} heads={num_heads}x{head_dim} "
          f"muon_lr={muon_lr:.4f} adam_lr={adam_lr:.4f} → "
          f"{'TTT' if reached else 'CENSORED'} obj={obj:.1f}s final={final:.4f}")
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--no-seed", action="store_true",
                    help="skip warm-starting from the head-sweep results")
    args = ap.parse_args()

    DIR.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=0, multivariate=True, group=True)
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=STUDY_DB,
        direction="minimize", sampler=sampler, load_if_exists=True,
    )
    if not args.no_seed:
        print("Seeding from head sweep:")
        seed_from_head_sweep(study)
    done = len([t for t in study.trials if t.state.is_finished()])
    print(f"Study '{STUDY_NAME}': {done} trials done, target +{args.n_trials}")
    print(f"Objective: seconds to reach val loss ≤ {TARGET_LOSS} "
          f"(avg {SEEDS} seeds, cap {STEP_CAP} steps)\n")

    study.optimize(objective, n_trials=args.n_trials)

    print("\n=== Best trial ===")
    b = study.best_trial
    print(f"  value: {b.value:.1f}s   params: {b.params}")
    print(f"  head_dim={b.user_attrs.get('head_dim')}  "
          f"final_loss={b.user_attrs.get('final_loss'):.4f}  "
          f"reached={b.user_attrs.get('reached_target')}")

    print("\n=== All trials (sorted by objective) ===")
    finished = [t for t in study.trials if t.value is not None]
    for t in sorted(finished, key=lambda t: t.value):
        md = t.params.get("model_dim")
        nh = next((v for k, v in t.params.items() if k.startswith("num_heads_")), "?")
        print(f"  #{t.number:2d}  dim={md} heads={nh}x{t.user_attrs.get('head_dim','?')}  "
              f"muon_lr={t.params.get('muon_lr',0):.4f} adam_lr={t.params.get('adam_lr',0):.4f}  "
              f"obj={t.value:7.1f}s  reached={t.user_attrs.get('reached_target')}")


if __name__ == "__main__":
    main()
