"""
Joint architecture + topology Bayesian optimization (Optuna TPE), minimizing
wall-clock time to reach a target validation loss.

Search space = the 4-dim study's dims PLUS the generative topology dims:
    model_dim, num_heads (conditional), muon_lr, adam_lr        (as in bo_search)
    num_layers      ∈ {8,9,10,11,12,13,14,16}
    num_skips       ∈ {0,1,2,3}                forward skips (each dst drops attn)
    skip_src_frac   ∈ [0.15, 0.50]
    skip_span_frac  ∈ [0.10, 0.45]
    backout_src_frac∈ [0.50, 0.95]
    backout_mode    ∈ {none, freeze_only, freeze_subtract}

Warm start: every completed trial of the 4-dim study is replayed here as a
fully-specified observation with the topology dims PINNED to their legacy values
(num_layers=11, num_skips=1, skip_src_frac=0.30, skip_span_frac=0.30,
 backout_src_frac=0.70, backout_mode=freeze_subtract) — because those trials ran
on the legacy topology. They are coplanar (all at legacy topology) so they give
no gradient along the topology axes, but they densely anchor the (width, LR)
sub-landscape and act as a mild "legacy is good" prior. The head sweep is already
absorbed into the 4-dim study, so it transfers transitively.

build_topology(11, legacy-core params) == legacy_topology() exactly, so the
injected points are consistent with what the topology search evaluates near the
legacy region.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import optuna

import bo_search as b4          # reuse 4-dim config + helpers
from topology import build_topology, BACKOUT_MODES

DIR = b4.DIR
STUDY_DB = f"sqlite:///{DIR}/bo_topo_study.db"
STUDY_NAME = "phase2_arch_topology_joint"

NUM_LAYERS_OPTS = [8, 9, 10, 11, 12, 13, 14, 16]
NUM_SKIPS_OPTS  = [0, 1, 2, 3]

# Legacy topology values, for pinning the replayed 4-dim observations.
LEGACY_TOPO_PARAMS = dict(
    num_layers=11, num_skips=1, skip_src_frac=0.30, skip_span_frac=0.30,
    backout_src_frac=0.70, backout_mode="freeze_subtract",
)


def _topo_distributions(model_dim):
    """Distribution dict for a trial at a given model_dim (define-by-run shape)."""
    return {
        "model_dim":      optuna.distributions.CategoricalDistribution(b4.MODEL_DIMS),
        f"num_heads_{model_dim}":
                          optuna.distributions.CategoricalDistribution(b4.valid_head_counts(model_dim)),
        "muon_lr":        optuna.distributions.FloatDistribution(0.005, 0.05, log=True),
        "adam_lr":        optuna.distributions.FloatDistribution(0.002, 0.02, log=True),
        "num_layers":     optuna.distributions.CategoricalDistribution(NUM_LAYERS_OPTS),
        "num_skips":      optuna.distributions.CategoricalDistribution(NUM_SKIPS_OPTS),
        "skip_src_frac":  optuna.distributions.FloatDistribution(0.15, 0.50),
        "skip_span_frac": optuna.distributions.FloatDistribution(0.10, 0.45),
        "backout_src_frac": optuna.distributions.FloatDistribution(0.50, 0.95),
        "backout_mode":   optuna.distributions.CategoricalDistribution(list(BACKOUT_MODES)),
    }


def run_eval(p, trial_number):
    """Run phase2 with architecture + topology params; return (obj, reached, final)."""
    model_dim, num_heads = p["model_dim"], p["num_heads"]
    head_dim = model_dim // num_heads
    out_path = DIR / f"bo_topo_trial_{trial_number:03d}.json"
    cmd = [
        sys.executable, "experiments/trilu/phase2.py",
        "--config", "speedrun", "--compile", "--batch-size", "8",
        "--paired-heads", "--ce-chunk", "4096", "--optimizer", "normuon",
        "--model-dim", str(model_dim), "--num-heads", str(num_heads),
        "--head-dim", str(head_dim),
        "--muon-lr", f"{p['muon_lr']:.6f}", "--adam-lr", f"{p['adam_lr']:.6f}",
        "--num-layers", str(p["num_layers"]),
        "--num-skips", str(p["num_skips"]),
        "--skip-src-frac", f"{p['skip_src_frac']:.4f}",
        "--skip-span-frac", f"{p['skip_span_frac']:.4f}",
        "--backout-src-frac", f"{p['backout_src_frac']:.4f}",
        "--backout-mode", p["backout_mode"],
        "--activations", "relu2", "--steps", str(b4.STEP_CAP),
        "--seeds", str(b4.SEEDS), "--target-loss", str(b4.TARGET_LOSS),
        "--stop-at-target", "--out", str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=7200)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        tail = (e.stderr or "")[-400:] if hasattr(e, "stderr") else ""
        print(f"  trial {trial_number} FAILED ({type(e).__name__}): {tail}")
        return None, False, None
    d = json.load(open(out_path))
    return b4._objective_from_seeds(next(iter(d.values())))


def objective(trial):
    model_dim = trial.suggest_categorical("model_dim", b4.MODEL_DIMS)
    num_heads = trial.suggest_categorical(f"num_heads_{model_dim}",
                                          b4.valid_head_counts(model_dim))
    p = dict(
        model_dim=model_dim, num_heads=num_heads,
        muon_lr=trial.suggest_float("muon_lr", 0.005, 0.05, log=True),
        adam_lr=trial.suggest_float("adam_lr", 0.002, 0.02, log=True),
        num_layers=trial.suggest_categorical("num_layers", NUM_LAYERS_OPTS),
        num_skips=trial.suggest_categorical("num_skips", NUM_SKIPS_OPTS),
        skip_src_frac=trial.suggest_float("skip_src_frac", 0.15, 0.50),
        skip_span_frac=trial.suggest_float("skip_span_frac", 0.10, 0.45),
        backout_src_frac=trial.suggest_float("backout_src_frac", 0.50, 0.95),
        backout_mode=trial.suggest_categorical("backout_mode", list(BACKOUT_MODES)),
    )
    obj, reached, final = run_eval(p, trial.number)
    if obj is None:
        raise optuna.TrialPruned()
    trial.set_user_attr("reached_target", reached)
    trial.set_user_attr("final_loss", final)
    print(f"  #{trial.number}: dim={model_dim} heads={num_heads} L={p['num_layers']} "
          f"skips={p['num_skips']} backout={p['backout_mode']} "
          f"muon={p['muon_lr']:.4f} adam={p['adam_lr']:.4f} -> "
          f"{'TTT' if reached else 'CENSORED'} obj={obj:.1f}s final={final:.4f}")
    return obj


def replay_4d(study):
    """Inject every completed 4-dim trial as a legacy-topology-pinned observation."""
    try:
        src = optuna.load_study(study_name=b4.STUDY_NAME, storage=b4.STUDY_DB)
    except Exception as e:
        print(f"  (no 4-dim study to replay: {e})")
        return
    existing = {tuple(sorted(t.params.items())) for t in study.trials}
    n = 0
    for t in src.trials:
        if t.value is None:
            continue
        params = dict(t.params)
        params.update(LEGACY_TOPO_PARAMS)
        if tuple(sorted(params.items())) in existing:
            continue
        md = params.get("model_dim")
        if md is None:
            continue
        study.add_trial(optuna.trial.create_trial(
            params=params, distributions=_topo_distributions(md), value=t.value,
            user_attrs={**t.user_attrs, "replayed_from": "bo_4d", "pinned": "legacy_topology"},
        ))
        n += 1
    print(f"  replayed {n} legacy-topology-pinned trials from the 4-dim study")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--no-replay", action="store_true",
                    help="skip warm-starting from the 4-dim study")
    args = ap.parse_args()

    DIR.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=0, multivariate=True, group=True)
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=STUDY_DB,
        direction="minimize", sampler=sampler, load_if_exists=True,
    )
    if not args.no_replay:
        print("Warm-starting from the 4-dim study:")
        replay_4d(study)
    done = len([t for t in study.trials if t.state.is_finished()])
    print(f"Study '{STUDY_NAME}': {done} observations present, running +{args.n_trials} live trials")
    print(f"Objective: seconds to val loss <= {b4.TARGET_LOSS} "
          f"(avg {b4.SEEDS} seeds, cap {b4.STEP_CAP} steps)\n")

    study.optimize(objective, n_trials=args.n_trials)

    print("\n=== Best trial ===")
    bt = study.best_trial
    print(f"  value: {bt.value:.1f}s   params: {bt.params}")
    print(f"  final_loss={bt.user_attrs.get('final_loss')}  reached={bt.user_attrs.get('reached_target')}")

    print("\n=== Top 15 (live + replayed) by objective ===")
    finished = [t for t in study.trials if t.value is not None]
    for t in sorted(finished, key=lambda t: t.value)[:15]:
        md = t.params.get("model_dim")
        nh = next((v for k, v in t.params.items() if k.startswith("num_heads_")), "?")
        src = t.user_attrs.get("replayed_from", "live")
        print(f"  obj={t.value:7.1f}s  dim={md} heads={nh} L={t.params.get('num_layers')} "
              f"skips={t.params.get('num_skips')} backout={t.params.get('backout_mode')}  [{src}]")


if __name__ == "__main__":
    main()
