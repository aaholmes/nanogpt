"""Plot Phase 1 TriLU results from results.json."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "relu2": "#888888",
    "gelu": "#2c5aa0",
    "trilu_sym": "#5fa84e",
    "trilu_asym": "#1a8a4a",
    "swiglu": "#cc4530",
    "geglu": "#d97f3f",
    "triglu": "#7a2d8c",
}


def plot_val_loss(results, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for act, runs in results.items():
        steps = np.array([s for s, _ in runs[0]["val_loss"]])
        losses = np.array([[l for _, l in r["val_loss"]] for r in runs])
        mean = losses.mean(axis=0)
        std = losses.std(axis=0)
        color = COLORS.get(act, "#444444")
        ax.plot(steps, mean, label=act, color=color, linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.18, color=color)
    ax.set_xlabel("step")
    ax.set_ylabel("val loss")
    ax.set_title("Phase 1: val loss vs step (mean ± std across seeds)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Saved {out_path}")


def plot_trilu_trajectories(results, out_path):
    """Plot how (L, R, alpha) drift over training, per layer, averaged across seeds."""
    trilu_acts = [a for a in results if a in ("trilu_sym", "trilu_asym", "triglu")]
    if not trilu_acts:
        print("No TriLU activations in results; skipping trajectory plot.")
        return

    fig, axes = plt.subplots(len(trilu_acts), 3, figsize=(13, 4 * len(trilu_acts)), squeeze=False)
    for row, act in enumerate(trilu_acts):
        runs = results[act]
        # Collect: traj[layer_name][seed] -> list of (step, L, R, alpha)
        traj = defaultdict(lambda: defaultdict(list))
        for seed_idx, r in enumerate(runs):
            for step, params in r["trilu_params"]:
                for p in params:
                    traj[p["layer"]][seed_idx].append((step, p["L"], p["R"], p["alpha"]))

        for col, (key, title) in enumerate([("L", "L (left breakpoint)"),
                                            ("R", "R (right breakpoint)"),
                                            ("alpha", "α (curvature)")]):
            ax = axes[row, col]
            for layer_idx, (layer_name, seed_data) in enumerate(sorted(traj.items())):
                # Average across seeds
                seeds = list(seed_data.values())
                steps = np.array([t[0] for t in seeds[0]])
                vals_idx = {"L": 1, "R": 2, "alpha": 3}[key]
                vals = np.array([[t[vals_idx] for t in s] for s in seeds]).mean(axis=0)
                ax.plot(steps, vals, label=f"L{layer_idx}", linewidth=1.5)
            ax.set_xlabel("step")
            ax.set_ylabel(title)
            ax.set_title(f"{act}: {title}")
            ax.grid(True, alpha=0.25)
            if col == 2 and row == 0:
                ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/trilu/results.json")
    ap.add_argument("--out-dir", default="experiments/trilu")
    args = ap.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_val_loss(results, out_dir / "val_loss.png")
    plot_trilu_trajectories(results, out_dir / "trilu_trajectories.png")


if __name__ == "__main__":
    main()
