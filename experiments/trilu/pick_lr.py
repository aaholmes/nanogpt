"""Pick a Muon LR by fitting a parabola to (log10 LR, smoothed val loss).

Reads results_muon_probe_<act>_lr<LR>.json files for one activation, fits a
least-squares quadratic in log10(LR), and prints the vertex (the loss-minimizing
LR). Falls back to the best grid point if the fit isn't convex. Clamped to a sane
range so a near-edge minimum extrapolates only modestly. Averages over seeds if
the probe has more than one.
"""
import glob, json, sys
import numpy as np

act = sys.argv[1]
lo, hi = 0.003, 0.05  # clamp range for the picked LR

lrs, losses = [], []
for f in sorted(glob.glob(f"experiments/trilu/results_muon_probe_{act}_lr*.json")):
    lr = float(f.split("_lr")[1].replace(".json", ""))
    d = json.load(open(f))
    seeds = list(d.values())[0]
    s = np.mean([r.get("final_loss_smoothed", r["val_loss"][-1][1]) for r in seeds])
    lrs.append(lr); losses.append(s)

lrs, losses = np.array(lrs), np.array(losses)
order = np.argsort(lrs); lrs, losses = lrs[order], losses[order]
x = np.log10(lrs)
best_grid = lrs[int(np.argmin(losses))]

if len(lrs) >= 3:
    a, b, c = np.polyfit(x, losses, 2)
    if a > 0:
        xv = np.clip(-b / (2 * a), np.log10(lo), np.log10(hi))
        print(f"{10**xv:.4f}")
        sys.exit(0)
print(f"{best_grid:.4f}")
