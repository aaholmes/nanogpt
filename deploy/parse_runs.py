"""Parse train_gpt.py A/B logs -> wallclock & steps to reach val_loss <= 3.28.

    python deploy/parse_runs.py runs_ab/*.log

Reads the standard log line:
    step:N/M val_loss:X.XXXX train_time:Yms step_avg:Zms
Reports, per activation and per seed, the first step (and wallclock) at which validation
loss crosses the 3.28 target, then the paired relu2_s1 - relu2 delta. The benchmark metric
is wallclock-to-target; steps-to-target is the hardware-independent companion.
"""
import sys, re, os
import statistics as st

TARGET = 3.28
PAT = re.compile(r"step:(\d+)/\d+\s+val_loss:([0-9.]+)\s+train_time:([0-9.]+)ms")


def act_of(f):
    b = os.path.basename(f)
    for a in ("relu2_s1", "sniqu", "relu2"):   # check compound/distinct names before "relu2"
        if a in b:
            return a
    return b


def seed_of(f):
    m = re.search(r"seed(\d+)", os.path.basename(f))
    return int(m.group(1)) if m else -1


def parse(f):
    hit = final = None
    for line in open(f):
        m = PAT.search(line)
        if not m:
            continue
        step, val, t = int(m.group(1)), float(m.group(2)), float(m.group(3))
        final = (step, val, t)
        if hit is None and val <= TARGET:
            hit = (step, val, t)
    return hit, final


def fmt(h):
    return f"{h[0]:>5} steps / {h[2]/1000:6.1f}s  (val {h[1]:.4f})" if h else "did NOT reach 3.28"


rows = {}
for f in sys.argv[1:]:
    hit, final = parse(f)
    rows.setdefault(act_of(f), {})[seed_of(f)] = {"hit": hit, "final": final}

for act in sorted(rows):
    print(f"\n[{act}]")
    for seed in sorted(rows[act]):
        r = rows[act][seed]
        extra = "" if r["hit"] else f"   (final: {fmt(r['final'])})"
        print(f"  seed {seed}: {fmt(r['hit'])}{extra}")

# each candidate paired against the relu2 baseline (wallclock-to-3.28)
if "relu2" in rows:
    for cand in [a for a in sorted(rows) if a != "relu2"]:
        seeds = sorted(set(rows["relu2"]) & set(rows[cand]))
        dsec, dstep = [], []
        print(f"\n[paired {cand} - relu2]")
        for s in seeds:
            base, c = rows["relu2"][s]["hit"], rows[cand][s]["hit"]
            if base and c:
                dsec.append((c[2] - base[2]) / 1000.0)
                dstep.append(c[0] - base[0])
                print(f"  seed {s}: {dsec[-1]:+.1f}s   {dstep[-1]:+d} steps")
            else:
                print(f"  seed {s}: incomplete (a variant did not reach 3.28)")
        if dsec:
            msec = sum(dsec) / len(dsec)
            mstep = sum(dstep) / len(dstep)
            tail = f", sd={st.stdev(dsec):.1f}s" if len(dsec) > 1 else ""
            print(f"  MEAN: {cand} - relu2 = {msec:+.1f}s  ({mstep:+.0f} steps)  to 3.28  "
                  f"(n={len(dsec)} paired{tail})  "
                  f"-> {'FASTER' if msec < 0 else 'not faster'}")
