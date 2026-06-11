# Architecture search: iso-compute Bayesian optimization (phase2)

A Bayesian optimization over model architecture + NorMuon learning rates, run on a
single RTX 5060 Ti against the `phase2.py` faithful-replica harness. Separate from
the activation-function study — this asks whether the speedrun's **768-dim / 6-head**
core is compute-optimal, or whether a different width/head split trains more
efficiently.

## Objective evolution (and why)

1. **First attempt — time to a fixed loss target (10.20).** Abandoned. The target
   sits only ~9% of the way from random (10.83) to the speedrun goal (3.28), so it
   rewarded "escapes the random-init plateau fastest," which biases toward whatever
   is cheap per step. An arbitrary target distorts the search.

2. **Final — lowest loss reached in a fixed wall-clock budget (iso-compute).**
   This is the fair speed/quality tradeoff: a config wins only by reaching lower
   loss in the *same* wall-clock, so fast-but-shallow and slow-but-deep models
   compete on equal footing (wall-clock is the speedrun's actual currency). Uses
   the **faithful softcapped CE** the record uses (`23*sigmoid((logits+5)/7.5)`),
   so no fidelity is sacrificed. NorMuon, relu2, `--max-seconds` budget with the
   LR cosine decaying over *time* so the schedule completes regardless of step count.

Search space (4-dim): `model_dim ∈ {512,640,768,896,1024}`, `num_heads` (valid
divisors with head_dim%4==0), `muon_lr ∈ [0.005,0.05]` log, `adam_lr ∈ [0.002,0.02]`
log. (adam_lr matters even under NorMuon — it drives the AdamW path for all
non-matrix params: embeddings, banks, scalars, gates.)

## Methodology

- **Optuna TPE**, SQLite-resumable, record config (768×6) enqueued as the anchor.
- **Adaptive racing seeds:** each config runs 1 seed; only configs within 0.05 / 0.02
  of the best-so-far earn a 2nd / 3rd seed. Extra compute concentrates at the top;
  clearly-bad configs are rejected cheaply at 1 seed.
- **Multi-fidelity promotion:** after the 30-min screen, the top-3 + 1 mid-rank
  control are re-run at a 1-hr budget to test whether the short-budget ranking is
  *budget-stable* (a precondition for trusting it / extrapolating).

## Results — 30-min screen (20 trials)

Under fair iso-compute, **the top 7 configs are all 896 or 1024; the record 768
is 8th–9th.** Six independent 896 configs (head_dim 56/64/112, muon_lr 0.007–0.049)
all land in a tight 9.926–9.966 band — a *plateau*, not a lucky point — ~0.12–0.14
ahead of the record's 10.065.

| rank | model_dim | heads × dim | loss | n |
|------|-----------|-------------|------|---|
| 1 | 896 | 8×112 | 9.926 | 3 |
| 2 | 896 | 8×112 | 9.929 | 3 |
| 3 | 896 | 8×112 | 9.931 | 3 |
| 4 | 896 | 14×64 | 9.944 | 3 |
| 5 | 896 | 16×56 | 9.952 | 3 |
| 6 | 1024 | 32×32 | 9.966 | 3 |
| 8 | 768 | 6×128 (record) | 10.065 | 3 |

Per-step decomposition (best 896 vs record 768, same 30 min):
- 896 did **3,800 steps**; 768 did **5,000 steps** → 896 is **~32% slower per step**
  (bigger model, more FLOPs).
- Yet 896 reached lower loss → its **per-step quality advantage outweighs the speed
  penalty in this early regime.**

Secondary findings:
- **head_dim ≈ 112–128 is the sweet spot**; narrow heads (≤64) are worse at 768,
  but the penalty shrinks at 896 (more capacity → more forgiving of head split).
- **LR is loosely constrained** at 896 (any reasonable muon_lr works). The earlier
  "adam_lr wants to be high" signal from the broken time-to-target BO did NOT
  replicate — it was an artifact of that objective.

## Budget-stability (30-min → 1-hr promotion)

**Verdict: BUDGET-STABLE = True.** The 30-min ranking predicted the 1-hr ranking
exactly, control included.

| config | 30-min | 1-hr | Δ |
|--------|--------|------|---|
| top1 896×8 | 9.9263 | 9.9045 | −0.022 |
| top2 896×8 | 9.9288 | 9.9050 | −0.024 |
| top3 896×8 | 9.9311 | 9.9095 | −0.022 |
| CONTROL 768×4 | 10.0402 | 10.0119 | −0.028 |

**But the gap is *eroding*:** doubling the budget moved everyone only ~0.02–0.03
(we're deep in the softcap plateau), and the 768 control improved *most* (−0.028 vs
896's −0.022). The 896-vs-768 gap went 0.114 (30 min) → **0.107** (1 hr) — it
shrank. Tiny and within n=2 noise, but the *direction* is the early signature of a
possible **crossover**: 896's lead is an early-regime capacity effect that the
data we can afford shows narrowing, not widening, with compute.

## Caveats / what this is NOT

- **Early-undertrained regime.** ~9.9 is ~9% of the way to 3.28. This GPU cannot
  reach a speedrun-relevant loss in feasible time; the proxy measures early dynamics.
- **The early lead does not extrapolate.** As training continues, the per-step
  *quality* edge of extra capacity typically shrinks (if 768 has *enough* capacity
  for the 3.28 solution) while 896's per-step *speed* penalty stays → the ranking
  can reverse. The one budget-doubling we can afford already trends that way.
- **Hardware transfer.** Per-step throughput is hardware/kernel-specific (FA3, FP8,
  the distributed sharding tuned for current shapes). Our SDPA/BF16 wall-clock
  ranking does NOT carry to 8×H100. Only the per-step *quality* ordering is
  partially transferable.
- **FA3 head_dim gap:** the best local config (896×8, head_dim **112**) is NOT an
  FA3-friendly head_dim. The natural H100 candidate **896×7 (head_dim 128)** keeps
  the kernel sweet-spot AND the bigger width — but it was **never evaluated** by the
  BO (it only tried 8/14/16 heads at 896). Any H100 test should include 896×7×128.

## Reproduce

- Screen: `python experiments/trilu/bo_budget_search.py --n-trials 20 --budget 1800`
- Promotion: `python experiments/trilu/bo_promote.py --budget 3600 --seeds 2 --top 3`
- Study DB: `experiments/trilu/bo_budget_study.db` (Optuna SQLite, resumable)
- Promotion results: `experiments/trilu/bo_promotion_results.json`
