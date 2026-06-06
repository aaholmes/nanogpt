# RUNBOOK — relu2_s1 A/B on rented 8×H100 (Gate 3)

**Goal:** measure whether `relu2_s1` reaches val loss 3.28 *faster* than the incumbent
`relu2` on the **real** stack (FP8, fused kernels, full data schedule), to decide whether a
full record attempt is warranted. **Budget ≈ $50–100, ~2–4 h** (matches the README Phase-2
estimate). Runs themselves are minutes each; the cost is box-up + data + iteration.

## Already integrated on this branch (validated locally)
- `triton_kernels.py` — `ACT` constexpr in the fused MLP kernel: `0` = `relu(z)²` (incumbent,
  default), `1` = `relu(z)²+relu(z)` (relu2_s1), `2` = `0.560·(iqu(z)−0.706)` (sniqu).
  The kernel is matmul-bound, so the pointwise choice (incl. sniqu's reciprocal) is ~free —
  all three are same-cost. Verified by `experiments/trilu/test_fused_mlp.py` (fp64 gradcheck +
  the real Triton kernel vs eager, fwd+bwd, all three activations).
- `train_gpt.py` — `NANOGPT_MLP_ACT` env (`relu2` default / `relu2_s1` / `sniqu`); opt-in `SEED`
  env for paired A/B. **With neither env set, the run is byte-for-byte the record baseline.**

## Steps

**0. Provision** an 8×H100 box (PrimeIntellect, per README), CUDA 12.6.

**1. Code:** `git clone <this branch>; cd modded-nanogpt`

**2. Environment** (pick one):
- Docker: `docker build -f deploy/Dockerfile -t nanogpt-relu2s1 .` then
  `docker run -it --rm --gpus all -v $(pwd):/workspace nanogpt-relu2s1 bash`
- Bare: `pip install -r requirements.txt` (ensure a torch 2.10 nightly cu126).
> For perf parity with a specific record, pin torch to **that record's exact nightly** (README
> record table). The A/B *delta* is robust to the exact build since both arms share it.

**3. Stage data:** `bash deploy/setup_data.sh 10`  (README uses 9; bump if a run runs past the
last shard.)

**4. SMOKE A — kernel at production tiles** (the one thing un-testable on consumer GPUs; our
5060 Ti caps at 101 KB shared mem, H100 tiles need 180 KB):
```
python experiments/trilu/test_fused_mlp.py --gpu       # TEST_* unset -> BM128/BN256/stages4
```
Must print `ALL PASS`. This validates the exact kernel numerics that will run.

**5. SMOKE B — baseline unchanged + candidate trains** (run a few val cycles, Ctrl-C):
```
NANOGPT_MLP_ACT=relu2    SEED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
NANOGPT_MLP_ACT=relu2_s1 SEED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
NANOGPT_MLP_ACT=sniqu    SEED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```
Confirm all train (val decreasing) and the `relu2` arm matches the known baseline curve.

**6. A/B:** `SEEDS=3 bash deploy/run_compare.sh`  (3 paired runs × 3 arms = 9 runs, minutes
apiece). Restrict arms with e.g. `VARIANTS="relu2 relu2_s1" SEEDS=3 bash deploy/run_compare.sh`.

**7. Decide** (auto-printed by `deploy/parse_runs.py`): wallclock & steps to 3.28 per variant +
the paired delta.
- **GO** (full record attempt) if `relu2_s1` reaches 3.28 in fewer steps / less wallclock,
  the gap holds across seeds, **and** the `relu2` arm reproduces the known record time
  (confirms the harness is faithful).
- **NO-GO** otherwise.

**8. Teardown:** stop the box.

## Notes / caveats
- **Official submissions** report the median of several runs with *no* fixed seed — `SEED` here
  exists only to make *our* A/B paired (lower variance). Drop `SEED` for a record submission.
- **Data size:** if a run aborts reading past the last shard, `bash deploy/setup_data.sh 20`.
- **Candidate scope:** `relu2_s1` and `sniqu` are both integrated and **same-cost** (the kernel
  is matmul-bound, so sniqu's extra reciprocal is negligible). `relu2_s1` is the minimal change
  off the incumbent; `sniqu` led the local grid at seed 0. The A/B runs all three so the H100
  picks the winner on wallclock-to-3.28 directly — no cost thumb on the scale.
- **What this gate cannot skip:** it *is* the transfer test (loss regime → 3.28, large batch,
  FP8, real arch). Don't go local-grid → record without it.
