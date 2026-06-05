# RUNBOOK — relu2_s1 A/B on rented 8×H100 (Gate 3)

**Goal:** measure whether `relu2_s1` reaches val loss 3.28 *faster* than the incumbent
`relu2` on the **real** stack (FP8, fused kernels, full data schedule), to decide whether a
full record attempt is warranted. **Budget ≈ $50–100, ~2–4 h** (matches the README Phase-2
estimate). Runs themselves are minutes each; the cost is box-up + data + iteration.

## Already integrated on this branch (validated locally)
- `triton_kernels.py` — `ACT` constexpr in the fused MLP kernel: `0` = `relu(z)²` (incumbent,
  default), `1` = `relu(z)²+relu(z)` (relu2_s1). Same matmuls, same cost; only the fused
  elementwise act/derivative differ. Verified by `experiments/trilu/test_fused_mlp.py`
  (fp64 gradcheck + the real Triton kernel vs eager, fwd+bwd).
- `train_gpt.py` — `NANOGPT_MLP_ACT` env (`relu2` default / `relu2_s1`); opt-in `SEED` env for
  paired A/B. **With neither env set, the run is byte-for-byte the record baseline.**

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
```
Confirm both train (val decreasing) and the `relu2` arm matches the known baseline curve.

**6. A/B:** `SEEDS=3 bash deploy/run_compare.sh`  (3 paired runs each = 6 runs, minutes apiece).

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
- **Candidate scope:** `relu2_s1` is the only *same-cost* candidate in this fused kernel.
  `eluquad`/`sniqu` would need a reciprocal inside the kernel (extra cost) — out of scope here;
  revisit only if the local grid shows them clearly ahead *and* the cost can be hidden.
- **What this gate cannot skip:** it *is* the transfer test (loss regime → 3.28, large batch,
  FP8, real arch). Don't go local-grid → record without it.
