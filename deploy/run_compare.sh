#!/bin/bash
# Paired A/B on 8xH100: incumbent relu2 vs candidate relu2_s1.
# Each pair shares a SEED (matched init+data order) -> paired comparison, lower variance.
# Seeds OUTER, variants INNER (project convention) so a 1-pair read is available after the
# first two runs.  Prereqs: deps installed (deploy/Dockerfile) and data staged (setup_data.sh).
#
#   SEEDS=3 bash deploy/run_compare.sh        # 3 paired A/B runs (6 runs total)
set -u
SEEDS="${SEEDS:-3}"
NPROC="${NPROC:-8}"
OUT="${OUT:-runs_ab}"
mkdir -p "$OUT"
cd "$(dirname "$0")/.."

run() {  # $1 = relu2|relu2_s1   $2 = seed
  local act="$1" seed="$2" log="$OUT/${act}_seed${seed}.log"
  echo "[ab] $act  seed=$seed  -> $log  ($(date))"
  SEED="$seed" NANOGPT_MLP_ACT="$act" \
    torchrun --standalone --nproc_per_node="$NPROC" train_gpt.py 2>&1 | tee "$log"
}

for s in $(seq 0 $((SEEDS-1))); do
  for act in relu2 relu2_s1; do
    run "$act" "$s"
  done
done

echo "=================== A/B SUMMARY ==================="
python deploy/parse_runs.py "$OUT"/*.log
