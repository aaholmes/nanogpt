#!/bin/bash
# Stage FineWeb10B GPT-2 tokens (pre-tokenized .bin shards) from HF kjj0/fineweb10B-gpt2.
# Each train chunk ~= 100M tokens (~200 MB). The README reference run uses 9; we default to
# 10 for headroom. Bump if a run aborts having read past the last shard.
set -euo pipefail
N="${1:-10}"
cd "$(dirname "$0")/.."
echo "[setup_data] downloading $N train chunks + val shard into data/fineweb10B/ ..."
python data/cached_fineweb10B.py "$N"
echo "[setup_data] staged $(ls data/fineweb10B/fineweb_train_*.bin 2>/dev/null | wc -l) train shards + val"

# To skip the re-download on future rented boxes, cache the staged dir to object storage once:
#   tar -C data -cf - fineweb10B | zstd -T0 -o fineweb10B.tar.zst   &&   <upload to S3/R2/GCS>
# Then on a fresh box, before running:
#   <download> fineweb10B.tar.zst && zstd -d -c fineweb10B.tar.zst | tar -C data -xf -
