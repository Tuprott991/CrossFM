#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CROSSFM_CONFIG:-configs/fullpaper.yaml}"
PROFILE="${CROSSFM_PROFILE:-author_a_h100_primary}"
OUTPUT="${CROSSFM_OUTPUT:-outputs/fullpaper/${PROFILE}}"
DATA_ROOT="${CROSSFM_DATA_ROOT:-data/fullpaper}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export HF_XET_HIGH_PERFORMANCE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

python -m crossfm.fullpaper.cli plan --config "$CONFIG" --profile "$PROFILE" --output "$OUTPUT" --data-root "$DATA_ROOT" --world-size 1
python -m crossfm.fullpaper.cli doctor --config "$CONFIG" --profile "$PROFILE" --output "$OUTPUT" --data-root "$DATA_ROOT"
for stage in response_bank llm_cache evaluate; do
  python -m crossfm.fullpaper.cli run-worker --config "$CONFIG" --profile "$PROFILE" --output "$OUTPUT" --data-root "$DATA_ROOT" --stage "$stage" --rank 0 --world-size 1
done
python -m crossfm.fullpaper.cli aggregate --config "$CONFIG" --profile "$PROFILE" --output "$OUTPUT" --data-root "$DATA_ROOT"
