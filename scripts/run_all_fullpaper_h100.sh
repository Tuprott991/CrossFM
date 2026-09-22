#!/usr/bin/env bash
set -euo pipefail

# One H100 is intentionally driven by one process at a time. Heavyweight
# backbones use most of the useful device bandwidth, while profile-level output
# directories remain independently resumable.
for profile in \
  author_a_h100_primary \
  author_a_h100_robustness \
  author_a_h100_scale
do
  CROSSFM_PROFILE="$profile" \
  CROSSFM_OUTPUT="${CROSSFM_OUTPUT_ROOT:-outputs/fullpaper}/${profile}" \
  bash scripts/run_fullpaper_h100.sh
done
