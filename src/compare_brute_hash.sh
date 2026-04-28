#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ANGLE=120
BRUTE_LOG="timing_brute_${ANGLE}.log"
HASH_LOG="timing_hash_${ANGLE}.log"

echo "Building"
make

echo "Running brute at ${ANGLE} degrees"
./sph_baseline "$ANGLE" brute | tee "$BRUTE_LOG"

echo "Running hash at ${ANGLE} degrees"
./sph_baseline "$ANGLE" hash | tee "$HASH_LOG"

summarize_log() {
  local mode_name="$1"
  local log_file="$2"

  awk -v mode="$mode_name" '
    /^FrameTiming /{
      count++
      t = $3
      sum += t
      if (count == 1 || t < min) min = t
      if (count == 1 || t > max) max = t
    }
    END{
      if (count == 0) {
        printf "%s summary failed no FrameTiming lines found\n", mode
        exit 1
      }
      printf "%s total_ms=%.3f avg_ms=%.3f min_ms=%.3f max_ms=%.3f frames=%d\n",
             mode, sum, sum / count, min, max, count
    }
  ' "$log_file"
}

echo
echo "Timing comparison"
summarize_log "brute" "$BRUTE_LOG"
summarize_log "hash" "$HASH_LOG"