#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ANGLE=120
BRUTE_LOG="timing_brute_sparse_boundary_${ANGLE}.log"
HASH_LOG="timing_hash_sparse_boundary_${ANGLE}.log"

echo "Building"
make

echo "Running brute at ${ANGLE} degrees"
./sph_baseline "$ANGLE" brute | tee "$BRUTE_LOG"

echo "Running hash at ${ANGLE} degrees"
./sph_baseline "$ANGLE" hash | tee "$HASH_LOG"

parse_summary() {
  local mode_name="$1"
  local log_file="$2"

  awk -v mode="$mode_name" '
    /^TimingSummary /{
      for (i = 1; i <= NF; i++) {
        split($i, a, "=")
        key = a[1]
        value = a[2]
        if (key == "total_ms") total_ms = value
        if (key == "avg_ms") avg_ms = value
        if (key == "min_ms") min_ms = value
        if (key == "max_ms") max_ms = value
      }
    }
    END{
      if (total_ms == "") {
        printf "%s summary failed no TimingSummary line found\n", mode
        exit 1
      }

      printf "%s total_ms=%s avg_ms=%s min_ms=%s max_ms=%s\n",
             mode, total_ms, avg_ms, min_ms, max_ms
    }
  ' "$log_file"
}

echo
echo "Timing comparison"
parse_summary "brute" "$BRUTE_LOG"
parse_summary "hash" "$HASH_LOG"