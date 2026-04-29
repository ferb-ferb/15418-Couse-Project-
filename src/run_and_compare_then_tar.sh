#!/usr/bin/env bash
set -euo pipefail

# move into the source folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# set the run values
ANGLE=120
OUTPUT_DIR="output"
BRUTE_LOG="timing_brute_${ANGLE}.log"
NEIGHBOR_LOG="timing_neighbor_${ANGLE}.log"
TAR_NAME="neighbor_output_${ANGLE}.tar.gz"

# build the code
echo "Building"
make clean
make

# clear old logs and tar
rm -f "$BRUTE_LOG" "$NEIGHBOR_LOG" "$TAR_NAME"

# helper to reset output folder
reset_output_dir() {
  rm -rf "$OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"
}

# run brute
echo
echo "Running brute at ${ANGLE} degrees"
reset_output_dir
./sph_baseline "$ANGLE" brute | tee "$BRUTE_LOG"

# run neighbor
echo
echo "Running neighbor at ${ANGLE} degrees"
reset_output_dir
./sph_baseline "$ANGLE" neighbor | tee "$NEIGHBOR_LOG"

# tar the whole output folder
echo
echo "Creating tar archive ${TAR_NAME}"
tar -czf "$TAR_NAME" -C "$(dirname "$OUTPUT_DIR")" "$(basename "$OUTPUT_DIR")"

# print timing summary from one log
print_summary() {
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
        if (key == "boundary_rebuild_ms") boundary_ms = value
        if (key == "neighbor_build_ms") neighbor_ms = value
        if (key == "density_ms") density_ms = value
        if (key == "force_ms") force_ms = value
        if (key == "integration_ms") integration_ms = value
      }
    }
    END{
      if (total_ms == "") {
        printf "%s summary failed no TimingSummary line found\n", mode
        exit 1
      }

      printf "%s total_ms=%s avg_ms=%s min_ms=%s max_ms=%s boundary_ms=%s neighbor_ms=%s density_ms=%s force_ms=%s integration_ms=%s\n",
             mode, total_ms, avg_ms, min_ms, max_ms, boundary_ms, neighbor_ms, density_ms, force_ms, integration_ms
    }
  ' "$log_file"
}

# print overflow summary from one log
print_overflow() {
  local mode_name="$1"
  local log_file="$2"

  awk -v mode="$mode_name" '
    /^NeighborOverflowSummary /{
      for (i = 1; i <= NF; i++) {
        split($i, a, "=")
        key = a[1]
        value = a[2]

        if (key == "fluid_particles") fluid = value
        if (key == "source_boundary_particles") source = value
        if (key == "receiver_boundary_particles") receiver = value
      }
    }
    END{
      if (fluid == "") {
        printf "%s overflow summary not found\n", mode
      } else {
        printf "%s overflow fluid=%s source=%s receiver=%s\n",
               mode, fluid, source, receiver
      }
    }
  ' "$log_file"
}

# print comparison
echo
echo "Timing comparison"
print_summary "brute" "$BRUTE_LOG"
print_summary "neighbor" "$NEIGHBOR_LOG"

echo
echo "Overflow comparison"
print_overflow "brute" "$BRUTE_LOG"
print_overflow "neighbor" "$NEIGHBOR_LOG"

echo
echo "Created ${TAR_NAME}"
echo "Archive contains folder $(basename "$OUTPUT_DIR")"