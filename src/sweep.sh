#!/usr/bin/env bash
set -euo pipefail

# move into the source folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# set the run values
ANGLE=120
FREQUENCIES=(1 2 3 4 5 7 10)
LOG_DIR="neighbor_rebuild_sweep_logs"

# build the code
echo "Building"
make clean
make

# reset the logs
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

# reset output for each run
reset_output_dir() {
  rm -rf output
  mkdir -p output
}

# print one summary line
print_summary() {
  local freq="$1"
  local log_file="$2"

  awk -v freq="$freq" '
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
        if (key == "neighbor_rebuild_calls") rebuild_calls = value
      }
    }
    END{
      printf "freq=%s total_ms=%s avg_ms=%s min_ms=%s max_ms=%s boundary_ms=%s neighbor_ms=%s density_ms=%s force_ms=%s integration_ms=%s rebuild_calls=%s\n",
             freq, total_ms, avg_ms, min_ms, max_ms, boundary_ms, neighbor_ms, density_ms, force_ms, integration_ms, rebuild_calls
    }
  ' "$log_file"
}

# print one overflow line
print_overflow() {
  local freq="$1"
  local log_file="$2"

  awk -v freq="$freq" '
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
      printf "freq=%s overflow_fluid=%s overflow_source=%s overflow_receiver=%s\n",
             freq, fluid, source, receiver
    }
  ' "$log_file"
}

# run the sweep
for freq in "${FREQUENCIES[@]}"; do
  log_file="${LOG_DIR}/neighbor_freq_${freq}.log"

  echo
  echo "Running neighbor_list with rebuild frequency ${freq}"
  reset_output_dir
  ./sph_baseline "$ANGLE" neighbor_list "$freq" | tee "$log_file" > /dev/null
done

echo
echo "Timing comparison"
for freq in "${FREQUENCIES[@]}"; do
  log_file="${LOG_DIR}/neighbor_freq_${freq}.log"
  print_summary "$freq" "$log_file"
done

echo
echo "Overflow comparison"
for freq in "${FREQUENCIES[@]}"; do
  log_file="${LOG_DIR}/neighbor_freq_${freq}.log"
  print_overflow "$freq" "$log_file"
done

echo
echo "Logs are in ${LOG_DIR}"