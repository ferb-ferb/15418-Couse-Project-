#!/usr/bin/env bash
set -euo pipefail

# run from the directory this script lives in
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ANGLE=120
TAR_NAME="output_120.tar.gz"

echo "Cleaning old binary and old frames"
make clean || true

echo "Ensuring output directory exists"
mkdir -p output

echo "Building"
make

echo "Running simulation at tilt angle ${ANGLE} degrees"
./sph_baseline "$ANGLE" hash

echo "Packing output folder into ${TAR_NAME}"
rm -f "$TAR_NAME"
tar -czf "$TAR_NAME" output

echo "Done"
echo "Created: $SCRIPT_DIR/$TAR_NAME"