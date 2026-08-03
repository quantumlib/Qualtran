#!/usr/bin/env bash
# Run qlt_fastsim of product.qlt on sample inputs and check against the reference file.
#
# This does a simple diff of stdout to compare vs reference, which might not be great for
# debugging incorrect behavior, but is a straightforward check of (existing) correctness.
#
# The simulation is correct if the diff results in no output on stdout. Timing information
# is reported to stderr.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QLT_FASTSIM="${QLT_FASTSIM:-cargo run --release --bin qlt_fastsim --no-default-features --}"

$QLT_FASTSIM --timing "$SCRIPT_DIR/product.qlt" Product \
    < "$SCRIPT_DIR/product.in.txt" \
    | diff - "$SCRIPT_DIR/product.out.ref.txt"
