#!/usr/bin/env bash
# Run qlt_fastsim of negate-large.qlt on the sample 2048-bit integer.
#
# This demonstrates large bitsize handling. You can inspect the result:
# it should be the negation of the input integer.
#
# With --timing, this will print timing info to stderr.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QLT_FASTSIM="${QLT_FASTSIM:-cargo run --release --bin qlt_fastsim --no-default-features --}"

echo "input:"
cat "$SCRIPT_DIR/negate-large.in.txt"

echo "output:"
$QLT_FASTSIM --timing "$SCRIPT_DIR/negate-large.qlt" Negate \
    < "$SCRIPT_DIR/negate-large.in.txt"
