#!/usr/bin/env bash
# Run qlt_fastsim of cswap.qlt with the test inputs from cswap.in.txt.
#
# CSwap is a simple routine that swaps two QAny(5) registers depending
# on the state of the ctrl QBit.
#
# qlt_fastsim natively supports '#' comments in input files.
# Output is displayed as a side-by-side table of inputs and outputs.
#
# You can inspect the output: the x, y inputs should be swapped iff ctrl=1
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QLT_FASTSIM="${QLT_FASTSIM:-cargo run --release --bin qlt_fastsim --no-default-features --}"

INPUT_LINES=$(grep -v '^#' "$SCRIPT_DIR/cswap.in.txt" | grep -v '^$')
OUTPUT_LINES=$($QLT_FASTSIM "$SCRIPT_DIR/cswap.qlt" CSwap \
    < "$SCRIPT_DIR/cswap.in.txt" 2>/dev/null \
    | grep -v '^#')

# Build aligned table with | separator between in and out columns
TABLE=$({
    echo "ctrl x y | phase_x ctrl x y"
    paste <(echo "$INPUT_LINES") <(echo "$OUTPUT_LINES") \
        | awk '{ print $1, $2, $3, "|", $4, $5, $6, $7 }'
} | column -t)

# Print "in"/"out" group headers, aligned with the | separator
SEP_POS=$(head -1 <<< "$TABLE" | grep -ob '|' | head -1 | cut -d: -f1)
printf "%-${SEP_POS}s  %s\n" "in" " out"

# Print the table (first line is column headers, rest is data)
echo "$TABLE"
