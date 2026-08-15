#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
