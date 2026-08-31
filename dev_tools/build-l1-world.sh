#!/bin/bash

#  Copyright 2026 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

################################################################################
# Build and regenerate the Qualtran-L1 world library under `qualtran/l1/world`.
#
# Usage:
#     uv run dev_tools/build-l1-world.sh
################################################################################

set -e

# Change directory to the repository root directory
thisdir="$(dirname "${BASH_SOURCE[0]}")" || exit $?
topdir="$(git -C "${thisdir}" rev-parse --show-toplevel)" || exit $?
cd "${topdir}" || exit $?

# Ensure destination directory exists
mkdir -p qualtran/l1/world

python dev_tools/build-l1-library.py \
    --regenerate \
    qualtran/l1/world \
    --report qualtran/l1/world/report.md \
    "$@" \
    2> qualtran/l1/world/compile.stderr
