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

# This script will generate "backwards compatible" environment lock files.
# Consider using `uv` to manage your environment if you need this level of locking and control.
#
# This script assumes you have already used `uv lock` to generate a `uv.lock` file in the
# project's root directory.

set -ex

uvoptions=(--project ../../ export --format requirements-txt --frozen --no-hashes)
uv "${uvoptions[@]}"                         -o ./envs/dev.env.txt
uv "${uvoptions[@]}" --no-dev                -o ./envs/runtime.env.txt
uv "${uvoptions[@]}" --no-dev --group doc    -o ./envs/docs.env.txt
uv "${uvoptions[@]}" --no-dev --group format -o ./envs/format.env.txt
uv "${uvoptions[@]}" --no-dev --group typing -o ./envs/mypy.env.txt
uv "${uvoptions[@]}" --no-dev --group lint   -o ./envs/pylint.env.txt
uv "${uvoptions[@]}" --no-dev --group test   -o ./envs/pytest.env.txt
