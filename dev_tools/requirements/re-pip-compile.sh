#!/bin/bash

#  Copyright 2023 Google LLC
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

set -ex

# First, compile a complete & consistent environment with all dependencies
mkdir -p ./envs/
pip-compile --output-file=envs/dev.env.txt --resolver=backtracking deps/dev-tools.txt deps/runtime.txt

# Compile the various sub-environments.
# For each of these, we include `constrain.txt` as an input requirements file. This file
# has only one line: "-c envs/dev.env.txt", which will use the above rendered dev.env.txt as a
# "constraints file", which is exactly what we want.
pip-compile --output-file=envs/runtime.env.txt   --resolver=backtracking --constraint=envs/dev.env.txt deps/runtime.txt
pip-compile --output-file=envs/format.env.txt    --resolver=backtracking --constraint=envs/dev.env.txt deps/runtime.txt deps/format.txt
pip-compile --output-file=envs/pylint.env.txt    --resolver=backtracking --constraint=envs/dev.env.txt deps/runtime.txt deps/pylint.txt
pip-compile --output-file=envs/pytest.env.txt    --resolver=backtracking --constraint=envs/dev.env.txt deps/runtime.txt deps/pytest.txt
pip-compile --output-file=envs/docs.env.txt      --resolver=backtracking --constraint=envs/dev.env.txt deps/runtime.txt deps/docs.txt
pip-compile --output-file=envs/pip-tools.env.txt --resolver=backtracking --constraint=envs/dev.env.txt deps/pip-tools.txt
