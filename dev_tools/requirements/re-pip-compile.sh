#!/bin/bash

set -ex

# First, compile a complete & consistent environment with all dependencies
pip-compile --upgrade --output-file=dev.env.txt --resolver=backtracking deps/dev-tools.txt deps/runtime.txt

# Compile the various sub-environments.
# For each of these, we include `constrain.txt` as an input requirements file. This file
# has only one line: "-c ../dev.env.txt", which will use the above rendered dev.env.txt as a
# "constraints file", which is exactly what we want.
pip-compile --output-file=runtime.env.txt --resolver=backtracking constrain.txt deps/runtime.txt
pip-compile --output-file=format.env.txt  --resolver=backtracking constrain.txt deps/runtime.txt deps/format.txt
pip-compile --output-file=pylint.env.txt  --resolver=backtracking constrain.txt deps/runtime.txt deps/pylint.txt
pip-compile --output-file=pytest.env.txt  --resolver=backtracking constrain.txt deps/runtime.txt deps/pytest.txt
pip-compile --output-file=docs.env.txt    --resolver=backtracking constrain.txt deps/runtime.txt deps/docs.txt

