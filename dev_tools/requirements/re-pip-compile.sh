#!/bin/bash

# First, compile the runtime requirements
pip-compile --output-file=runtime.env.txt --resolver=backtracking deps/runtime.txt

# Then, mix in extra requirements for tests
pip-compile --output-file=format.env.txt --resolver=backtracking deps/format.txt runtime.env.txt
pip-compile --output-file=pylint.env.txt --resolver=backtracking deps/pylint.txt runtime.env.txt
pip-compile --output-file=pytest.env.txt --resolver=backtracking deps/pytest.txt runtime.env.txt

# Everything is included in dev env
pip-compile --output-file=dev.env.txt --resolver=backtracking deps/dev-tools.txt runtime.env.txt
