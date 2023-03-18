#!/bin/bash

pip-compile --output-file=format.env.txt --resolver=backtracking deps/format.txt deps/runtime.txt
pip-compile --output-file=pylint.env.txt --resolver=backtracking deps/pylint.txt deps/runtime.txt
pip-compile --output-file=pytest.env.txt --resolver=backtracking deps/pytest.txt deps/runtime.txt
pip-compile --output-file=dev.env.txt --resolver=backtracking deps/dev-tools.txt deps/runtime.txt
