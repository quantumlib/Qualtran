#!/bin/bash
set -e

# Regenerate the kickmix python API reference
python dev_tools/kickmix/gen_api_reference.py > docs/kickmix/api_reference.md

# Auto-format all C++ source and header files.
find kickmix/src | grep "\.\(cc\|h\)$" | xargs clang-format-21 -i

# Check C++ code builds.
bazel build :all

# Check C++ unit tests.
bazel test :kickmix_test

# Run kickmix python unit tests.
PYTHONPATH=kickmix pytest kickmix/src

# Run kickmix doc tests.
dev_tools/kickmix/doctest_proper.py --module kickmix
