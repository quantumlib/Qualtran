# Developer Documentation

## Install Development Dependencies

```
sudo apt install bazel-7.2.0 pybind11-dev

pip install pytest
```

## Run C++ Unit Tests

```bash
bazel test :all
```

## Run C++ Command Line Tool

```bash
bazel run :kickmix
```

## Build Python Wheel

```bash
bazel build :kickmix_dev_wheel
# output is at bazel-bin/kickmix-0.0.dev0-py3-none-any.whl
```

### ... and install the wheel

```
pip install bazel-bin/kickmix-0.0.dev0-py3-none-any.whl
```

### ... and run python unit tests

```bash
PYTHONPATH=kickmix pytest kickmix/src
```

## Auto-format code (using clang-format-21)

The project includes a `.clang-format` file that clang format will use automatically.
Clang format has some behavior differences  between version; the code is currently formatted using `clang-format-21`
which can be installed on debian systems uses `sudo apt-get install clang-format-21`. 

The following command can be used to format all C++ source files in the project:

```bash
find kickmix/src | grep "\.\(cc\|h\)$" | xargs clang-format-21 -i
```

## Run Python Performance Tests

```bash
python kickmix/src/kickmix/py/perf/run_all_perf.py
```

## Run C++ Performance Tests

```bash
bazel run :kickmix_perf

# example usage:
#     bazel run :kickmix_perf -- --target_seconds 0.1 --only "sim_sample_*"
```

## Regenerate python api reference

```bash
# note: requires the python wheel to be installed
python dev_tools/kickmix/gen_api_reference.py > docs/kickmix/api_reference.md
```
