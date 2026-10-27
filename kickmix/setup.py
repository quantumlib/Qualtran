# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import platform

import pybind11
from pybind11.setup_helpers import ParallelCompile
from setuptools import Extension, setup

ParallelCompile().install()

ALL_SOURCE_FILES = glob.glob("src/**/*.cc", recursive=True)
TEST_FILES = glob.glob("src/**/*.test.cc", recursive=True)
PERF_FILES = glob.glob("src/**/*.perf.cc", recursive=True)
MAIN_FILES = glob.glob("src/**/main.cc", recursive=True)
HEADER_FILES = glob.glob("src/**/*.h", recursive=True) + glob.glob("src/**/*.inl", recursive=True)
RELEVANT_SOURCE_FILES = sorted(set(ALL_SOURCE_FILES) - set(TEST_FILES + PERF_FILES + MAIN_FILES))

__version__ = '1.0.0'

if platform.system().startswith('Win'):
    common_compile_args = ['/std:c++20', '/O2', f'/DVERSION_INFO={__version__}']
else:
    common_compile_args = [
        '-std=c++20',
        '-fno-strict-aliasing',
        '-fvisibility=hidden',
        '-O3',
        '-g0',
        f'-DVERSION_INFO={__version__}',
    ]
    if any(arch in platform.machine().lower() for arch in ['x86', 'amd64', 'i386', 'i686']):
        common_compile_args.append('-mno-avx')

kickmix_extension = Extension(
    'kickmix',
    sources=RELEVANT_SOURCE_FILES,
    depends=HEADER_FILES,
    include_dirs=[pybind11.get_include(), 'src'],
    language='c++',
    extra_compile_args=common_compile_args,
)

setup(
    name='kickmix',
    version=__version__,
    author='Qualtran Developers',
    url='https://github.com/quantumlib/Qualtran',
    license='Apache-2.0',
    description='Fast construction and simulation of quantum circuits with classical operations, phasing, and X basis demolition measurement.',
    long_description="""
Kickmix
=======

Kickmix is a library for building and simulating quantum circuits that contain
classically reversible operations (like the CCX gate),
phasing operations (like the Z gate),
and X basis demolition measurement (the Hadamard+Measure+Reset gate sequence).
See [the kickmix circuit file format (v2)](https://github.com/quantumlib/Qualtran/blob/main/docs/kickmix/kickmix_format.md)
and [the kickmix instruction set](https://github.com/quantumlib/Qualtran/blob/main/docs/kickmix/kickmix_instructions.md)
for more information.

Also see the [kickmix api reference](https://github.com/quantumlib/Qualtran/blob/main/docs/kickmix/api_reference.md).

Example usage:

```python
import random
import tempfile

import kickmix as km

# Build a circuit.
builder = km.CircuitBuilder()
target = builder.create_quantum_register(10, "target")
offset = builder.create_quantum_register(10, "offset")
extra = builder.create_quantum_register(1, "extra")[0]
builder.ccx(target[0], offset[0], extra)
builder.iadd(target=target, offset=offset)
circuit = builder.finish_circuit()

# Simulate the circuit on 100 randomly chosen cases.
sim = km.Simulator(batch_size=100)
sim.use_same_registers_as(circuit)
sim.clear_for_shot()
offsets = [random.randrange(2**10) for _ in range(100)]
targets = [random.randrange(2**10) for _ in range(100)]
for k in range(100):
    sim.write_within_shot("offset", k, offsets[k])
    sim.write_within_shot("target", k, targets[k])
    sim.write_within_shot("extra", k, 0)
sim.do(circuit)

# Check that the registers ended with the correct values.
for k in range(100):
    assert sim.read_within_shot("offset", k, out=int) == offsets[k]
    assert sim.read_within_shot("target", k, out=int) == (offsets[k] + targets[k]) % 2**10
    assert sim.read_within_shot("extra", k, out=int) == offsets[k] & targets[k] & 1

# Check that there's no phase errors.
assert sim.read_phase_flipped_across_shots(out=int) == 0

# Output the circuit.
with tempfile.NamedTemporaryFile() as path:
    circuit.to_file(path, format="kmx")
print(circuit.text_diagram())
```

which prints:

```
 q0: -target[0]-@-----@----------------------------------------------------------------------@-----X-
                |     |                                                                      |     |
 q1: -target[1]-|-----|-X---@-----------------------------------------------@-----X-X--------|-----|-
                |     | |   |                                               |     | |        |     |
 q2: -target[2]-|-----|-|---|---X---@--------------------@-----X-X----------|-----|-|--------|-----|-
                |     | |   |   |   |                    |     | |          |     | |        |     |
 q3: -target[3]-|-----|-|---|---|---|---X-@-X-X----------|-----|-|----------|-----|-|--------|-----|-
                |     | |   |   |   |   | | | |          |     | |          |     | |        |     |
 q4: -target[4]-|-X---|-|---|---|---|---X-X-|-|----------|-----|-|----------|-----|-|--------|-----|-
                | |   | |   |   |   |   | | | |          |     | |          |     | |        |     |
 q5: -offset[0]-@-|---@-|---|---|---|---|-|-|-|----------|-----|-|----------|-----|-|--------Z**b0-@-
                | |   | |   |   |   |   | | | |          |     | |          |     | |
 q6: -offset[1]-|-|---|-X---@---|---|---|-|-|-|----------|-----|-|----------Z**b0-@-X----------------
                | |   | |   |   |   |   | | | |          |     | |                  |
 q7: -offset[2]-|-|---|-|---|---X---@---|-|-|-|----------Z**b0-@-X------------------|----------------
                | |   | |   |   |   |   | | | |                  |                  |
 q8: -offset[3]-|-|---|-|---|---|---|---X-@-@-X------------------|------------------|----------------
                | |   | |   |   |   |   |     |                  |                  |
 q9: -offset[4]-|-@---|-|---|---|---|---|-----|------------------|------------------|----------------
                |     | |   |   |   |   |     |                  |                  |
q10: -extra[0]--X-----|-|---|---|---|---|-----|------------------|------------------|----------------
                      | |   |   |   |   |     |                  |                  |
q11:              |0>-X-@---|-X-@---|-X-@-----@-X----------------@-X----------------@-HMR=b0
                            | |     | |         |                  |
q12:                    |0>-X-@-----|-|---------|------------------@-HMR=b0
                                    | |         |
q13:                            |0>-X-@---------@-HMR=b0
```
    """,
    long_description_content_type='text/markdown',
    ext_modules=[kickmix_extension],
    python_requires='>=3.10.0',
    install_requires=['numpy'],
    # Needed on Windows/macOS to avoid the default `build` colliding with Bazel's `BUILD`.
    options={'build': {'build_base': 'python_build_kickmix'}},
)
