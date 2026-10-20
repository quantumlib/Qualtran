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

__version__ = '0.1.dev0'

if platform.system().startswith('Win'):
    common_compile_args = [
        '/std:c++20',
        '/O2',
        f'/DVERSION_INFO={__version__}',
    ]
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
    ext_modules=[kickmix_extension],
    python_requires='>=3.10.0',
    install_requires=['numpy'],
    # Needed on Windows/macOS to avoid the default `build` colliding with Bazel's `BUILD`.
    options={'build': {'build_base': 'python_build_kickmix'}},
)
