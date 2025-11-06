#  Copyright 2025 Google LLC
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

r"""A submodule for compiling $\mathbb{Z}[e^{i \pi/4}]$ matrices to Clifford+T as well as generating them."""

from qualtran.rotation_synthesis.matrix.generation import (
    generate_rotations,
    generate_rotations_iter,
)
from qualtran.rotation_synthesis.matrix.su2_ct import generate_cliffords, SU2CliffordT
