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


def _make_uniform_superposition():
    from qualtran.bloqs.chemistry.thc import UniformSuperpositionTHC

    num_mu = 10
    bitsize_mu = (num_mu - 1).bit_length()
    bitsize_rot = 8
    return UniformSuperpositionTHC(bitsize=bitsize_mu, bitsize_rot=bitsize_rot)


def _make_prepare():
    from qualtran.bloqs.chemistry.thc import PrepareTHC

    return PrepareTHC(num_mu=10, num_spin_orb=4, keep_bitsize=8)

def test_notebook():
    execute_notebook('thc')
