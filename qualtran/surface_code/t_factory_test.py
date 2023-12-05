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


from qualtran.surface_code import t_factory
from qualtran.surface_code.algorithm_summary import AlgorithmSummary


def test_15to1factory():
    factory = t_factory.Simple15to1TFactory(
        num_qubits=16000, cycle_time_us=83.2, error_rate=2.1e-15
    )
    magic_count = AlgorithmSummary(t_gates=1, toffoli_gates=1)
    assert factory.footprint() == 16000
    assert factory.n_cycles(magic_count) == 5
    assert factory.spacetime_footprint() == 16000 * 83.2
    assert factory.distillation_error(magic_count, 1e-3) == 5 * 2.1e-15
