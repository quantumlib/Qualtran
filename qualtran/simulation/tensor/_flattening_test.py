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
from qualtran.bloqs.basic_gates import CNOT
from qualtran.bloqs.mcmt.and_bloq import MultiAnd
from qualtran.simulation.tensor import bloq_has_custom_tensors


def test_bloq_has_custom_tensors():
    assert bloq_has_custom_tensors(CNOT())
    assert not bloq_has_custom_tensors(MultiAnd((1,) * 5))
