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
import cirq
import numpy as np

from qualtran import BloqBuilder
from qualtran.bloqs.basic_gates import Hadamard, OneState
from qualtran.bloqs.basic_gates.hadamard import _hadamard


def _make_Hadamard():
    from qualtran.bloqs.basic_gates import Hadamard

    return Hadamard()


def test_to_cirq():
    bb = BloqBuilder()
    q = bb.add(OneState())
    q = bb.add(Hadamard(), q=q)
    cbloq = bb.finalize(q=q)
    circuit, _ = cbloq.to_cirq_circuit()
    cirq.testing.assert_has_diagram(circuit, "_c(0): ───X───H───")
    vec1 = cbloq.tensor_contract()
    vec2 = cirq.final_state_vector(circuit)
    np.testing.assert_allclose(vec1, vec2)


def test_hadamard(bloq_autotester):
    bloq_autotester(_hadamard)
