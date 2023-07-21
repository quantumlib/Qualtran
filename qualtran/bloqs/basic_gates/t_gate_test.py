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

from qualtran import BloqBuilder
from qualtran.bloqs.basic_gates import PlusState, TGate
from qualtran.resource_counting import get_bloq_counts_graph


def _make_t_gate():
    from qualtran.bloqs.basic_gates import TGate

    return TGate()


def test_bloq_counts():
    g, simga = get_bloq_counts_graph(TGate())
    assert simga == {TGate(): 1}


def test_to_cirq():
    bb = BloqBuilder()
    q = bb.add(PlusState())
    q = bb.add(TGate(), q=q)
    cbloq = bb.finalize(q=q)
    circuit, _ = cbloq.to_cirq_circuit()
    cirq.testing.assert_has_diagram(circuit, "_c(0): ───H───T───")
