#  Copyright 2024 Google LLC
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
import numpy as np
import pytest
import quimb.tensor as qtn

from qualtran import Bloq, ConnectionT, QAny, QUInt, Signature
from qualtran.bloqs.arithmetic import Add, Xor
from qualtran.bloqs.basic_gates import Toffoli, TwoBitCSwap, XGate
from qualtran.simulation.classical_sim import ClassicalValT
from qualtran.simulation.tensor._tensor_from_classical import (
    bloq_to_dense_via_classical_action,
    my_tensors_from_classical_action,
)


@pytest.mark.parametrize(
    "bloq", [XGate(), TwoBitCSwap(), Toffoli(), Add(QUInt(3)), Xor(QAny(3))], ids=str
)
def test_tensor_consistent_with_classical(bloq: Bloq):
    from_classical = bloq_to_dense_via_classical_action(bloq)
    from_tensor = bloq.tensor_contract()

    np.testing.assert_allclose(from_classical, from_tensor)


class TestClassicalBloq(Bloq):
    @property
    def signature(self) -> 'Signature':
        return Signature.build(a=1, b=1, c=1)

    def on_classical_vals(
        self, a: 'ClassicalValT', b: 'ClassicalValT', c: 'ClassicalValT'
    ) -> dict[str, 'ClassicalValT']:
        if a == 1 and b == 1:
            c = c ^ 1
        return {'a': a, 'b': b, 'c': c}

    def my_tensors(
        self, incoming: dict[str, 'ConnectionT'], outgoing: dict[str, 'ConnectionT']
    ) -> list['qtn.Tensor']:
        return my_tensors_from_classical_action(self, incoming, outgoing)


def test_my_tensors_from_classical_action():
    bloq = TestClassicalBloq()

    expected_tensor = Toffoli().tensor_contract()
    actual_tensor = bloq.tensor_contract()
    np.testing.assert_allclose(actual_tensor, expected_tensor)
