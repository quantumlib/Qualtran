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
import itertools

import cirq
import numpy as np
import pytest

from qualtran import Bloq
from qualtran.bloqs.basic_gates import Rz, TGate
from qualtran.bloqs.basic_gates.f_gate import _f_gate, _fkn_matrix, FGate


def test_t_gate(bloq_autotester):
    bloq_autotester(_f_gate)


def test_call_graph():
    def all_t(bloq) -> Bloq:
        if isinstance(bloq, TGate):
            return TGate()
        return bloq

    assert (
        FGate(2, 3).call_graph(generalizer=all_t)[1]
        == FGate(2, 3).decompose_bloq().call_graph(generalizer=all_t)[1]
    )


@pytest.mark.parametrize('k,n', itertools.product(range(0, 4), range(1, 4)))
def test_tensors(k, n):
    # Eq. E11 in https://arxiv.org/pdf/2012.09238.pdf
    from_tensors = FGate(k, n).tensor_contract()
    np.testing.assert_allclose(from_tensors, _fkn_matrix(k, n))
    from_decomp = FGate(k, n).decompose_bloq().tensor_contract()
    cirq.testing.assert_allclose_up_to_global_phase(from_decomp, _fkn_matrix(k, n), atol=1e-12)
