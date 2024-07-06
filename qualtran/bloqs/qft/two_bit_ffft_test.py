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
import pytest

from qualtran import Bloq
from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.qft.two_bit_ffft import _fkn_matrix, _two_bit_ffft, TwoBitFFFT


def test_t_gate(bloq_autotester):
    bloq_autotester(_two_bit_ffft)


def test_call_graph():
    def all_t(bloq) -> Bloq:
        if isinstance(bloq, TGate):
            return TGate()
        return bloq

    assert (
        TwoBitFFFT(2, 3).call_graph(generalizer=all_t)[1]
        == TwoBitFFFT(2, 3).decompose_bloq().call_graph(generalizer=all_t)[1]
    )


@pytest.mark.parametrize('k,n', itertools.product(range(0, 4), range(1, 4)))
def test_tensors(k, n):
    # Eq. E11 in https://arxiv.org/pdf/2012.09238.pdf
    from_decomp = TwoBitFFFT(k, n).decompose_bloq().tensor_contract()
    cirq.testing.assert_allclose_up_to_global_phase(from_decomp, _fkn_matrix(k, n), atol=1e-12)
