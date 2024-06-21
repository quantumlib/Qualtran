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

import numpy as np
import pytest

from qualtran import QAny, QBit, Register, Signature
from qualtran.bloqs.basic_gates import IntState, TGate
from qualtran.bloqs.block_encoding.unitary import _unitary_block_encoding, Unitary


def test_unitary(bloq_autotester):
    bloq_autotester(_unitary_block_encoding)


def test_unitary_signature():
    assert _unitary_block_encoding().signature == Signature(
        [Register("system", QBit()), Register("ancilla", QAny(0)), Register("resource", QAny(0))]
    )
    with pytest.raises(AssertionError):
        _ = Unitary(TGate(), dtype=QAny(2))
    with pytest.raises(AssertionError):
        _ = Unitary(IntState(55, bitsize=8), dtype=QAny(8))


def test_unitary_params():
    bloq = _unitary_block_encoding()
    assert bloq.alpha == 1
    assert bloq.epsilon == 0
    assert bloq.num_ancillas == 0
    assert bloq.num_resource == 0


def test_unitary_tensors():
    from_gate = TGate().tensor_contract()
    from_tensors = _unitary_block_encoding().tensor_contract()
    np.testing.assert_allclose(from_gate, from_tensors)
