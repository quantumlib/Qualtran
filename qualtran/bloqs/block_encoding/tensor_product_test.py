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
from attrs import evolve

from qualtran import QAny, QBit, Register, Signature
from qualtran.bloqs.basic_gates import CNOT, Hadamard, TGate
from qualtran.bloqs.block_encoding.tensor_product import (
    _tensor_product_block_encoding,
    TensorProduct,
)
from qualtran.bloqs.block_encoding.unitary import Unitary


def test_tensor_product(bloq_autotester):
    bloq_autotester(_tensor_product_block_encoding)


def test_tensor_product_signature():
    assert _tensor_product_block_encoding().signature == Signature(
        [Register("system", QAny(2)), Register("ancilla", QAny(0)), Register("resource", QAny(0))]
    )


def test_tensor_product_params():
    u1 = evolve(
        Unitary(TGate(), dtype=QBit()), alpha=0.5, num_ancillas=2, num_resource=1, epsilon=0.01
    )
    u2 = evolve(
        Unitary(CNOT(), dtype=QAny(2)), alpha=0.5, num_ancillas=1, num_resource=1, epsilon=0.1
    )
    bloq = TensorProduct([u1, u2])
    assert bloq.dtype == QAny(1 + 2)
    assert bloq.alpha == 0.5 * 0.5
    assert bloq.epsilon == 0.5 * 0.01 + 0.5 * 0.1
    assert bloq.num_ancillas == 2 + 1
    assert bloq.num_resource == 1 + 1


def test_tensor_product_tensors():
    from_gate = np.kron(TGate().tensor_contract(), Hadamard().tensor_contract())
    from_tensors = _tensor_product_block_encoding().tensor_contract()
    np.testing.assert_allclose(from_gate, from_tensors)
