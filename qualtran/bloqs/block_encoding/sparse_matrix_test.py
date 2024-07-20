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

from typing import cast

import numpy as np

from qualtran import BloqBuilder, QAny, Register, Signature, Soquet
from qualtran.bloqs.basic_gates import IntEffect, IntState
from qualtran.bloqs.block_encoding.sparse_matrix import _sparse_matrix_block_encoding


def test_sparse_matrix(bloq_autotester):
    bloq_autotester(_sparse_matrix_block_encoding)


def test_sparse_matrix_signature():
    bloq = _sparse_matrix_block_encoding()
    assert bloq.signature == Signature(
        [Register(name="system", dtype=QAny(2)), Register(name="ancilla", dtype=QAny(3))]
    )


def test_sparse_matrix_params():
    bloq = _sparse_matrix_block_encoding()
    assert bloq.system_bitsize == 2
    assert bloq.alpha == 4
    assert bloq.epsilon == 0
    assert bloq.ancilla_bitsize == 2 + 1
    assert bloq.resource_bitsize == 0


def test_sparse_matrix_tensors():
    bloq = _sparse_matrix_block_encoding()
    alpha = bloq.alpha
    bb = BloqBuilder()
    system = bb.add_register("system", 2)
    ancilla = cast(Soquet, bb.add(IntState(0, 3)))
    system, ancilla = bb.add_t(bloq, system=system, ancilla=ancilla)
    bb.add(IntEffect(0, 3), val=ancilla)
    bloq = bb.finalize(system=system)

    from_gate = np.full((4, 4), 0.3)
    from_tensors = bloq.tensor_contract() * alpha
    np.testing.assert_allclose(from_gate, from_tensors)
