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
import sympy

import qualtran.testing as qlt_testing
from qualtran import BloqBuilder, Soquet
from qualtran.bloqs.basic_gates import Hadamard, IntEffect, IntState
from qualtran.bloqs.block_encoding.sparse_matrix import TopLeftRowColumnOracle
from qualtran.bloqs.block_encoding.sparse_matrix_hermitian import (
    _sparse_matrix_hermitian_block_encoding,
    _sparse_matrix_symb_hermitian_block_encoding,
    SparseMatrixHermitian,
    UniformSqrtEntryOracle,
)
from qualtran.resource_counting.generalizers import ignore_split_join


def test_sparse_matrix(bloq_autotester):
    bloq_autotester(_sparse_matrix_hermitian_block_encoding)


def test_sparse_matrix_symb(bloq_autotester):
    bloq_autotester(_sparse_matrix_symb_hermitian_block_encoding)


def test_sparse_matrix_params():
    bloq = _sparse_matrix_hermitian_block_encoding()
    assert bloq.system_bitsize == 2
    assert bloq.alpha == 4
    assert bloq.epsilon == 0
    assert bloq.ancilla_bitsize == 2 + 2
    assert bloq.resource_bitsize == 0

    bloq = _sparse_matrix_symb_hermitian_block_encoding()
    n = sympy.Symbol('n', positive=True, integer=True)
    assert bloq.system_bitsize == n
    assert bloq.alpha == 2**n
    assert bloq.epsilon == 0
    assert bloq.ancilla_bitsize == n + 2
    assert bloq.resource_bitsize == 0


def test_call_graph():
    bloq = _sparse_matrix_hermitian_block_encoding()
    _, sigma = bloq.call_graph(generalizer=ignore_split_join)
    assert sigma[Hadamard()] == 4

    bloq = _sparse_matrix_symb_hermitian_block_encoding()
    _, sigma = bloq.call_graph(generalizer=ignore_split_join)
    n = sympy.Symbol('n', integer=True, positive=True)
    assert sigma[Hadamard()] == 6 * n


def test_sparse_matrix_tensors():
    bloq = _sparse_matrix_hermitian_block_encoding()
    alpha = bloq.alpha
    bb = BloqBuilder()
    system = bb.add_register("system", 2)
    ancilla = cast(Soquet, bb.add(IntState(0, 4)))
    system, ancilla = bb.add_t(bloq, system=system, ancilla=ancilla)
    bb.add(IntEffect(0, 4), val=ancilla)
    bloq = bb.finalize(system=system)

    from_gate = np.full((4, 4), 0.3)
    from_tensors = bloq.tensor_contract() * alpha
    np.testing.assert_allclose(from_gate, from_tensors)


topleft_matrix = [
    [0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]


def test_top_left_matrix():
    col_oracle = TopLeftRowColumnOracle(system_bitsize=3, num_nonzero=3)
    entry_oracle = UniformSqrtEntryOracle(system_bitsize=3, entry=0.3)
    bloq = SparseMatrixHermitian(col_oracle, entry_oracle, eps=0)
    alpha = bloq.alpha

    bb = BloqBuilder()
    system = bb.add_register("system", 3)
    ancilla = cast(Soquet, bb.add(IntState(0, 3 + 2)))
    system, ancilla = bb.add_t(bloq, system=system, ancilla=ancilla)
    bb.add(IntEffect(0, 3 + 2), val=ancilla)
    bloq = bb.finalize(system=system)

    from_tensors = bloq.tensor_contract() * alpha
    np.testing.assert_allclose(topleft_matrix, from_tensors, atol=0.003)


def test_counts():
    qlt_testing.assert_equivalent_bloq_counts(
        _sparse_matrix_hermitian_block_encoding(), generalizer=ignore_split_join
    )
    qlt_testing.assert_equivalent_bloq_counts(
        _sparse_matrix_hermitian_block_encoding().controlled(), generalizer=ignore_split_join
    )
