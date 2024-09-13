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
import pytest
import sympy

import qualtran.testing as qlt_testing
from qualtran import BloqBuilder, QAny, Register, Signature, Soquet
from qualtran.bloqs.basic_gates import Hadamard, IntEffect, IntState
from qualtran.bloqs.block_encoding.sparse_matrix import (
    _explicit_matrix_block_encoding,
    _sparse_matrix_block_encoding,
    _sparse_matrix_symb_block_encoding,
    _symmetric_banded_matrix_block_encoding,
    ExplicitEntryOracle,
    SparseMatrix,
    SymmetricBandedRowColumnOracle,
    TopLeftRowColumnOracle,
    UniformEntryOracle,
)
from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.testing import execute_notebook


def test_sparse_matrix(bloq_autotester):
    bloq_autotester(_sparse_matrix_block_encoding)


def test_sparse_matrix_symb(bloq_autotester):
    bloq_autotester(_sparse_matrix_symb_block_encoding)


def test_explicit_matrix(bloq_autotester):
    bloq_autotester(_explicit_matrix_block_encoding)


def test_symmetric_banded_matrix(bloq_autotester):
    bloq_autotester(_symmetric_banded_matrix_block_encoding)


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

    bloq = _sparse_matrix_symb_block_encoding()
    n = sympy.Symbol('n', positive=True, integer=True)
    assert bloq.system_bitsize == n
    assert bloq.alpha == 2**n
    assert bloq.epsilon == 0
    assert bloq.ancilla_bitsize == n + 1
    assert bloq.resource_bitsize == 0


def test_call_graph():
    bloq = _sparse_matrix_block_encoding()
    _, sigma = bloq.call_graph(generalizer=ignore_split_join)
    assert sigma[Hadamard()] == 4

    bloq = _sparse_matrix_symb_block_encoding()
    _, sigma = bloq.call_graph(generalizer=ignore_split_join)
    n = sympy.Symbol('n', integer=True, positive=True)
    assert sigma[Hadamard()] == 6 * n


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


@pytest.mark.slow
def test_explicit_entry_oracle():
    rs = np.random.RandomState(1234)

    def gen_test():
        n = rs.randint(1, 3)
        N = 2**n
        data = rs.rand(N, N)
        return n, data

    tests = [(1, [[0.0, 0.25], [1 / 3, 0.467]])] + [gen_test() for _ in range(10)]
    for n, data in tests:
        row_oracle = TopLeftRowColumnOracle(n)
        col_oracle = TopLeftRowColumnOracle(n)
        entry_oracle = ExplicitEntryOracle(n, data=data, entry_bitsize=10)
        bloq = SparseMatrix(row_oracle, col_oracle, entry_oracle, eps=0)

        alpha = bloq.alpha
        bb = BloqBuilder()
        system = bb.add_register("system", n)
        ancilla = cast(Soquet, bb.add(IntState(0, n + 1)))
        system, ancilla = bb.add_t(bloq, system=system, ancilla=ancilla)
        bb.add(IntEffect(0, n + 1), val=ancilla)
        bloq = bb.finalize(system=system)

        from_tensors = bloq.tensor_contract() * alpha
        np.testing.assert_allclose(data, from_tensors, atol=0.003)


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
    row_oracle = TopLeftRowColumnOracle(system_bitsize=3, num_nonzero=3)
    col_oracle = TopLeftRowColumnOracle(system_bitsize=3, num_nonzero=3)
    entry_oracle = UniformEntryOracle(system_bitsize=3, entry=0.3)
    bloq = SparseMatrix(row_oracle, col_oracle, entry_oracle, eps=0)
    alpha = bloq.alpha

    bb = BloqBuilder()
    system = bb.add_register("system", 3)
    ancilla = cast(Soquet, bb.add(IntState(0, 3 + 1)))
    system, ancilla = bb.add_t(bloq, system=system, ancilla=ancilla)
    bb.add(IntEffect(0, 3 + 1), val=ancilla)
    bloq = bb.finalize(system=system)

    from_tensors = bloq.tensor_contract() * alpha
    np.testing.assert_allclose(topleft_matrix, from_tensors, atol=0.003)


test_matrix = [
    [0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
    [0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.3, 0.3, 0.3, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3],
    [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3],
]

test_matrix_nonzeros = [
    [7, 0, 1],
    [0, 1, 2],
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 0],
]


def test_symmetric_banded_row_column_oracle_classical():
    n = 3
    bloq = SymmetricBandedRowColumnOracle(n, bandsize=1)

    def test_entry(l, i):
        l_out, i_out = bloq.call_classically(l=l, i=i)
        assert i_out == i
        assert l_out == test_matrix_nonzeros[i][l]

    for i in range(2**n):
        for l in range(3):
            test_entry(l, i)


def test_symmetric_banded_row_column_oracle():
    n = 3
    bloq = SymmetricBandedRowColumnOracle(n, bandsize=1)

    def test_entry(l, i):
        bb = BloqBuilder()
        l_soq = cast(Soquet, bb.add(IntState(l, n)))
        i_soq = cast(Soquet, bb.add(IntState(i, n)))
        l_soq, i_soq = bb.add_t(bloq, l=l_soq, i=i_soq)
        bb.add(IntEffect(i, n), val=i_soq)
        out = bb.finalize(l=l_soq)
        np.testing.assert_allclose(
            IntState(test_matrix_nonzeros[i][l], n).tensor_contract(), out.tensor_contract()
        )

    for i in range(2**n):
        for l in range(3):
            test_entry(l, i)


def test_symmetric_banded_row_column_matrix():
    n = 3
    bloq = _symmetric_banded_matrix_block_encoding()
    alpha = bloq.alpha

    bb = BloqBuilder()
    system = bb.add_register("system", n)
    ancilla = cast(Soquet, bb.add(IntState(0, n + 1)))
    system, ancilla = bb.add_t(bloq, system=system, ancilla=ancilla)
    bb.add(IntEffect(0, n + 1), val=ancilla)
    bloq = bb.finalize(system=system)

    from_tensors = bloq.tensor_contract() * alpha
    np.testing.assert_allclose(test_matrix, from_tensors, atol=0.003)


def test_counts():
    qlt_testing.assert_equivalent_bloq_counts(
        _sparse_matrix_block_encoding(), generalizer=ignore_split_join
    )
    qlt_testing.assert_equivalent_bloq_counts(
        _explicit_matrix_block_encoding(), generalizer=ignore_split_join
    )


@pytest.mark.slow
def test_matrix_stress():
    rs = np.random.RandomState(1234)
    f = lambda: rs.randint(0, 10) / 10
    data = [
        [f(), f(), f(), 0.0, 0.0, 0.0, 0.0, 0.0],
        [f(), f(), f(), f(), 0.0, 0.0, 0.0, 0.0],
        [f(), f(), f(), f(), f(), 0.0, 0.0, 0.0],
        [0.0, f(), f(), f(), f(), f(), 0.0, 0.0],
        [0.0, 0.0, f(), f(), f(), f(), f(), 0.0],
        [0.0, 0.0, 0.0, f(), f(), f(), f(), f()],
        [0.0, 0.0, 0.0, 0.0, f(), f(), f(), f()],
        [0.0, 0.0, 0.0, 0.0, 0.0, f(), f(), f()],
    ]
    n = 3
    row_oracle = SymmetricBandedRowColumnOracle(n, bandsize=2)
    col_oracle = SymmetricBandedRowColumnOracle(n, bandsize=2)
    entry_oracle = ExplicitEntryOracle(system_bitsize=n, data=np.array(data), entry_bitsize=7)
    bloq = SparseMatrix(row_oracle, col_oracle, entry_oracle, eps=0)
    alpha = bloq.alpha

    bb = BloqBuilder()
    system = bb.add_register("system", n)
    ancilla = cast(Soquet, bb.add(IntState(0, n + 1)))
    system, ancilla = bb.add_t(bloq, system=system, ancilla=ancilla)
    bb.add(IntEffect(0, n + 1), val=ancilla)
    bloq = bb.finalize(system=system)

    from_tensors = bloq.tensor_contract() * alpha
    np.testing.assert_allclose(data, from_tensors, atol=0.03)


def gen_vlasov_hamiltonian(n, alpha, m):
    data = np.zeros((2**n, 2**n))
    data[0][1] = data[1][0] = np.sqrt((1 + alpha) / 2)
    for i in range(2, m + 1):
        data[i - 1][i] = data[i][i - 1] = np.sqrt(i / 2)
    data /= np.max(data)
    return data


@pytest.mark.slow
def test_vlasov_explicit():
    n = 3
    k = 2
    alpha = 2 / k**2
    data = gen_vlasov_hamiltonian(n, alpha, m=(2**n - 1))
    row_oracle = SymmetricBandedRowColumnOracle(n, bandsize=1)
    col_oracle = SymmetricBandedRowColumnOracle(n, bandsize=1)
    entry_oracle = ExplicitEntryOracle(system_bitsize=n, data=data, entry_bitsize=7)
    bloq = SparseMatrix(row_oracle, col_oracle, entry_oracle, eps=0)
    alpha = bloq.alpha

    bb = BloqBuilder()
    system = bb.add_register("system", n)
    ancilla = cast(Soquet, bb.add(IntState(0, n + 1)))
    system, ancilla = bb.add_t(bloq, system=system, ancilla=ancilla)
    bb.add(IntEffect(0, n + 1), val=ancilla)
    bloq = bb.finalize(system=system)

    from_tensors = bloq.tensor_contract() * alpha
    np.testing.assert_allclose(data, from_tensors, atol=0.02)


def test_symmetric_banded_counts():
    bloq = SymmetricBandedRowColumnOracle(3, bandsize=1)
    qlt_testing.assert_equivalent_bloq_counts(bloq)


def test_sparse_matrix_signal_state():
    assert isinstance(_sparse_matrix_block_encoding().signal_state.prepare, PrepareIdentity)
    _ = _sparse_matrix_block_encoding().signal_state.decompose_bloq()


@pytest.mark.notebook
def test_notebook():
    execute_notebook('sparse_matrix')
