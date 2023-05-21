from functools import cached_property
from typing import Optional, Sequence

import cirq
import numpy as np
import pytest
from attrs import frozen

import cirq_qubitization as cq
from cirq_qubitization.cirq_algos.mean_estimation import (
    CodeForRandomVariable,
    MeanEstimationOperator,
)
from cirq_qubitization.cirq_algos.select_and_prepare import PrepareOracle, SelectOracle


@frozen
class GroverPrepare(PrepareOracle):
    """Prepare a uniform superposition over the first $N$ elements."""

    N: int

    @cached_property
    def selection_registers(self) -> cq.SelectionRegisters:
        return cq.SelectionRegisters.build(selection=((self.N - 1).bit_length(), self.N))

    @cached_property
    def junk_registers(self) -> cq.Registers:
        return cq.Registers([])

    def decompose_from_registers(self, selection: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        yield cq.PrepareUniformSuperposition(self.N).on_registers(controls=[], target=selection)


@frozen
class GroverSelect(SelectOracle):
    """Y|x>|0> = |x>|sqrt(N)> if `x` is marked else |x>|0>."""

    N: int
    marked_item: Optional[int] = None

    @cached_property
    def control_registers(self) -> cq.Registers:
        return cq.Registers([])

    @cached_property
    def selection_registers(self) -> cq.SelectionRegisters:
        return cq.SelectionRegisters.build(selection=((self.N - 1).bit_length(), self.N))

    @cached_property
    def sqrtN(self) -> int:
        return int(np.floor(np.sqrt(self.N)))

    @cached_property
    def target_registers(self) -> cq.Registers:
        return cq.Registers.build(target=self.sqrtN.bit_length())

    def decompose_from_registers(self, selection, target) -> cirq.OP_TREE:
        if self.marked_item is None:
            return cirq.I.on_each(*target)
        selection_cv = [*cq.bit_tools.iter_bits(self.marked_item, self.selection_registers.bitsize)]
        sqrt_bin = [*cq.bit_tools.iter_bits(self.sqrtN, self.target_registers.bitsize)]
        for b, q in zip(sqrt_bin, target):
            if b:
                yield cirq.X(q).controlled_by(*selection, control_values=selection_cv)


def compute_unitary(op: cirq.Operation):
    """Computes the reduced unitary, when the decomposition of op can allocate new ancillas."""
    qubit_order = cirq.QubitOrder.explicit(op.qubits, fallback=cirq.QubitOrder.DEFAULT)
    circuit = cirq.Circuit(cirq.decompose(op))
    nq, nall = len(op.qubits), len(circuit.all_qubits())
    assert nall <= 13, "Too many qubits to compute the reduced unitary for."
    u = circuit.unitary(qubit_order=qubit_order).reshape((2,) * 2 * nall)
    new_order = [*range(nq, nall), *range(nall + nq, 2 * nall), *range(nq), *range(nall, nall + nq)]
    u = np.transpose(u, axes=new_order)
    u = u[(0, 0) * (nall - nq)]
    return u.reshape(2**nq, 2**nq)


def _phase_kickback_via_cx(q: cirq.Qid, anc: cirq.Qid) -> cirq.OP_TREE:
    yield [cirq.X(anc), cirq.H(anc)]
    yield cirq.CX(q, anc)
    yield [cirq.H(anc), cirq.X(anc)]


def _phase_kickback_via_z(q: cirq.Qid, anc: cirq.Qid) -> cirq.OP_TREE:
    yield cirq.CX(q, anc)
    yield cirq.Z(anc)
    yield cirq.CX(q, anc)


@pytest.mark.parametrize(
    'decompose_func, expected', [(_phase_kickback_via_z, cirq.Z), (_phase_kickback_via_cx, cirq.Z)]
)
def test_compute_unitary(decompose_func, expected):
    class GateWithDecompose(cirq.Gate):
        def _num_qubits_(self):
            return 1

        def _decompose_(self, q):
            anc = cirq.NamedQubit("anc")
            yield from decompose_func(*q, anc)

    op = GateWithDecompose().on(*cirq.LineQubit.range(1))
    u = compute_unitary(op)
    assert cirq.is_unitary(u)
    assert np.allclose(u, cirq.unitary(expected))


@pytest.mark.parametrize('N, arctan_bitsize, marked_item', [(4, 2, 1), (4, 2, 2)])
def test_mean_estimation_walk(N: int, arctan_bitsize: int, marked_item: int):
    # TODO: Make the test work for other combinations of (N, arctan_bitsize)
    code = CodeForRandomVariable(synthesizer=GroverPrepare(N), encoder=GroverSelect(N, marked_item))
    mean_gate = MeanEstimationOperator(code, arctan_bitsize=arctan_bitsize)
    mean_op = mean_gate.on_registers(**mean_gate.registers.get_named_qubits())
    prep_gate = mean_gate.reflect.prepare_gate
    prep_op = prep_gate.on_registers(**prep_gate.registers.get_named_qubits())

    # Compute a reduced unitary for mean_op.
    u = compute_unitary(mean_op)
    eigvals, eigvects = np.linalg.eigh(u)

    # Compute the final state vector obtained using the synthesizer `Prep |0>`
    u = compute_unitary(prep_op)
    prep_state = cirq.Circuit(
        cirq.MatrixGate(u).on(*prep_op.qubits), cirq.I.on_each(*mean_op.qubits)
    ).final_state_vector()

    def is_good_eigvect(eig_val, eig_vect):
        # 1. For grover setup, |u| = 1/sqrt(N) if there is a marked item. Assert that the walk
        # operator has eigenvalue ~= 2 * |u|.
        # 2. Assert that the state `Prep |0>` has high overlap with eigenvector corresponding to
        # eigenvalue 2 * |u|.
        # TODO: 0.5 overlap exists in this case. Verify what should be the generic number here.
        return np.isclose(abs(eig_val), 2 / np.sqrt(N)) and abs(np.dot(prep_state, eig_vect)) >= 0.5

    assert any(is_good_eigvect(eig_val, eig_vect) for (eig_val, eig_vect) in zip(eigvals, eigvects))
