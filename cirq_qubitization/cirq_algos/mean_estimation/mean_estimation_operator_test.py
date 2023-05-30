from functools import cached_property
from typing import Sequence, Tuple

import cirq
import numpy as np
import pytest
from attrs import frozen

import cirq_qubitization as cq
import cirq_qubitization.cirq_infra.testing as cq_testing
from cirq_qubitization.cirq_algos.mean_estimation import (
    CodeForRandomVariable,
    MeanEstimationOperator,
)
from cirq_qubitization.cirq_algos.select_and_prepare import PrepareOracle, SelectOracle


def compute_unitary(op: cirq.Operation):
    """Computes the reduced unitary, when the decomposition of op can allocate new ancillas."""
    qubit_order = cirq.QubitOrder.explicit(op.qubits, fallback=cirq.QubitOrder.DEFAULT)
    circuit = cirq.Circuit(
        cirq.decompose(op, keep=lambda val: cirq.has_unitary(val, allow_decompose=False))
    )
    nq, nall = len(op.qubits), len(circuit.all_qubits())
    assert nall <= 16, "Too many qubits to compute the reduced unitary for."
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


@frozen
class BernoulliSynthesizer(PrepareOracle):
    r"""Synthesizes the state $sqrt(1 - p)|00..00> + sqrt(p)|11..11>$"""
    p: float
    num_qubits: int

    @cached_property
    def selection_registers(self) -> cq.SelectionRegisters:
        return cq.SelectionRegisters.build(q=(self.num_qubits, 2))

    def decompose_from_registers(self, q: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        theta = np.arccos(np.sqrt(1 - self.p))
        yield cirq.ry(2 * theta).on(q[0])
        yield [cirq.CNOT(q[0], q[i]) for i in range(1, len(q))]


@frozen
class BernoulliEncoder(SelectOracle):
    r"""Encodes Bernoulli random variable y0/y1 as $Enc|ii..i>|0> = |ii..i>|y_{i}>$ where i=0/1."""
    p: float
    y: Tuple[int, int]
    selection_bitsize: int
    target_bitsize: int

    @cached_property
    def control_registers(self) -> cq.Registers:
        return cq.Registers([])

    @cached_property
    def selection_registers(self) -> cq.SelectionRegisters:
        return cq.SelectionRegisters.build(q=(self.selection_bitsize, 2))

    @cached_property
    def target_registers(self) -> cq.Registers:
        return cq.Registers.build(t=self.target_bitsize)

    def decompose_from_registers(
        self, q: Sequence[cirq.Qid], t: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        y0_bin = cq.bit_tools.iter_bits(self.y[0], self.target_bitsize)
        y1_bin = cq.bit_tools.iter_bits(self.y[1], self.target_bitsize)

        for y0, y1, tq in zip(y0_bin, y1_bin, t):
            if y0:
                yield cirq.X(tq).controlled_by(*q, control_values=[0] * self.selection_bitsize)
            if y1:
                yield cirq.X(tq).controlled_by(*q, control_values=[1] * self.selection_bitsize)

    @cached_property
    def mu(self) -> float:
        return self.p * self.y[1] + (1 - self.p) * self.y[0]

    @cached_property
    def s_square(self) -> float:
        return self.p * (self.y[1] ** 2) + (1 - self.p) * (self.y[0] ** 2)


def satisfies_theorem_321(
    synthesizer: PrepareOracle,
    encoder: SelectOracle,
    c: float,
    s: float,
    mu: float,
    arctan_bitsize: int,
    debug_print: bool = False,
):
    """Verifies Theorem 3.21 of https://arxiv.org/abs/2208.07544

    Pr[∣sin(θ/2)∣ ∈ ∣µ∣ / √(1 + s ** 2) . [1 / (1 + cs), 1 / (1 - cs)]] >= (1 - 2 / c**2)
    """
    code = CodeForRandomVariable(synthesizer=synthesizer, encoder=encoder)
    mean_gate = MeanEstimationOperator(code, arctan_bitsize=arctan_bitsize)
    mean_gate_helper = cq_testing.GateHelper(mean_gate)

    # Compute a reduced unitary for mean_op.
    u = compute_unitary(mean_gate_helper.operation)
    assert cirq.is_unitary(u)

    # Compute the final state vector obtained using the synthesizer `Prep |0>`
    prep_op = synthesizer.on_registers(**synthesizer.registers.get_named_qubits())
    prep_state = cirq.Circuit(prep_op).final_state_vector()
    if debug_print:
        print(cirq.dirac_notation(prep_state))

    expected_hav = abs(mu) * np.sqrt(1 / (1 + s**2))
    expected_hav_low = expected_hav / (1 + c * s)
    expected_hav_high = expected_hav / (1 - c * s)

    if debug_print:
        print(f'{expected_hav_low=}, {expected_hav=}, {expected_hav_high=}')

    def is_good_eigvect(eig_val, eig_vect):
        theta = np.abs(np.angle(eig_val))
        hav_theta = np.sin(theta / 2)
        overlap_prob = abs(np.dot(prep_state, eig_vect))
        if expected_hav_low <= hav_theta <= expected_hav_high:
            if debug_print:
                print(f'{hav_theta=}, {overlap_prob=}, {1-2/c**2=}')
            return overlap_prob >= 1 - 2 / (c**2)
        return False

    eigvals, eigvects = cirq.linalg.unitary_eig(u)

    return any(is_good_eigvect(eig_val, eig_vect) for (eig_val, eig_vect) in zip(eigvals, eigvects))


@pytest.mark.parametrize('selection_bitsize', [1, 2])
@pytest.mark.parametrize(
    'p, y_1, target_bitsize, c',
    [
        (1 / 100 * 1 / 100, 3, 2, 100 / 7),
        (1 / 50 * 1 / 50, 2, 2, 50 / 4),
        (1 / 50 * 1 / 50, 1, 1, 50 / 10),
        (1 / 4 * 1 / 4, 1, 1, 1.5),
    ],
)
def test_mean_estimation_bernoulli(
    p: int, y_1: int, selection_bitsize: int, target_bitsize: int, c: float, arctan_bitsize: int = 5
):
    synthesizer = BernoulliSynthesizer(p, selection_bitsize)
    encoder = BernoulliEncoder(p, (0, y_1), selection_bitsize, target_bitsize)
    s = np.sqrt(encoder.s_square)
    # For hav_theta interval to be reasonably wide, 1/(1-cs) term should be <=2; thus cs <= 0.5.
    # The theorem assumes that C >= 1 and s <= 1 / c.
    assert c * s <= 0.5 and c >= 1 >= s

    assert satisfies_theorem_321(
        synthesizer=synthesizer,
        encoder=encoder,
        c=c,
        s=s,
        mu=encoder.mu,
        arctan_bitsize=arctan_bitsize,
    )


@frozen
class GroverSynthesizer(PrepareOracle):
    r"""Prepare a uniform superposition over the first $2^n$ elements."""

    n: int

    @cached_property
    def selection_registers(self) -> cq.SelectionRegisters:
        return cq.SelectionRegisters.build(selection=(self.n, 2**self.n))

    def decompose_from_registers(self, selection: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        yield cirq.H.on_each(*selection)

    def __pow__(self, power):
        if power in [+1, -1]:
            return self
        return NotImplemented


@frozen
class GroverEncoder(SelectOracle):
    """Enc|marked_item>|0> --> |marked_item>|marked_val>"""

    n: int
    marked_item: int
    marked_val: int

    @cached_property
    def control_registers(self) -> cq.Registers:
        return cq.Registers([])

    @cached_property
    def selection_registers(self) -> cq.SelectionRegisters:
        return cq.SelectionRegisters.build(selection=(self.n, 2**self.n))

    @cached_property
    def target_registers(self) -> cq.Registers:
        return cq.Registers.build(target=self.marked_val.bit_length())

    def decompose_from_registers(
        self, selection: Sequence[cirq.Qid], target: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        selection_cv = [*cq.bit_tools.iter_bits(self.marked_item, self.selection_registers.bitsize)]
        yval_bin = [*cq.bit_tools.iter_bits(self.marked_val, self.target_registers.bitsize)]

        for b, q in zip(yval_bin, target):
            if b:
                yield cirq.X(q).controlled_by(*selection, control_values=selection_cv)

    @cached_property
    def mu(self) -> float:
        return self.marked_val / 2**self.n

    @cached_property
    def s_square(self) -> float:
        return (self.marked_val**2) / 2**self.n


@pytest.mark.parametrize('n, marked_val, c', [(5, 1, 1.5), (4, 1, 1.5), (2, 1, np.sqrt(2))])
def test_mean_estimation_grover(
    n: int, marked_val: int, c: float, marked_item: int = 1, arctan_bitsize: int = 5
):
    synthesizer = GroverSynthesizer(n)
    encoder = GroverEncoder(n, marked_item=marked_item, marked_val=marked_val)
    s = np.sqrt(encoder.s_square)
    # TODO(#229): This test currently passes only for small `c` since the overlap prob for these
    # cases is ~15-30%. When we increase `c`, the expected overlap is much higher (c=1.5 implies an
    # expected overlap of 0.11 whereas c=2 implies 0.5), which doesn't seem to be the case here.
    assert c * s < 1 and c >= 1 >= s

    assert satisfies_theorem_321(
        synthesizer=synthesizer,
        encoder=encoder,
        c=c,
        s=s,
        mu=encoder.mu,
        arctan_bitsize=arctan_bitsize,
    )
