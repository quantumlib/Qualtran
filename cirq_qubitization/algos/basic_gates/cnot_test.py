import cirq
import numpy as np

from cirq_qubitization.algos.basic_gates import CNOT, PlusState, ZeroState
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegisters
from cirq_qubitization.quantum_graph.quimb_sim import bloq_to_dense, cbloq_to_dense


def _make_CNOT():
    from cirq_qubitization.algos.basic_gates import CNOT

    return CNOT()


def test_cnot():
    bloq = CNOT()
    matrix = bloq_to_dense(bloq)
    # fmt: off
    should_be = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]])
    # fmt: on
    np.testing.assert_allclose(should_be, matrix)


def test_cnot_cbloq():
    bb = CompositeBloqBuilder(FancyRegisters.build(c=1, t=1))
    soqs = bb.initial_soquets()
    c, t = bb.add(CNOT(), ctrl=soqs['c'], target=soqs['t'])
    cbloq = bb.finalize(c=c, t=t)
    matrix = cbloq_to_dense(cbloq)

    c_qs = cirq.LineQubit.range(2)
    c_circ = cirq.Circuit(cirq.CNOT(c_qs[0], c_qs[1]))
    c_matrix = c_circ.unitary(qubit_order=c_qs)

    np.testing.assert_allclose(c_matrix, matrix)


def test_bell_state():
    bb = CompositeBloqBuilder()

    (q0,) = bb.add(PlusState())
    (q1,) = bb.add(ZeroState())

    q0, q1 = bb.add(CNOT(), ctrl=q0, target=q1)

    cbloq = bb.finalize(q0=q0, q1=q1)
    matrix = cbloq_to_dense(cbloq)

    should_be = np.array([1, 0, 0, 1]) / np.sqrt(2)
    np.testing.assert_allclose(should_be, matrix)
