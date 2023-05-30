import cirq
import numpy as np

from cirq_qubitization.bloq_algos.basic_gates import (
    OneEffect,
    OneState,
    TwoBitCSwap,
    TwoBitSwap,
    ZeroEffect,
    ZeroState,
)
from cirq_qubitization.bloq_algos.basic_gates.swap import (
    _controlled_swap_matrix,
    _swap_matrix,
    CSwap,
)
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder


def _make_CSwap():
    from cirq_qubitization.bloq_algos.basic_gates import CSwap

    return CSwap(bitsize=64)


def test_swap_matrix():
    m = _swap_matrix().reshape(4, 4)
    np.testing.assert_array_equal(m, cirq.unitary(cirq.SWAP))


def test_cswap_matrix():
    m = _controlled_swap_matrix().reshape(8, 8)
    np.testing.assert_array_equal(m, cirq.unitary(cirq.CSWAP))


def test_two_bit_swap():
    swap = TwoBitSwap()
    np.testing.assert_array_equal(swap.tensor_contract(), cirq.unitary(cirq.SWAP))

    x, y = swap.call_classically(x=0, y=1)
    assert x == 1
    assert y == 0


def _set_ctrl_two_bit_swap(ctrl_bit):
    states = [ZeroState(), OneState()]
    effs = [ZeroEffect(), OneEffect()]

    bb = CompositeBloqBuilder()
    (q0,) = bb.add(states[ctrl_bit])
    q1 = bb.add_register('q1', 1)
    q2 = bb.add_register('q2', 1)
    q0, q1, q2 = bb.add(TwoBitCSwap(), ctrl=q0, x=q1, y=q2)
    bb.add(effs[ctrl_bit], q=q0)
    return bb.finalize(q1=q1, q2=q2)


def test_two_bit_cswap():
    cswap = TwoBitCSwap()
    np.testing.assert_array_equal(cswap.tensor_contract(), cirq.unitary(cirq.CSWAP))

    # Zero ctrl -- it's identity
    np.testing.assert_array_equal(np.eye(4), _set_ctrl_two_bit_swap(0).tensor_contract())
    # One ctrl -- it's swap
    np.testing.assert_array_equal(
        _swap_matrix().reshape(4, 4), _set_ctrl_two_bit_swap(1).tensor_contract()
    )

    # classical logic
    ctrl, x, y = cswap.call_classically(ctrl=0, x=1, y=0)
    assert (ctrl, x, y) == (0, 1, 0)
    ctrl, x, y = cswap.call_classically(ctrl=1, x=1, y=0)
    assert (ctrl, x, y) == (1, 0, 1)

    # cirq
    c1 = cirq.Circuit([cirq.CSWAP(*cirq.LineQubit.range(3))]).freeze()
    c2, _ = cswap.as_composite_bloq().to_cirq_circuit(
        ctrl=[cirq.LineQubit(0)], x=[cirq.LineQubit(1)], y=[cirq.LineQubit(2)]
    )
    assert c1 == c2


def _set_ctrl_swap(ctrl_bit, bloq: CSwap):
    states = [ZeroState(), OneState()]
    effs = [ZeroEffect(), OneEffect()]

    bb = CompositeBloqBuilder()
    (q0,) = bb.add(states[ctrl_bit])
    q1 = bb.add_register('q1', bloq.bitsize)
    q2 = bb.add_register('q2', bloq.bitsize)
    q0, q1, q2 = bb.add(bloq, ctrl=q0, x=q1, y=q2)
    bb.add(effs[ctrl_bit], q=q0)
    return bb.finalize(q1=q1, q2=q2)


def test_cswap_unitary():
    cswap = CSwap(bitsize=4)

    # Zero ctrl -- it's identity
    np.testing.assert_array_equal(np.eye(2 ** (4 * 2)), _set_ctrl_swap(0, cswap).tensor_contract())

    # One ctrl -- it's multi-swap
    qubits = cirq.LineQubit.range(8)
    q_x, q_y = qubits[:4], qubits[4:]
    unitary = cirq.unitary(cirq.Circuit(cirq.SWAP(x, y) for x, y in zip(q_x, q_y)))
    np.testing.assert_array_equal(unitary, _set_ctrl_swap(1, cswap).tensor_contract())


def test_cswap_classical():
    cswap = CSwap(bitsize=8)
    cswap_d = cswap.decompose_bloq()

    ctrl, x, y = cswap.call_classically(ctrl=0, x=255, y=128)
    assert (ctrl, x, y) == (0, 255, 128)
    ctrl, x, y = cswap_d.call_classically(ctrl=0, x=255, y=128)
    assert (ctrl, x, y) == (0, 255, 128)

    ctrl, x, y = cswap.call_classically(ctrl=1, x=255, y=128)
    assert (ctrl, x, y) == (1, 128, 255)
    ctrl, x, y = cswap_d.call_classically(ctrl=1, x=255, y=128)
    assert (ctrl, x, y) == (1, 128, 255)
