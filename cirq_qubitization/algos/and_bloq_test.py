import itertools

import numpy as np

from cirq_qubitization.algos.and_bloq import And
from cirq_qubitization.algos.basic_gates import OneState, ZeroEffect, ZeroState
from cirq_qubitization.algos.basic_gates.zero_state import OneEffect
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegisters
from cirq_qubitization.quantum_graph.quimb_sim import cbloq_to_dense


def _make_and():
    from cirq_qubitization.algos.and_bloq import And

    return And()


def test_truth_table():
    k = [ZeroState(), OneState()]
    d = [ZeroEffect(), OneEffect()]

    for a, b in itertools.product([0, 1], repeat=2):

        bb = CompositeBloqBuilder()
        (q_a,) = bb.add(k[a])
        (q_b,) = bb.add(k[b])
        (q_a, q_b), res = bb.add(And(), ctrl=[q_a, q_b])
        bb.add(d[a], q=q_a)
        bb.add(d[b], q=q_b)
        cbloq = bb.finalize(res=res)

        vec = cbloq_to_dense(cbloq)
        if a and b:
            np.testing.assert_allclose([0, 1], vec)
        else:
            np.testing.assert_allclose([1, 0], vec)


def test_inverse():
    bb, (q0, q1) = CompositeBloqBuilder.make_with_soqs(FancyRegisters.build(q0=1, q1=1))
    qs, trg = bb.add(And(), ctrl=[q0, q1])
    (qs,) = bb.add(And(adjoint=True), ctrl=qs, target=trg)
    cbloq = bb.finalize(q0=qs[0], q1=qs[1])

    mat = cbloq_to_dense(cbloq)
    np.testing.assert_allclose(np.eye(4), mat)
