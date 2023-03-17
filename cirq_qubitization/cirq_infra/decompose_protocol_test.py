import cirq
import numpy as np

from cirq_qubitization.cirq_infra.decompose_protocol import _fredkin, decompose_once_into_operations


def test_fredkin_unitary():
    c, t1, t2 = cirq.LineQid.for_gate(cirq.FREDKIN)
    np.testing.assert_allclose(
        cirq.Circuit(_fredkin((c, t1, t2))).unitary(),
        cirq.unitary(cirq.FREDKIN(c, t1, t2)),
        atol=1e-8,
    )


def test_decompose_fredkin():
    c, t1, t2 = cirq.LineQid.for_gate(cirq.FREDKIN)
    op = cirq.FREDKIN(c, t1, t2)
    want = tuple(cirq.flatten_op_tree(_fredkin((c, t1, t2))))
    assert want == decompose_once_into_operations(op)

    op = cirq.FREDKIN(c, t1, t2).with_classical_controls('key')
    classical_controls = op.classical_controls
    want = tuple(
        o.with_classical_controls(classical_controls)
        for o in cirq.flatten_op_tree(_fredkin((c, t1, t2)))
    )
    assert want == decompose_once_into_operations(op)
