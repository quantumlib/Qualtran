import cirq
import pytest
import numpy as np

from cirq_qubitization.cirq_infra.decompose_protocol import (
    _fredkin,
    _try_decompose_from_known_decompositions,
)


def test_fredkin_unitary():
    c, t1, t2 = cirq.LineQid.for_gate(cirq.FREDKIN)
    context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())
    np.testing.assert_allclose(
        cirq.Circuit(_fredkin((c, t1, t2), context)).unitary(),
        cirq.unitary(cirq.FREDKIN(c, t1, t2)),
        atol=1e-8,
    )


@pytest.mark.parametrize('gate', [cirq.FREDKIN, cirq.FREDKIN**-1])
def test_decompose_fredkin(gate):
    c, t1, t2 = cirq.LineQid.for_gate(cirq.FREDKIN)
    op = cirq.FREDKIN(c, t1, t2)
    context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())
    want = tuple(cirq.flatten_op_tree(_fredkin((c, t1, t2), context)))
    assert want == _try_decompose_from_known_decompositions(op, context)

    op = cirq.FREDKIN(c, t1, t2).with_classical_controls('key')
    classical_controls = op.classical_controls
    want = tuple(
        o.with_classical_controls(*classical_controls)
        for o in cirq.flatten_op_tree(_fredkin((c, t1, t2), context))
    )
    assert want == _try_decompose_from_known_decompositions(op, context)
