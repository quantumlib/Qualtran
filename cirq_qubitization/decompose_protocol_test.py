import cirq
from cirq_qubitization.decompose_protocol import decompose_once_into_operations


def test_fredkin():
    c, t1, t2 = cirq.LineQid.for_gate(cirq.FREDKIN)
    want = cirq.Circuit(
        cirq.CNOT(t2, t1),
        cirq.CNOT(c, t1),
        cirq.H(t2),
        cirq.T(c),
        cirq.T(t1) ** -1,
        cirq.T(t2),
        cirq.CNOT(t2, t1),
        cirq.CNOT(c, t2),
        cirq.T(t1),
        cirq.CNOT(c, t1),
        cirq.T(t2) ** -1,
        cirq.T(t1) ** -1,
        cirq.CNOT(c, t2),
        cirq.CNOT(t2, t1),
        cirq.T(t1),
        cirq.H(t2),
        cirq.CNOT(t2, t1),
    )
    assert want == cirq.Circuit(decompose_once_into_operations(cirq.FREDKIN))


def test_classcial_controls():
    c, t1, t2 = cirq.LineQid.for_gate(cirq.FREDKIN)
    op = cirq.FREDKIN(c, t1, t2).with_classical_controls('key')
    classical_controls = op.classical_controls
    want = (
        cirq.CNOT(t2, t1).with_classical_controls(classical_controls),
        cirq.CNOT(c, t1).with_classical_controls(classical_controls),
        cirq.H(t2).with_classical_controls(classical_controls),
        cirq.T(c).with_classical_controls(classical_controls),
        (cirq.T(t1) ** -1).with_classical_controls(classical_controls),
        cirq.T(t2).with_classical_controls(classical_controls),
        cirq.CNOT(t2, t1).with_classical_controls(classical_controls),
        cirq.CNOT(c, t2).with_classical_controls(classical_controls),
        cirq.T(t1).with_classical_controls(classical_controls),
        cirq.CNOT(c, t1).with_classical_controls(classical_controls),
        (cirq.T(t2) ** -1).with_classical_controls(classical_controls),
        (cirq.T(t1) ** -1).with_classical_controls(classical_controls),
        cirq.CNOT(c, t2).with_classical_controls(classical_controls),
        cirq.CNOT(t2, t1).with_classical_controls(classical_controls),
        cirq.T(t1).with_classical_controls(classical_controls),
        cirq.H(t2).with_classical_controls(classical_controls),
        cirq.CNOT(t2, t1).with_classical_controls(classical_controls),
    )
    assert want == decompose_once_into_operations(op)


def test_simple_circuit():
    """A circuit of T, CNOT, and single qubit cliffords shouldn't be simplified"""
    q0, q1 = cirq.LineQubit(2).range(2)
    simple_circuit = cirq.Circuit(
        cirq.S(q0),
        cirq.X(q1),
        cirq.Y(q0),
        cirq.Z(q0),
        cirq.T(q1) ** -1,
        cirq.H(q1),
        cirq.T(q1),
        cirq.CNOT(q0, q1),
    )
    assert simple_circuit == cirq.Circuit(decompose_once_into_operations(simple_circuit))
