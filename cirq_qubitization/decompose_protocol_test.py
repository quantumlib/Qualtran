import cirq
from cirq_qubitization.decompose_protocol import decompose, decompose_once



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
        cirq.T(t2)**-1,
        cirq.T(t1)**-1,
        cirq.CNOT(c, t2),
        cirq.CNOT(t2, t1),
        cirq.T(t1),
        cirq.H(t2),
        cirq.CNOT(t2, t1),
    )
    assert want == cirq.Circuit(decompose_once(cirq.FREDKIN))
    # decompose mustn't recurse on T, CNOT or single qubit cliffords.
    assert want == cirq.Circuit(decompose(cirq.FREDKIN))

def test_simple_circuit():
    """A circuit of T, CNOT, and single qubit cliffords shouldn't be simplified"""
    q0, q1 = cirq.LineQubit(2).range(2)
    simple_circuit = cirq.Circuit(
        cirq.S(q0),
        cirq.X(q1),
        cirq.Y(q0),
        cirq.Z(q0),
        cirq.T(q1)**-1,
        cirq.H(q1),
        cirq.T(q1),
        cirq.CNOT(q0, q1),
    )
    assert simple_circuit == cirq.Circuit(decompose(simple_circuit))
    assert simple_circuit == cirq.Circuit(decompose_once(simple_circuit))