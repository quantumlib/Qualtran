import cirq
from cirq_qubitization.qubit_manager import qalloc, qfree, qtot, _reset


def test_qubit_manager():
    _reset() # reset manager so that tests don't depend on each other.
    # allocate 2 qubits with names starting with `ancilla`
    qs = qalloc(2)
    assert qs == [cirq.NamedQubit("ancilla0"), cirq.NamedQubit("ancilla1")] 
    qfree(qs.pop(0)) # free a qubit 
    # allocate 1 qubits with names starting with `x` and reuses the already freed qubit
    assert qalloc(2, "x") == [cirq.NamedQubit("ancilla0"), cirq.NamedQubit("x0")] 
    assert qtot() == 3
    qfree(qs.pop(0)) # free a qubit
    # allocate 2 qubits with no reuse
    assert qalloc(2, prefix='noreuse', reuse=False) == [cirq.NamedQubit("noreuse0"), cirq.NamedQubit("noreuse1")]
    assert qtot() == 5