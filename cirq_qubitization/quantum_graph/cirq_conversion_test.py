import cirq
import numpy as np

from cirq_qubitization.jupyter_tools import execute_notebook
from cirq_qubitization.quantum_graph.cirq_conversion import cirq_circuit_to_cbloq, CirqGateAsBloq
from cirq_qubitization.quantum_graph.fancy_registers import Side


def test_cirq_gate():
    x = CirqGateAsBloq(cirq.X)
    rx = CirqGateAsBloq(cirq.Rx(rads=0.123 * np.pi))
    toffoli = CirqGateAsBloq(cirq.TOFFOLI)

    for b in [x, rx, toffoli]:
        assert len(b.registers) == 1
        assert b.registers[0].side == Side.THRU

    assert x.registers[0].wireshape == (1,)
    assert toffoli.registers[0].wireshape == (3,)

    assert str(x) == 'CirqGateAsBloq(gate=cirq.X)'
    assert x.pretty_name() == 'cirq.X'
    assert x.short_name() == 'cirq.X'

    assert rx.pretty_name() == 'cirq.Rx(0.123π)'
    assert rx.short_name() == 'cirq.Rx'

    assert toffoli.pretty_name() == 'cirq.TOFFOLI'
    assert toffoli.short_name() == 'cirq.TOFFOLI'


def test_cirq_circuit_to_cbloq():
    qubits = cirq.LineQubit.range(6)
    circuit = cirq.testing.random_circuit(qubits, n_moments=7, op_density=1.0, random_state=52)
    cbloq = cirq_circuit_to_cbloq(circuit)

    bloq_unitary = cbloq.tensor_contract()
    cirq_unitary = circuit.unitary(qubits)
    np.testing.assert_allclose(cirq_unitary, bloq_unitary, atol=1e-8)


def test_cbloq_to_cirq_circuit():
    qubits = cirq.LineQubit.range(6)
    circuit = cirq.testing.random_circuit(qubits, n_moments=7, op_density=1.0, random_state=52)
    cbloq = cirq_circuit_to_cbloq(circuit)

    # important! we lose moment structure
    circuit = cirq.Circuit(circuit.all_operations())

    # Note: a 1d `wireshape` bloq register is actually two-dimensional in cirq-world
    # because of the implicit `bitsize` dimension (which must be explicit in cirq-world).
    # CirqGate has registers of bitsize=1 and wireshape=(n,); hence the list transpose below.
    circuit2, _ = cbloq.to_cirq_circuit({'qubits': [[q] for q in qubits]})

    assert circuit == circuit2


def test_notebook():
    execute_notebook('cirq_conversion')
