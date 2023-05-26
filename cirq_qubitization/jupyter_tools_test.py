import cirq
import IPython.display
import ipywidgets

import cirq_qubitization as cq
import cirq_qubitization.cirq_infra.testing as cq_testing
from cirq_qubitization import And
from cirq_qubitization.jupyter_tools import display_gate_and_compilation, svg_circuit


def test_svg_circuit():
    g = cq_testing.GateHelper(And(cv=(1, 1, 1)))
    svg = svg_circuit(g.circuit, g.r)
    svg_str: str = svg.data

    # check that the order is respected in the svg data.
    assert svg_str.find('control') < svg_str.find('ancilla') < svg_str.find('target')


def test_display_gate_and_compilation(monkeypatch):
    call_args = []

    def _dummy_display(stuff):
        call_args.append(stuff)

    monkeypatch.setattr(IPython.display, "display", _dummy_display)
    g = cq_testing.GateHelper(And(cv=(1, 1, 1)))
    display_gate_and_compilation(g)

    (display_arg,) = call_args
    assert isinstance(display_arg, ipywidgets.HBox)
    assert len(display_arg.children) == 2


def test_circuit_with_costs():
    g = cq_testing.GateHelper(And(cv=(1, 1, 1)))
    circuit = cq.jupyter_tools.circuit_with_costs(g.circuit)
    expected_circuit = cirq.Circuit(g.operation.with_tags('t:8,r:0'))
    cirq.testing.assert_same_circuits(circuit, expected_circuit)
