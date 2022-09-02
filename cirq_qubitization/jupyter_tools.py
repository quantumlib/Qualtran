import cirq
import cirq.contrib.svg.svg as ccsvg
import ipywidgets
from IPython.display import display, SVG

import cirq_qubitization.testing as cq_testing
from cirq_qubitization import GateWithRegisters, Registers


def display_gate_and_compilation(g: cq_testing.GateSystem, vertical=False):
    """Use ipywidgets to display SVG circuits for a `GateSystem` next to each other.

    Args:
        g: The `GateSystem` to draw
        vertical: If true, lay-out the original gate and its decomposition vertically
            rather than side-by-side.
    """
    out1 = ipywidgets.Output()
    out2 = ipywidgets.Output()
    if vertical:
        box = ipywidgets.VBox([out1, out2])
    else:
        box = ipywidgets.HBox([out1, out2])

    out1.append_display_data(svg_circuit(g.circuit, registers=g.r))
    out2.append_display_data(
        svg_circuit(cirq.Circuit(cirq.decompose_once(g.operation)), registers=g.r)
    )

    display(box)


def svg_circuit(circuit: 'cirq.AbstractCircuit', registers: Registers = None):
    """Return an SVG object representing a circuit.

    Args:
        circuit: The circuit to draw.
        registers: Optional `Registers` object to order the qubits.
    """
    if len(circuit) == 0:
        raise ValueError("Circuit is empty.")

    if registers is not None:
        qubit_order = registers.merge_qubits(**registers.get_named_qubits())
    else:
        qubit_order = cirq.QubitOrder.DEFAULT
    tdd = circuit.to_text_diagram_drawer(transpose=False, qubit_order=qubit_order)
    if len(tdd.horizontal_lines) == 0:
        raise ValueError("No non-empty moments.")
    return SVG(ccsvg.tdd_to_svg(tdd))
