from pathlib import Path

import cirq
import cirq.contrib.svg.svg as ccsvg
import IPython.display
import ipywidgets
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

import cirq_qubitization.cirq_infra.testing as cq_testing
from cirq_qubitization.cirq_infra.gate_with_registers import Registers
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.graphviz import PrettyGraphDrawer
from cirq_qubitization.t_complexity_protocol import t_complexity


def display_gate_and_compilation(g: cq_testing.GateHelper, vertical=False, include_costs=False):
    """Use ipywidgets to display SVG circuits for a `GateHelper` next to each other.

    Args:
        g: The `GateHelper` to draw
        vertical: If true, lay-out the original gate and its decomposition vertically
            rather than side-by-side.
        include_costs: If true, each operation is annotated with it's T-complexity cost.
    """
    out1 = ipywidgets.Output()
    out2 = ipywidgets.Output()
    if vertical:
        box = ipywidgets.VBox([out1, out2])
    else:
        box = ipywidgets.HBox([out1, out2])

    out1.append_display_data(svg_circuit(g.circuit, registers=g.r, include_costs=include_costs))
    out2.append_display_data(
        svg_circuit(
            cirq.Circuit(cirq.decompose_once(g.operation)),
            registers=g.r,
            include_costs=include_costs,
        )
    )

    IPython.display.display(box)


def circuit_with_costs(circuit: 'cirq.AbstractCircuit') -> 'cirq.AbstractCircuit':
    """Annotates each operation in the circuit with its T-complexity cost."""

    def _map_func(op: cirq.Operation, _):
        t_cost = t_complexity(op)
        return op.with_tags(f't:{t_cost.t:g},r:{t_cost.rotations:g}')

    return cirq.map_operations(circuit, map_func=_map_func)


def svg_circuit(
    circuit: 'cirq.AbstractCircuit', registers: Registers = None, include_costs: bool = False
):
    """Return an SVG object representing a circuit.

    Args:
        circuit: The circuit to draw.
        registers: Optional `Registers` object to order the qubits.
        include_costs: If true, each operation is annotated with it's T-complexity cost.
    """
    if len(circuit) == 0:
        raise ValueError("Circuit is empty.")

    if registers is not None:
        qubit_order = cirq.QubitOrder.explicit(
            registers.merge_qubits(**registers.get_named_qubits()), fallback=cirq.QubitOrder.DEFAULT
        )
    else:
        qubit_order = cirq.QubitOrder.DEFAULT

    if include_costs:
        circuit = circuit_with_costs(circuit)

    tdd = circuit.to_text_diagram_drawer(transpose=False, qubit_order=qubit_order)
    if len(tdd.horizontal_lines) == 0:
        raise ValueError("No non-empty moments.")
    return IPython.display.SVG(ccsvg.tdd_to_svg(tdd))


def show_bloq(bloq: Bloq):
    return PrettyGraphDrawer(bloq).get_svg()


def execute_notebook(name: str):
    """Execute a jupyter notebook in the caller's directory.

    Args:
        name: The name of the notebook without extension.

    """
    import traceback

    # Assumes that the notebook is in the same path from where the function was called,
    # which may be different from `__file__`.
    notebook_path = Path(traceback.extract_stack()[-2].filename).parent / f"{name}.ipynb"
    with notebook_path.open() as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb)


def format_dirac(s: str, n: int) -> str:
    """Reformats a dirac vector on as |work qubits|ancilla qubits|result qubit>"""
    return s[: n + 1] + '|' + s[n + 1 : -2] + '|' + s[-2:]


def check_oracle(n_qubits: int, oracle_func, oracle_name: str, oracle_desc: str):
    """Displays the result of running the oracle given by `oracle_func` on all 2^n inputs individually and in superposition."""
    A = cirq.NamedQubit.range(n_qubits, prefix='A')
    z = cirq.NamedQubit('z')
    c = cirq.Circuit(oracle_func(A, z))
    print(oracle_name)
    print(c)
    print()
    print('-' * 50)
    print()
    print(f'simulation result for "{oracle_name}" oracle')
    print('state vector form is |qubit registers | ancillas |z⟩')
    sim = cirq.Simulator()
    qubit_order = [q for q in A]
    qubit_order += [q for q in c.all_qubits() if q not in A + [z]]
    qubit_order += [z]
    for v in range(1 << n_qubits):
        bits = [(v >> i) & 1 for i in range(n_qubits - 1, -1, -1)]
        bits += (len(qubit_order) - n_qubits) * [0]
        result = sim.simulate(c, qubit_order=qubit_order, initial_state=bits)
        IPython.display.display(
            IPython.display.Latex(
                rf'z = $\mathbb{1}({v} {oracle_desc})$ = {result.dirac_notation()[-2]}'
            )
        )
        print('\tfinal state vector', format_dirac(result.dirac_notation(), n_qubits))

    c = cirq.Circuit([cirq.H(q) for q in A] + [c])
    result = sim.simulate(c, qubit_order=qubit_order)
    result = result.dirac_notation()
    final = []
    for s in result.split('|'):
        if '⟩' not in s:
            final.append(s)
            continue
        parts = s.split('⟩')
        parts[0] = format_dirac('|' + parts[0] + '⟩', n_qubits)
        final.append(''.join(parts))
    print('Acting on the uniform superposition of all states we get:')
    print('\t', ''.join(final))
