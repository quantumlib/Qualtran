from pathlib import Path
from typing import Optional

import cirq
import cirq.contrib.svg.svg as ccsvg
import IPython.display
import ipywidgets
import jinja2
import nbconvert
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

import cirq_qubitization.cirq_infra.testing as cq_testing
from cirq_qubitization.cirq_infra.gate_with_registers import Registers
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.graphviz import PrettyGraphDrawer


def display_gate_and_compilation(g: cq_testing.GateHelper, vertical=False):
    """Use ipywidgets to display SVG circuits for a `GateHelper` next to each other.

    Args:
        g: The `GateHelper` to draw
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

    IPython.display.display(box)


def svg_circuit(circuit: 'cirq.AbstractCircuit', registers: Registers = None):
    """Return an SVG object representing a circuit.

    Args:
        circuit: The circuit to draw.
        registers: Optional `Registers` object to order the qubits.
    """
    if len(circuit) == 0:
        raise ValueError("Circuit is empty.")

    if registers is not None:
        qubit_order = cirq.QubitOrder.explicit(
            registers.merge_qubits(**registers.get_named_qubits()), fallback=cirq.QubitOrder.DEFAULT
        )
    else:
        qubit_order = cirq.QubitOrder.DEFAULT
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


def export_notebook(nbpath: Path, htmlpath: Path) -> Optional[Exception]:
    with nbpath.open() as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    try:
        nb, resources = ep.preprocess(nb)
    except Exception as e:
        print(f'{nbpath} failed!')
        print(e)
        return e
    nb['metadata']['title'] = 'heyo'
    template_file = Path(__file__).parent / '../dev_tools/index.html.j2'
    exporter = nbconvert.HTMLExporter(
        # extra_template_basedirs=[Path(__file__).parent /'../dev_tools'],
        extra_loaders=[jinja2.FileSystemLoader(Path(__file__).parent / '../dev_tools')],
        # template_name='nbconvert_style',
        template_file ='crazy.html.j2',
    )

    html, resources = nbconvert.export(exporter, nb, resources=resources)
    with htmlpath.open('w') as f:
        f.write(html)
