from pathlib import Path
from typing import Tuple

import cirq
import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

import cirq_qubitization


@pytest.mark.parametrize("cv", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_and_gate(cv: Tuple[int, int]):
    c1, c2, t = cirq.LineQubit.range(3)
    input_states = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]
    output_states = [inp[:2] + (1 if inp[:2] == cv else 0,) for inp in input_states]

    circuit = cirq.Circuit(cirq_qubitization.And(cv).on(c1, c2, t))
    for input, output in zip(input_states, output_states):
        result = cirq.Simulator().simulate(circuit, initial_state=input)
        assert result.dirac_notation()[1:-1] == "".join(str(x) for x in output)


@pytest.mark.parametrize("cv", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_and_gate_adjoint(cv: Tuple[int, int]):
    c1, c2, t = cirq.LineQubit.range(3)
    all_cvs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    input_states = [inp + (1 if inp == cv else 0,) for inp in all_cvs]
    output_states = [inp + (0,) for inp in all_cvs]

    circuit = cirq.Circuit(cirq_qubitization.And(cv, adjoint=True).on(c1, c2, t))
    for input, output in zip(input_states, output_states):
        result = cirq.Simulator().simulate(circuit, initial_state=input)
        assert result.dirac_notation()[1:-1] == "".join(str(x) for x in output)


def test_notebook():
    notebook_path = Path(__file__).parent / "and_gate.ipynb"
    with notebook_path.open() as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb)
