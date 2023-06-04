"""Tests for jupyter autogen.

This module is used as a mock NotebookSpec module, so this docstring shows up in the tests.
"""
import inspect

import numpy as np

import cirq_qubitization
from cirq_qubitization.jupyter_autogen import (
    _get_code_for_demoing_a_gate,
    _get_lines_for_constructing_an_object,
    GateNbSpec,
    get_markdown_docstring_lines,
    NotebookSpec,
    render_notebook_cells,
)


def _make_QROM():
    from cirq_qubitization import QROM

    return QROM([np.array([1, 2, 3, 4, 5])], selection_bitsizes=[3], target_bitsizes=[3])


def test_gate_nb_spec():
    gspec = GateNbSpec(factory=_make_QROM)
    assert gspec.cqid == '_make_QROM'
    assert gspec.gate_cls == cirq_qubitization.QROM


def test_notebook_spec():
    nbspec = NotebookSpec(
        title='test',
        module=cirq_qubitization.jupyter_autogen_test,
        gate_specs=[GateNbSpec(_make_QROM)],
    )
    assert nbspec.title == 'test'
    assert inspect.ismodule(nbspec.module)
    assert len(nbspec.gate_specs) == 1


class ClassWithDocstrings:
    """This class has some nifty docstrings.

    Parameters:
        x: The variable x
        y: The variable y used by `my_function`.

    References:
        [Google](www.google.com). Brin et. al. 1999.
    """


def test_get_markdown_docstring_lines():
    lines = get_markdown_docstring_lines(ClassWithDocstrings)
    assert lines == [
        '## `ClassWithDocstrings`',
        'This class has some nifty docstrings.',
        '',
        '#### Parameters',
        ' - `x`: The variable x',
        ' - `y`: The variable y used by `my_function`. ',
        '',
        '#### References',
        '[Google](www.google.com). Brin et. al. 1999.',
        '',
    ]


def test_get_lines_for_constructing_an_object():
    lines, obj_expr = _get_lines_for_constructing_an_object(_make_QROM)
    assert lines == ['from cirq_qubitization import QROM', '']
    assert (
        obj_expr == 'QROM([np.array([1, 2, 3, 4, 5])], selection_bitsizes=[3], target_bitsizes=[3])'
    )


def test_get_code_for_demoing_a_gate():
    code = _get_code_for_demoing_a_gate(_make_QROM, vertical=False)
    assert code.endswith('display_gate_and_compilation(g)')


def test_render_notebook_cells():
    cells = render_notebook_cells(
        NotebookSpec(
            title='Test Notebook',
            module=cirq_qubitization.jupyter_autogen_test,
            gate_specs=[GateNbSpec(_make_QROM)],
        )
    )

    assert cells.title_cell.metadata == {'cq.autogen': 'title_cell'}
    assert cells.title_cell.source == '\n'.join(
        [
            '# Test Notebook',
            '',
            'Tests for jupyter autogen.',
            '',
            'This module is used as a mock NotebookSpec module, so this docstring shows up in the tests.',
        ]
    )
    assert cells.top_imports.metadata == {'cq.autogen': 'top_imports'}

    assert list(cells.gate_cells.keys()) == ['_make_QROM']
    gcell = cells.gate_cells['_make_QROM']
    assert gcell.md.source.startswith('## `QROM`')
    assert gcell.py.source.startswith('from cirq_qubitization import QROM')
