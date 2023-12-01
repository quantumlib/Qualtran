#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Tests for jupyter autogen.

This module is used as a mock NotebookSpec module, so this docstring shows up in the tests.
"""
import inspect

import numpy as np

from qualtran.bloqs.qrom import QROM

from .jupyter_autogen import (
    _get_code_for_demoing_a_gate,
    _get_lines_for_constructing_an_object,
    GateNbSpec,
    get_markdown_docstring_lines,
    NotebookSpec,
    render_notebook_cells,
)


def _make_QROM():
    from qualtran.bloqs.qrom import QROM

    return QROM([np.array([1, 2, 3, 4, 5])], selection_bitsizes=(3,), target_bitsizes=(3,))


def test_gate_nb_spec():
    gspec = GateNbSpec(factory=_make_QROM)
    assert gspec.cqid == '_make_QROM'
    assert gspec.gate_cls == QROM


def test_notebook_spec():
    import qualtran_dev_tools.jupyter_autogen_test  # pylint: disable=import-self

    nbspec = NotebookSpec(
        title='test',
        module=qualtran_dev_tools.jupyter_autogen_test,
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
    assert lines == ['from qualtran.bloqs.qrom import QROM', '']
    assert (
        obj_expr
        == 'QROM([np.array([1, 2, 3, 4, 5])], selection_bitsizes=(3,), target_bitsizes=(3,))'
    )


def test_get_code_for_demoing_a_gate():
    code = _get_code_for_demoing_a_gate(_make_QROM, vertical=False)
    assert code.endswith('display_gate_and_compilation(g)')


def test_render_notebook_cells():
    import qualtran_dev_tools.jupyter_autogen_test  # pylint: disable=import-self

    cells = render_notebook_cells(
        NotebookSpec(
            title='Test Notebook',
            module=qualtran_dev_tools.jupyter_autogen_test,
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
    assert gcell.py.source.startswith('from qualtran.bloqs.qrom import QROM')
