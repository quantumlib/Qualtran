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

import qualtran.bloqs.for_testing
from qualtran import bloq_example, BloqDocSpec
from qualtran.bloqs.for_testing.atom import TestAtom

from .jupyter_autogen_v2 import (
    _get_bloq_example_source_lines,
    _MarkdownCell,
    _PyCell,
    get_cells,
    NotebookSpecV2,
)


def _make_QROM():
    from qualtran.bloqs.data_loading.qrom import QROM

    return QROM([np.array([1, 2, 3, 4, 5])], selection_bitsizes=(3,), target_bitsizes=(3,))


def test_notebook_spec():
    nbspec = NotebookSpecV2(
        title='test',
        module=qualtran.bloqs.for_testing,
        bloq_specs=[BloqDocSpec(bloq_cls=TestAtom, examples=[])],
    )
    assert nbspec.title == 'test'
    assert inspect.ismodule(nbspec.module)
    assert len(nbspec.bloq_specs) == 1


@bloq_example
def _my_bloq_example() -> TestAtom:
    # Comment
    x = 'y' + str(2)
    my_bloq_example = TestAtom(tag=x)
    return my_bloq_example


def test_get_bloq_example_source_lines():
    lines = _get_bloq_example_source_lines(_my_bloq_example)
    source = '\n'.join(lines)
    assert (
        source
        == """\
# Comment
x = 'y' + str(2)
my_bloq_example = TestAtom(tag=x)"""
    )


def test_get_cells():

    bds = BloqDocSpec(bloq_cls=TestAtom, examples=[_my_bloq_example])
    cells = get_cells(bds)
    assert isinstance(cells[0], _MarkdownCell)
    assert cells[0].text == (
        '## `TestAtom`\n'
        'An atomic bloq useful for generic testing and demonstration.\n'
        '\n'
        '#### Parameters\n'
        ' - `tag`: An optional string for differentiating `TestAtom`s. \n'
        '\n'
        '#### Registers\n'
        ' - `q`: One bit\n'
    )
    assert isinstance(cells[1], _PyCell)
    assert cells[1].text == 'from qualtran.bloqs.for_testing import TestAtom'
