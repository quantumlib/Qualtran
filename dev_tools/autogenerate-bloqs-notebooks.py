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

"""Autogeneration of stub Jupyter notebooks.

For each module listed in the `NOTEBOOK_SPECS` global variable (in this file)
we write a notebook with a title, module docstring,
standard imports, and information on each bloq listed in the
`gate_specs` field. For each gate, we render a docstring and diagrams.

## Adding a new gate.

 1. Create a new function that takes no arguments
    and returns an instance of the desired gate.
 2. If this is a new module: add a new key/value pair to the NOTEBOOK_SPECS global variable
    in this file. The key should be the name of the module with a `NotebookSpec` value. See
    the docstring for `NotebookSpec` for more information.
 3. Update the `NotebookSpec` `gate_specs` field to include a `BloqNbSpec` for your new gate.
    Provide your factory function from step (1).

## Autogen behavior.

Each autogenerated notebook cell is tagged, so we know it was autogenerated. Each time
this script is re-run, these cells will be re-rendered. *Modifications to generated _cells_
will not be persisted*.

If you add additional cells to the notebook it will *preserve them* even when this script is
re-run

Usage as a script:
    python dev_tools/autogenerate-bloqs-notebooks.py
"""

from typing import List

from qualtran_dev_tools.git_tools import get_git_root
from qualtran_dev_tools.jupyter_autogen import BloqNbSpec, NotebookSpec, render_notebook

import qualtran.bloqs.arithmetic
import qualtran.bloqs.basic_gates.cnot_test
import qualtran.bloqs.basic_gates.hadamard_test
import qualtran.bloqs.basic_gates.rotation_test
import qualtran.bloqs.basic_gates.x_basis_test
import qualtran.bloqs.basic_gates.z_basis_test

SOURCE_DIR = get_git_root() / 'qualtran/'

NOTEBOOK_SPECS: List[NotebookSpec] = [
    NotebookSpec(
        title='Basic Gates',
        module=qualtran.bloqs.basic_gates,
        gate_specs=[
            BloqNbSpec(qualtran.bloqs.basic_gates.cnot_test._make_CNOT),
            BloqNbSpec(qualtran.bloqs.basic_gates.x_basis_test._make_plus_state),
            BloqNbSpec(qualtran.bloqs.basic_gates.z_basis_test._make_zero_state),
            BloqNbSpec(qualtran.bloqs.basic_gates.rotation_test._make_Rz),
            BloqNbSpec(qualtran.bloqs.basic_gates.hadamard_test._make_Hadamard),
        ],
        directory=f'{SOURCE_DIR}/bloqs/basic_gates',
    )
]


def render_notebooks():
    for nbspec in NOTEBOOK_SPECS:
        render_notebook(nbspec)


if __name__ == '__main__':
    render_notebooks()
