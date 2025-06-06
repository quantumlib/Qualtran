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

"""Autogeneration of Jupyter notebooks.

For each notebook spec listed in the various global variables imported from
`qualtran_dev_tools.notebook_specs` we write a notebook with a title, module
docstring, standard imports, and information on each bloq listed in the
`bloq_specs` field. For each bloq, we render a docstring and diagrams.

## Adding a new bloq.

Follow the instructions in `dev_tools/qualtran_dev_tools/notebook_specs.py` to
add a new bloq.

## Autogen behavior.

Each autogenerated notebook cell is tagged, so we know it was autogenerated. Each time
this script is re-run, these cells will be re-rendered. *Modifications to generated _cells_
will not be persisted*.

If you add additional cells to the notebook it will *preserve them* even when this script is
re-run.

This script will also generate a `docs/bloqs/index.rst` table of contents. The organization
and ordering of the entries corresponds to the ordering and organization of specs in this file.

If this script finds a BloqDocSpec in the qualtran library that isn't listed anywhere, it will emit
a warning.

Usage as a script:
    python dev_tools/autogenerate-bloqs-notebooks-v2.py
"""

from typing import Iterable, List

from qualtran_dev_tools.bloq_finder import get_bloqdocspecs
from qualtran_dev_tools.jupyter_autogen import NotebookSpecV2, render_notebook
from qualtran_dev_tools.notebook_specs import GIT_ROOT, NB_BY_SECTION, SOURCE_DIR

# --------------------------------------------------------------------------
# -----   Concepts   -------------------------------------------------------
# --------------------------------------------------------------------------
CONCEPTS = [
    # Note! These are just straight paths to existing notebooks. Used to generate
    # the table of contents.
    'multiplexers/unary_iteration.ipynb',
    'arithmetic/t_complexity_of_comparison_gates.ipynb',
    'arithmetic/error_analysis_for_fxp_arithmetic.ipynb',
    'phase_estimation/phase_estimation_of_quantum_walk.ipynb',
    'chemistry/trotter/grid_ham/trotter_costs.ipynb',
    'chemistry/trotter/hubbard/qpe_cost_optimization.ipynb',
    'chemistry/resource_estimation.ipynb',
    'chemistry/writing_algorithms.ipynb',
    'cryptography/rsa/factoring-via-modexp.ipynb',
    'state_preparation/state_preparation_via_rotation_tutorial.ipynb',
    'optimization/k_xor_sat/kikuchi_guiding_state_tutorial.ipynb',
]


# --------------------------------------------------------------------------
# -----   Root Bloqs   -----------------------------------------------------
# --------------------------------------------------------------------------
ROOT_BLOQS = ['cryptography/ecc/ecc.ipynb']


def _all_nbspecs() -> Iterable[NotebookSpecV2]:
    for _, nbspecs in NB_BY_SECTION:
        yield from nbspecs


def render_notebooks():
    for nbspec in _all_nbspecs():
        render_notebook(nbspec)


def _get_toc_section_lines(caption: str, entries: List[str], maxdepth: int = 2) -> List[str]:
    """Helper function to get the lines for a section of the table-of-contents."""
    return (
        ['.. toctree::', f'    :maxdepth: {maxdepth}', f'    :caption: {caption}:', '']
        + [f'    {entry}' for entry in entries]
        + ['']
    )


def write_toc():
    """Write the table-of-contents for the library based on `NB_BY_SECTION`."""
    header = [
        '.. _bloqs_library:',
        '',
        'Bloqs Library',
        '=============',
        '',
        '``qualtran.bloqs`` contains implementations of quantum operations and subroutines.',
        '',
        '.. Note: this file is autogenerated. See dev_tools/autogenerate-bloqs-notebooks-v2.py.',
        '',
    ]

    toc_lines = header + _get_toc_section_lines('Concepts', CONCEPTS, maxdepth=1)
    toc_lines += _get_toc_section_lines('Root Bloqs', ROOT_BLOQS, maxdepth=1)
    bloqs_dir = SOURCE_DIR / 'bloqs'
    for section, nbspecs in NB_BY_SECTION:
        entries = [str(nbspec.path.relative_to(bloqs_dir)) for nbspec in nbspecs]
        toc_lines += _get_toc_section_lines(section, entries)

    with (GIT_ROOT / 'docs/bloqs/index.rst').open('w') as f:
        f.write('\n'.join(toc_lines))


def check_all_bloqs_included():
    """Scour the library for BloqDocSpecs. Emit a warning if they're not listed in any of the nb specs."""
    bspecs = get_bloqdocspecs()
    rendered_bspecs = []
    for nbspec in _all_nbspecs():
        rendered_bspecs += [bspec for bspec in nbspec.bloq_specs]

    undoc = set(bspecs) - set(rendered_bspecs)
    if undoc:
        print("\nWarning: found a BloqDocSpec for these, but they're not in any NotebookSpecs:")
        for bspec in undoc:
            print('   ', bspec.bloq_cls.__name__)


if __name__ == '__main__':
    render_notebooks()
    write_toc()
    check_all_bloqs_included()
