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

For each notebook spec listed in the various global variables (in this file)
we write a notebook with a title, module docstring,
standard imports, and information on each bloq listed in the
`bloq_specs` field. For each bloq, we render a docstring and diagrams.

## Adding a new bloq.

 1. Create a qualtran.BloqExample perhaps using the `@bloq_example` decorator. Wrap it in a
    `qualtran.BloqDocSpec`. This code should live alongside the bloq.
 2. If this is a new module: add a new entry to the appropriate notebook spec global variable
    in this file (according to its category/organization).
 3. Update the `NotebookSpec` `bloq_specs` field to include the `BloqDocSpec` for your new bloq.

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
from qualtran_dev_tools.git_tools import get_git_root
from qualtran_dev_tools.jupyter_autogen_v2 import NotebookSpecV2, render_notebook

import qualtran.bloqs.arithmetic.addition
import qualtran.bloqs.arithmetic.sorting
import qualtran.bloqs.basic_gates.swap
import qualtran.bloqs.block_encoding
import qualtran.bloqs.chemistry.df.double_factorization
import qualtran.bloqs.chemistry.pbc.first_quantization.prepare_t
import qualtran.bloqs.chemistry.pbc.first_quantization.prepare_uv
import qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_and_prepare
import qualtran.bloqs.chemistry.pbc.first_quantization.select_t
import qualtran.bloqs.chemistry.pbc.first_quantization.select_uv
import qualtran.bloqs.chemistry.sf.single_factorization
import qualtran.bloqs.chemistry.sparse.prepare
import qualtran.bloqs.chemistry.thc.prepare
import qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt
import qualtran.bloqs.chemistry.trotter.grid_ham.qvr
import qualtran.bloqs.chemistry.trotter.hubbard.hopping
import qualtran.bloqs.chemistry.trotter.hubbard.interaction
import qualtran.bloqs.chemistry.trotter.ising.unitaries
import qualtran.bloqs.chemistry.trotter.trotterized_unitary
import qualtran.bloqs.data_loading.qrom
import qualtran.bloqs.factoring.mod_exp
import qualtran.bloqs.mcmt.and_bloq
import qualtran.bloqs.multiplexers.apply_gate_to_lth_target
import qualtran.bloqs.multiplexers.select_pauli_lcu
import qualtran.bloqs.phase_estimation.lp_resource_state
import qualtran.bloqs.qft.approximate_qft
import qualtran.bloqs.qft.two_bit_ffft
import qualtran.bloqs.qubitization_walk_operator
import qualtran.bloqs.reflection
import qualtran.bloqs.rotations.phasing_via_cost_function
import qualtran.bloqs.rotations.quantum_variable_rotation
import qualtran.bloqs.state_preparation.prepare_uniform_superposition
import qualtran.bloqs.state_preparation.state_preparation_alias_sampling
import qualtran.bloqs.swap_network.cswap_approx
import qualtran.bloqs.swap_network.multiplexed_cswap
import qualtran.bloqs.swap_network.swap_with_zero

GIT_ROOT = get_git_root()
SOURCE_DIR = GIT_ROOT / 'qualtran/'

# --------------------------------------------------------------------------
# -----   Basic Gates   ----------------------------------------------------
# --------------------------------------------------------------------------
BASIC_GATES: List[NotebookSpecV2] = [
    NotebookSpecV2(
        title='T Gate',
        module=qualtran.bloqs.basic_gates.t_gate,
        bloq_specs=[qualtran.bloqs.basic_gates.t_gate._T_GATE_DOC],
    ),
    NotebookSpecV2(
        title='Toffoli',
        module=qualtran.bloqs.basic_gates.toffoli,
        bloq_specs=[qualtran.bloqs.basic_gates.toffoli._TOFFOLI_DOC],
    ),
    NotebookSpecV2(
        title='Hadamard',
        module=qualtran.bloqs.basic_gates.hadamard,
        bloq_specs=[qualtran.bloqs.basic_gates.hadamard._HADAMARD_DOC],
    ),
    NotebookSpecV2(
        title='S Gate',
        module=qualtran.bloqs.basic_gates.s_gate,
        bloq_specs=[qualtran.bloqs.basic_gates.s_gate._S_GATE_DOC],
    ),
    NotebookSpecV2(
        title='And',
        module=qualtran.bloqs.mcmt.and_bloq,
        bloq_specs=[
            qualtran.bloqs.mcmt.and_bloq._AND_DOC,
            qualtran.bloqs.mcmt.and_bloq._MULTI_AND_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Swap Network',
        module=qualtran.bloqs.swap_network,
        bloq_specs=[
            qualtran.bloqs.basic_gates.swap._CSWAP_DOC,
            qualtran.bloqs.swap_network.cswap_approx._APPROX_CSWAP_DOC,
            qualtran.bloqs.swap_network.swap_with_zero._SWZ_DOC,
            qualtran.bloqs.swap_network.multiplexed_cswap._MULTIPLEXED_CSWAP_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Global Phase',
        module=qualtran.bloqs.basic_gates.global_phase,
        bloq_specs=[qualtran.bloqs.basic_gates.global_phase._GLOBAL_PHASE_DOC],
    ),
]


# --------------------------------------------------------------------------
# -----   Chemistry   ------------------------------------------------------
# --------------------------------------------------------------------------
CHEMISTRY: List[NotebookSpecV2] = [
    NotebookSpecV2(
        title='Sparse',
        module=qualtran.bloqs.chemistry.sparse,
        bloq_specs=[
            qualtran.bloqs.chemistry.sparse.prepare._SPARSE_PREPARE,
            qualtran.bloqs.chemistry.sparse.select_bloq._SPARSE_SELECT,
        ],
        directory=f'{SOURCE_DIR}/bloqs/chemistry/sparse',
    ),
    NotebookSpecV2(
        title='Single Factorization',
        module=qualtran.bloqs.chemistry.sf.single_factorization,
        bloq_specs=[
            qualtran.bloqs.chemistry.sf.single_factorization._SF_ONE_BODY,
            qualtran.bloqs.chemistry.sf.single_factorization._SF_BLOCK_ENCODING,
        ],
        directory=f'{SOURCE_DIR}/bloqs/chemistry/sf',
    ),
    NotebookSpecV2(
        title='Double Factorization',
        module=qualtran.bloqs.chemistry.df.double_factorization,
        bloq_specs=[
            qualtran.bloqs.chemistry.df.double_factorization._DF_ONE_BODY,
            qualtran.bloqs.chemistry.df.double_factorization._DF_BLOCK_ENCODING,
        ],
        directory=f'{SOURCE_DIR}/bloqs/chemistry/df',
    ),
    NotebookSpecV2(
        title='Tensor Hypercontraction',
        module=qualtran.bloqs.chemistry.thc,
        bloq_specs=[
            qualtran.bloqs.chemistry.thc.prepare._THC_UNI_PREP,
            qualtran.bloqs.chemistry.thc.prepare._THC_PREPARE,
            qualtran.bloqs.chemistry.thc.select_bloq._THC_SELECT,
        ],
        directory=f'{SOURCE_DIR}/bloqs/chemistry/thc',
    ),
    NotebookSpecV2(
        title='First Quantized Hamiltonian',
        module=qualtran.bloqs.chemistry.pbc.first_quantization,
        bloq_specs=[
            qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare._FIRST_QUANTIZED_PREPARE_DOC,
            qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare._FIRST_QUANTIZED_SELECT_DOC,
            qualtran.bloqs.chemistry.pbc.first_quantization.prepare_t._PREPARE_T,
            qualtran.bloqs.chemistry.pbc.first_quantization.prepare_uv._PREPARE_UV,
            qualtran.bloqs.chemistry.pbc.first_quantization.select_t._SELECT_T,
            qualtran.bloqs.chemistry.pbc.first_quantization.select_uv._SELECT_UV,
        ],
        directory=f'{SOURCE_DIR}/bloqs/chemistry/pbc/first_quantization',
    ),
    NotebookSpecV2(
        title='First Quantized Hamiltonian with Quantum Projectile',
        module=qualtran.bloqs.chemistry.pbc.first_quantization.projectile,
        bloq_specs=[
            qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_and_prepare._FIRST_QUANTIZED_WITH_PROJ_PREPARE_DOC,
            qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_and_prepare._FIRST_QUANTIZED_WITH_PROJ_SELECT_DOC,
        ],
        directory=f'{SOURCE_DIR}/bloqs/chemistry/pbc/first_quantization/projectile',
    ),
    NotebookSpecV2(
        title='Trotter Bloqs',
        module=qualtran.bloqs.chemistry.trotter.grid_ham,
        bloq_specs=[
            qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt._POLY_INV_SQRT,
            qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt._NR_INV_SQRT,
            qualtran.bloqs.chemistry.trotter.grid_ham.qvr._QVR,
            qualtran.bloqs.chemistry.trotter.grid_ham.kinetic._KINETIC_ENERGY,
            qualtran.bloqs.chemistry.trotter.grid_ham.potential._PAIR_POTENTIAL,
            qualtran.bloqs.chemistry.trotter.grid_ham.potential._POTENTIAL_ENERGY,
        ],
        path_stem='trotter',
    ),
    NotebookSpecV2(
        title='Trotterization',
        module=qualtran.bloqs.chemistry.trotter.trotterized_unitary,
        bloq_specs=[qualtran.bloqs.chemistry.trotter.trotterized_unitary._TROTT_UNITARY_DOC],
        directory=f'{SOURCE_DIR}/bloqs/chemistry/trotter',
    ),
    NotebookSpecV2(
        title='Ising Trotter Bloqs',
        module=qualtran.bloqs.chemistry.trotter.ising,
        bloq_specs=[
            qualtran.bloqs.chemistry.trotter.ising.unitaries._ISING_X_UNITARY_DOC,
            qualtran.bloqs.chemistry.trotter.ising.unitaries._ISING_ZZ_UNITARY_DOC,
        ],
        directory=f'{SOURCE_DIR}/bloqs/chemistry/trotter/ising',
    ),
    NotebookSpecV2(
        title='Trotterized Hubbard',
        module=qualtran.bloqs.chemistry.trotter.hubbard,
        bloq_specs=[
            qualtran.bloqs.chemistry.trotter.hubbard.hopping._HOPPING_DOC,
            qualtran.bloqs.chemistry.trotter.hubbard.hopping._PLAQUETTE_DOC,
            qualtran.bloqs.chemistry.trotter.hubbard.interaction._INTERACTION_DOC,
        ],
        directory=f'{SOURCE_DIR}/bloqs/chemistry/trotter/hubbard',
    ),
]

# --------------------------------------------------------------------------
# -----   Arithmetic   -----------------------------------------------------
# --------------------------------------------------------------------------
ARITHMETIC = [
    NotebookSpecV2(
        title='Addition',
        module=qualtran.bloqs.arithmetic.addition,
        bloq_specs=[
            qualtran.bloqs.arithmetic.addition._ADD_DOC,
            qualtran.bloqs.arithmetic.addition._ADD_OOP_DOC,
            qualtran.bloqs.arithmetic.addition._SIMPLE_ADD_K_DOC,
            qualtran.bloqs.arithmetic.addition._ADD_K_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Multiplication',
        module=qualtran.bloqs.arithmetic.multiplication,
        bloq_specs=[
            qualtran.bloqs.arithmetic.multiplication._PLUS_EQUALS_PRODUCT_DOC,
            qualtran.bloqs.arithmetic.multiplication._PRODUCT_DOC,
            qualtran.bloqs.arithmetic.multiplication._SQUARE_DOC,
            qualtran.bloqs.arithmetic.multiplication._SUM_OF_SQUARES_DOC,
            qualtran.bloqs.arithmetic.multiplication._SCALE_INT_BY_REAL_DOC,
            qualtran.bloqs.arithmetic.multiplication._MULTIPLY_TWO_REALS_DOC,
            qualtran.bloqs.arithmetic.multiplication._SQUARE_REAL_NUMBER_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Comparison',
        module=qualtran.bloqs.arithmetic.comparison,
        bloq_specs=[
            qualtran.bloqs.arithmetic.comparison._LT_K_DOC,
            qualtran.bloqs.arithmetic.comparison._GREATER_THAN_DOC,
            qualtran.bloqs.arithmetic.comparison._GREATER_THAN_K_DOC,
            qualtran.bloqs.arithmetic.comparison._EQUALS_K_DOC,
            qualtran.bloqs.arithmetic.comparison._BI_QUBITS_MIXER_DOC,
            qualtran.bloqs.arithmetic.comparison._SQ_CMP_DOC,
            qualtran.bloqs.arithmetic.comparison._LEQ_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Sorting',
        module=qualtran.bloqs.arithmetic.sorting,
        bloq_specs=[
            qualtran.bloqs.arithmetic.sorting._COMPARATOR_DOC,
            qualtran.bloqs.arithmetic.sorting._BITONIC_SORT_DOC,
        ],
        directory=f'{SOURCE_DIR}/bloqs/arithmetic/',
    ),
    NotebookSpecV2(
        title='Conversions',
        module=qualtran.bloqs.arithmetic.conversions,
        bloq_specs=[
            qualtran.bloqs.arithmetic.conversions._SIGNED_TO_TWOS,
            qualtran.bloqs.arithmetic.conversions._TO_CONTG_INDX,
        ],
    ),
    NotebookSpecV2(
        title='Modular Exponentiation',
        module=qualtran.bloqs.factoring.mod_exp,
        bloq_specs=[qualtran.bloqs.factoring.mod_exp._MODEXP_DOC],
        directory=f'{SOURCE_DIR}/bloqs/factoring',
    ),
    NotebookSpecV2(
        title='Modular Multiplication',
        module=qualtran.bloqs.factoring.mod_mul,
        bloq_specs=[qualtran.bloqs.factoring.mod_mul._MODMUL_DOC],
        directory=f'{SOURCE_DIR}/bloqs/factoring',
    ),
]


ROT_QFT_PE = [
    # --------------------------------------------------------------------------
    # -----   Rotations    -----------------------------------------------------
    # --------------------------------------------------------------------------
    NotebookSpecV2(
        title='Quantum Variable Rotation',
        module=qualtran.bloqs.rotations.quantum_variable_rotation,
        bloq_specs=[
            qualtran.bloqs.rotations.quantum_variable_rotation._QVR_ZPOW,
            qualtran.bloqs.rotations.quantum_variable_rotation._QVR_PHASE_GRADIENT,
        ],
        directory=f'{SOURCE_DIR}/bloqs/rotations/',
    ),
    NotebookSpecV2(
        title='Phasing via Cost function',
        module=qualtran.bloqs.rotations.phasing_via_cost_function,
        bloq_specs=[qualtran.bloqs.rotations.phasing_via_cost_function._PHASING_VIA_COST_FUNCTION],
        directory=f'{SOURCE_DIR}/bloqs/rotations/',
    ),
    # --------------------------------------------------------------------------
    # -----   QFT          -----------------------------------------------------
    # --------------------------------------------------------------------------
    NotebookSpecV2(
        title='Two Bit FFFT Gate',
        module=qualtran.bloqs.qft.two_bit_ffft,
        bloq_specs=[qualtran.bloqs.qft.two_bit_ffft._TWO_BIT_FFFT_DOC],
    ),
    NotebookSpecV2(
        title='Approximate QFT',
        module=qualtran.bloqs.qft.approximate_qft,
        bloq_specs=[qualtran.bloqs.qft.approximate_qft._CC_AQFT_DOC],
    ),
    # --------------------------------------------------------------------------
    # -----   Phase Estimation          ----------------------------------------
    # --------------------------------------------------------------------------
    NotebookSpecV2(
        title='Optimal resource states for Phase Estimation by A. Luis and J. Peřina',
        module=qualtran.bloqs.phase_estimation.lp_resource_state,
        bloq_specs=[
            qualtran.bloqs.phase_estimation.lp_resource_state._CC_LPRS_INTERIM_PREP_DOC,
            qualtran.bloqs.phase_estimation.lp_resource_state._CC_LP_RESOURCE_STATE_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Textbook Quantum Phase Estimation',
        module=qualtran.bloqs.phase_estimation.text_book_qpe,
        bloq_specs=[
            qualtran.bloqs.phase_estimation.text_book_qpe._CC_TEXTBOOK_PHASE_ESTIMATION_DOC
        ],
    ),
    NotebookSpecV2(
        title='Qubitization Walk Operator',
        module=qualtran.bloqs.qubitization_walk_operator,
        bloq_specs=[qualtran.bloqs.qubitization_walk_operator._QUBITIZATION_WALK_DOC],
    ),
]

# --------------------------------------------------------------------------
# -----   Other   ----------------------------------------------------------
# --------------------------------------------------------------------------
OTHER: List[NotebookSpecV2] = [
    NotebookSpecV2(
        title='Prepare Uniform Superposition',
        module=qualtran.bloqs.state_preparation.prepare_uniform_superposition,
        bloq_specs=[
            qualtran.bloqs.state_preparation.prepare_uniform_superposition._PREP_UNIFORM_DOC
        ],
    ),
    NotebookSpecV2(
        title='Apply to Lth Target',
        module=qualtran.bloqs.multiplexers.apply_gate_to_lth_target,
        bloq_specs=[qualtran.bloqs.multiplexers.apply_gate_to_lth_target._APPLYLTH_DOC],
    ),
    NotebookSpecV2(
        title='QROM',
        module=qualtran.bloqs.data_loading.qrom,
        bloq_specs=[qualtran.bloqs.data_loading.qrom._QROM_DOC],
    ),
    NotebookSpecV2(
        title='Block Encoding',
        module=qualtran.bloqs.block_encoding,
        bloq_specs=[
            qualtran.bloqs.block_encoding._BLACK_BOX_BLOCK_BLOQ_DOC,
            qualtran.bloqs.block_encoding._CHEBYSHEV_BLOQ_DOC,
        ],
        directory=f'{SOURCE_DIR}/bloqs/',
    ),
    NotebookSpecV2(
        title='Reflection',
        module=qualtran.bloqs.reflection,
        bloq_specs=[qualtran.bloqs.reflection._REFLECTION_DOC],
        directory=f'{SOURCE_DIR}/bloqs/',
    ),
    NotebookSpecV2(
        title='Multi-Paulis',
        module=qualtran.bloqs.mcmt.multi_control_multi_target_pauli,
        bloq_specs=[
            qualtran.bloqs.mcmt.multi_control_multi_target_pauli._C_MULTI_NOT_DOC,
            qualtran.bloqs.mcmt.multi_control_multi_target_pauli._CC_PAULI_DOC,
        ],
        directory=f'{SOURCE_DIR}/bloqs/mcmt/',
    ),
    NotebookSpecV2(
        title='Generic Select',
        module=qualtran.bloqs.multiplexers.select_pauli_lcu,
        bloq_specs=[qualtran.bloqs.multiplexers.select_pauli_lcu._SELECT_PAULI_LCU_DOC],
    ),
    NotebookSpecV2(
        title='State Preparation via Alias Sampling',
        module=qualtran.bloqs.state_preparation.state_preparation_alias_sampling,
        bloq_specs=[
            qualtran.bloqs.state_preparation.state_preparation_alias_sampling._STATE_PREP_ALIAS_DOC
        ],
    ),
    NotebookSpecV2(
        title='State Preparation Using Rotations',
        module=qualtran.bloqs.state_preparation.state_preparation_via_rotation,
        bloq_specs=[
            qualtran.bloqs.state_preparation.state_preparation_via_rotation._STATE_PREP_VIA_ROTATIONS_DOC
        ],
        directory=f'{SOURCE_DIR}/bloqs/state_preparation/',
    ),
]

# --------------------------------------------------------------------------
# -----   Concepts   -------------------------------------------------------
# --------------------------------------------------------------------------
CONCEPTS = [
    # Note! These are just straight paths to existing notebooks. Used to generate
    # the table of contents.
    'multiplexers/unary_iteration.ipynb',
    'arithmetic/t_complexity_of_comparison_gates.ipynb',
    'arithmetic/error_analysis_for_fxp_arithmetic.ipynb',
    'phase_estimation_of_quantum_walk.ipynb',
    'chemistry/trotter/grid_ham/trotter_costs.ipynb',
    'chemistry/resource_estimation.ipynb',
    'chemistry/writing_algorithms.ipynb',
    'factoring/factoring-via-modexp.ipynb',
    'state_preparation/state_preparation_via_rotation_tutorial.ipynb',
]

NB_BY_SECTION = [
    ('Basic Gates', BASIC_GATES),
    ('Chemistry', CHEMISTRY),
    ('Arithmetic', ARITHMETIC),
    ('Rotations', ROT_QFT_PE),
    ('Other', OTHER),
]


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
