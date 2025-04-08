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

"""List of Jupyter notebooks.

The notebooks listed in this file are used to generate Jupyter notebooks that
document and provide examples for each of the bloqs. This list is also used to
generate the static exports for the Qualtran web UI.

## Adding a new bloq.

 1. Create a qualtran.BloqExample perhaps using the `@bloq_example` decorator. Wrap it in a
    `qualtran.BloqDocSpec`. This code should live alongside the bloq.
 2. If this is a new module: add a new entry to the appropriate notebook spec global variable
    in this file (according to its category/organization).
 3. Update the `NotebookSpec` `bloq_specs` field to include the `BloqDocSpec` for your new bloq.
"""

from typing import List

from qualtran_dev_tools.git_tools import get_git_root

import qualtran.bloqs.arithmetic.addition
import qualtran.bloqs.arithmetic.bitwise
import qualtran.bloqs.arithmetic.comparison
import qualtran.bloqs.arithmetic.controlled_add_or_subtract
import qualtran.bloqs.arithmetic.controlled_addition
import qualtran.bloqs.arithmetic.conversions
import qualtran.bloqs.arithmetic.lists
import qualtran.bloqs.arithmetic.multiplication
import qualtran.bloqs.arithmetic.negate
import qualtran.bloqs.arithmetic.permutation
import qualtran.bloqs.arithmetic.sorting
import qualtran.bloqs.arithmetic.subtraction
import qualtran.bloqs.arithmetic.trigonometric
import qualtran.bloqs.basic_gates.swap
import qualtran.bloqs.block_encoding.block_encoding_base
import qualtran.bloqs.block_encoding.chebyshev_polynomial
import qualtran.bloqs.block_encoding.lcu_block_encoding
import qualtran.bloqs.block_encoding.linear_combination
import qualtran.bloqs.block_encoding.phase
import qualtran.bloqs.block_encoding.product
import qualtran.bloqs.block_encoding.sparse_matrix
import qualtran.bloqs.block_encoding.sparse_matrix_hermitian
import qualtran.bloqs.block_encoding.tensor_product
import qualtran.bloqs.block_encoding.unitary
import qualtran.bloqs.bookkeeping
import qualtran.bloqs.bookkeeping.allocate
import qualtran.bloqs.bookkeeping.auto_partition
import qualtran.bloqs.bookkeeping.cast
import qualtran.bloqs.bookkeeping.free
import qualtran.bloqs.bookkeeping.join
import qualtran.bloqs.bookkeeping.partition
import qualtran.bloqs.bookkeeping.split
import qualtran.bloqs.chemistry.df.double_factorization
import qualtran.bloqs.chemistry.hubbard_model.qubitization
import qualtran.bloqs.chemistry.pbc.first_quantization.prepare_t
import qualtran.bloqs.chemistry.pbc.first_quantization.prepare_uv
import qualtran.bloqs.chemistry.pbc.first_quantization.prepare_zeta
import qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_and_prepare
import qualtran.bloqs.chemistry.pbc.first_quantization.select_t
import qualtran.bloqs.chemistry.pbc.first_quantization.select_uv
import qualtran.bloqs.chemistry.quad_fermion.givens_bloq
import qualtran.bloqs.chemistry.sf.single_factorization
import qualtran.bloqs.chemistry.sparse.prepare
import qualtran.bloqs.chemistry.sparse.walk_operator
import qualtran.bloqs.chemistry.thc.prepare
import qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt
import qualtran.bloqs.chemistry.trotter.grid_ham.qvr
import qualtran.bloqs.chemistry.trotter.hubbard.hopping
import qualtran.bloqs.chemistry.trotter.hubbard.interaction
import qualtran.bloqs.chemistry.trotter.ising.unitaries
import qualtran.bloqs.chemistry.trotter.trotterized_unitary
import qualtran.bloqs.cryptography.ecc
import qualtran.bloqs.cryptography.rsa
import qualtran.bloqs.data_loading.qrom
import qualtran.bloqs.data_loading.qrom_base
import qualtran.bloqs.data_loading.select_swap_qrom
import qualtran.bloqs.gf_arithmetic.gf2_add_k
import qualtran.bloqs.gf_arithmetic.gf2_addition
import qualtran.bloqs.gf_arithmetic.gf2_inverse
import qualtran.bloqs.gf_arithmetic.gf2_multiplication
import qualtran.bloqs.gf_arithmetic.gf2_square
import qualtran.bloqs.gf_poly_arithmetic.gf2_poly_add_k
import qualtran.bloqs.gf_poly_arithmetic.gf_poly_split_and_join
import qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp
import qualtran.bloqs.mcmt.and_bloq
import qualtran.bloqs.mcmt.controlled_via_and
import qualtran.bloqs.mcmt.ctrl_spec_and
import qualtran.bloqs.mcmt.multi_control_pauli
import qualtran.bloqs.mcmt.multi_target_cnot
import qualtran.bloqs.mod_arithmetic.mod_addition
import qualtran.bloqs.multiplexers.apply_gate_to_lth_target
import qualtran.bloqs.multiplexers.apply_lth_bloq
import qualtran.bloqs.multiplexers.black_box_select
import qualtran.bloqs.multiplexers.select_base
import qualtran.bloqs.multiplexers.select_pauli_lcu
import qualtran.bloqs.optimization.k_xor_sat.kikuchi_guiding_state
import qualtran.bloqs.phase_estimation.lp_resource_state
import qualtran.bloqs.phase_estimation.qubitization_qpe
import qualtran.bloqs.phase_estimation.text_book_qpe
import qualtran.bloqs.qft.approximate_qft
import qualtran.bloqs.qft.qft_phase_gradient
import qualtran.bloqs.qft.qft_text_book
import qualtran.bloqs.qft.two_bit_ffft
import qualtran.bloqs.qsp.generalized_qsp
import qualtran.bloqs.qubitization.qubitization_walk_operator
import qualtran.bloqs.reflections
import qualtran.bloqs.reflections.prepare_identity
import qualtran.bloqs.reflections.reflection_using_prepare
import qualtran.bloqs.rotations.hamming_weight_phasing
import qualtran.bloqs.rotations.phase_gradient
import qualtran.bloqs.rotations.phasing_via_cost_function
import qualtran.bloqs.rotations.programmable_rotation_gate_array
import qualtran.bloqs.rotations.quantum_variable_rotation
import qualtran.bloqs.rotations.rz_via_phase_gradient
import qualtran.bloqs.rotations.zpow_via_phase_gradient
import qualtran.bloqs.state_preparation.black_box_prepare
import qualtran.bloqs.state_preparation.prepare_base
import qualtran.bloqs.state_preparation.prepare_uniform_superposition
import qualtran.bloqs.state_preparation.state_preparation_alias_sampling
import qualtran.bloqs.state_preparation.state_preparation_via_rotation
import qualtran.bloqs.swap_network.cswap_approx
import qualtran.bloqs.swap_network.multiplexed_cswap
import qualtran.bloqs.swap_network.swap_with_zero

from .jupyter_autogen import NotebookSpecV2

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
        bloq_specs=[
            qualtran.bloqs.basic_gates.hadamard._HADAMARD_DOC,
            qualtran.bloqs.basic_gates.hadamard._CHADAMARD_DOC,
        ],
    ),
    NotebookSpecV2(
        title='CNOT',
        module=qualtran.bloqs.basic_gates.cnot,
        bloq_specs=[qualtran.bloqs.basic_gates.cnot._CNOT_DOC],
    ),
    NotebookSpecV2(
        title='Z, S, and CZ',
        module=qualtran.bloqs.basic_gates.z_basis,
        path_stem='diag_gates',
        bloq_specs=[
            qualtran.bloqs.basic_gates.z_basis._Z_GATE_DOC,
            qualtran.bloqs.basic_gates.s_gate._S_GATE_DOC,
            qualtran.bloqs.basic_gates.z_basis._CZ_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Y Gate',
        module=qualtran.bloqs.basic_gates.y_gate,
        bloq_specs=[
            qualtran.bloqs.basic_gates.y_gate._Y_GATE_DOC,
            qualtran.bloqs.basic_gates.y_gate._CY_GATE_DOC,
        ],
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
        title='States and Effects',
        module=qualtran.bloqs.basic_gates.z_basis,
        path_stem='states_and_effects',
        bloq_specs=[
            qualtran.bloqs.basic_gates.z_basis._ZERO_STATE_DOC,
            qualtran.bloqs.basic_gates.z_basis._ZERO_EFFECT_DOC,
            qualtran.bloqs.basic_gates.z_basis._ONE_STATE_DOC,
            qualtran.bloqs.basic_gates.z_basis._ONE_EFFECT_DOC,
            qualtran.bloqs.basic_gates.z_basis._INT_STATE_DOC,
            qualtran.bloqs.basic_gates.z_basis._INT_EFFECT_DOC,
            qualtran.bloqs.basic_gates.x_basis._PLUS_STATE_DOC,
            qualtran.bloqs.basic_gates.x_basis._PLUS_EFFECT_DOC,
            qualtran.bloqs.basic_gates.x_basis._MINUS_STATE_DOC,
            qualtran.bloqs.basic_gates.x_basis._MINUS_EFFECT_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Basic Swaps',
        module=qualtran.bloqs.basic_gates.swap,
        bloq_specs=[
            qualtran.bloqs.basic_gates.swap._TWO_BIT_SWAP_DOC,
            qualtran.bloqs.basic_gates.swap._TWO_BIT_CSWAP_DOC,
            qualtran.bloqs.basic_gates.swap._SWAP_DOC,
            qualtran.bloqs.basic_gates.swap._CSWAP_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Swap Networks',
        module=qualtran.bloqs.swap_network,
        bloq_specs=[
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
    NotebookSpecV2(
        title='Identity Gate',
        module=qualtran.bloqs.basic_gates.identity,
        bloq_specs=[qualtran.bloqs.basic_gates.identity._IDENTITY_DOC],
    ),
    NotebookSpecV2(
        title='Bookkeeping Bloqs',
        module=qualtran.bloqs.bookkeeping,
        bloq_specs=[
            qualtran.bloqs.bookkeeping.allocate._ALLOC_DOC,
            qualtran.bloqs.bookkeeping.free._FREE_DOC,
            qualtran.bloqs.bookkeeping.split._SPLIT_DOC,
            qualtran.bloqs.bookkeeping.join._JOIN_DOC,
            qualtran.bloqs.bookkeeping.partition._PARTITION_DOC,
            qualtran.bloqs.bookkeeping.auto_partition._AUTO_PARTITION_DOC,
            qualtran.bloqs.bookkeeping.cast._CAST_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Control Specification (And)',
        module=qualtran.bloqs.mcmt.ctrl_spec_and,
        bloq_specs=[qualtran.bloqs.mcmt.ctrl_spec_and._CTRLSPEC_AND_DOC],
    ),
    NotebookSpecV2(
        title='Multi control bloq via single control bloq and `And` ladder',
        module=qualtran.bloqs.mcmt.controlled_via_and,
        bloq_specs=[qualtran.bloqs.mcmt.controlled_via_and._CONTROLLED_VIA_AND_DOC],
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
            qualtran.bloqs.chemistry.pbc.first_quantization.prepare_zeta._PREPARE_ZETA,
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
            qualtran.bloqs.chemistry.trotter.hubbard.hopping._HOPPING_TILE_HWP_DOC,
            qualtran.bloqs.chemistry.trotter.hubbard.interaction._INTERACTION_DOC,
            qualtran.bloqs.chemistry.trotter.hubbard.interaction._INTERACTION_HWP_DOC,
        ],
        directory=f'{SOURCE_DIR}/bloqs/chemistry/trotter/hubbard',
    ),
    NotebookSpecV2(
        title='Givens Rotations',
        module=qualtran.bloqs.chemistry.quad_fermion.givens_bloq,
        bloq_specs=[
            qualtran.bloqs.chemistry.quad_fermion.givens_bloq._REAL_GIVENS_DOC,
            qualtran.bloqs.chemistry.quad_fermion.givens_bloq._CPLX_GIVENS_DOC,
        ],
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
            qualtran.bloqs.arithmetic.addition._ADD_K_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Controlled Addition',
        module=qualtran.bloqs.arithmetic.controlled_addition,
        bloq_specs=[qualtran.bloqs.arithmetic.controlled_addition._CADD_DOC],
    ),
    NotebookSpecV2(
        title='Negation',
        module=qualtran.bloqs.arithmetic.negate,
        bloq_specs=[qualtran.bloqs.arithmetic.negate._NEGATE_DOC],
    ),
    NotebookSpecV2(
        title='Subtraction',
        module=qualtran.bloqs.arithmetic.subtraction,
        bloq_specs=[
            qualtran.bloqs.arithmetic.subtraction._SUB_DOC,
            qualtran.bloqs.arithmetic.subtraction._SUB_FROM_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Controlled Add-or-Subtract',
        module=qualtran.bloqs.arithmetic.controlled_add_or_subtract,
        bloq_specs=[
            qualtran.bloqs.arithmetic.controlled_add_or_subtract._CONTROLLED_ADD_OR_SUBTRACT_DOC
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
            qualtran.bloqs.arithmetic.multiplication._INVERT_REAL_NUMBER_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Comparison',
        module=qualtran.bloqs.arithmetic.comparison,
        bloq_specs=[
            qualtran.bloqs.arithmetic.comparison._LT_K_DOC,
            qualtran.bloqs.arithmetic.comparison._GREATER_THAN_DOC,
            qualtran.bloqs.arithmetic.comparison._GREATER_THAN_K_DOC,
            qualtran.bloqs.arithmetic.comparison._EQUALS_DOC,
            qualtran.bloqs.arithmetic.comparison._EQUALS_K_DOC,
            qualtran.bloqs.arithmetic.comparison._BI_QUBITS_MIXER_DOC,
            qualtran.bloqs.arithmetic.comparison._SQ_CMP_DOC,
            qualtran.bloqs.arithmetic.comparison._LEQ_DOC,
            qualtran.bloqs.arithmetic.comparison._CLinearDepthGreaterThan_DOC,
            qualtran.bloqs.arithmetic.comparison._LINEAR_DEPTH_HALF_GREATERTHAN_DOC,
            qualtran.bloqs.arithmetic.comparison._LINEAR_DEPTH_HALF_GREATERTHANEQUAL_DOC,
            qualtran.bloqs.arithmetic.comparison._LINEAR_DEPTH_HALF_LESSTHAN_DOC,
            qualtran.bloqs.arithmetic.comparison._LINEAR_DEPTH_HALF_LESSTHANEQUAL_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Integer Conversions',
        module=qualtran.bloqs.arithmetic.conversions,
        bloq_specs=[
            qualtran.bloqs.arithmetic.conversions.ones_complement_to_twos_complement._SIGNED_TO_TWOS,
            qualtran.bloqs.arithmetic.conversions.sign_extension._SIGN_EXTEND_DOC,
            qualtran.bloqs.arithmetic.conversions.sign_extension._SIGN_TRUNCATE_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Sorting',
        module=qualtran.bloqs.arithmetic.sorting,
        bloq_specs=[
            qualtran.bloqs.arithmetic.sorting._COMPARATOR_DOC,
            qualtran.bloqs.arithmetic.sorting._PARALLEL_COMPARATORS_DOC,
            qualtran.bloqs.arithmetic.sorting._BITONIC_MERGE_DOC,
            qualtran.bloqs.arithmetic.sorting._BITONIC_SORT_DOC,
        ],
        directory=f'{SOURCE_DIR}/bloqs/arithmetic/',
    ),
    NotebookSpecV2(
        title='Indexing',
        module=qualtran.bloqs.arithmetic.conversions.contiguous_index,
        bloq_specs=[qualtran.bloqs.arithmetic.conversions.contiguous_index._TO_CONTG_INDX],
    ),
    NotebookSpecV2(
        title='Permutations',
        module=qualtran.bloqs.arithmetic.permutation,
        bloq_specs=[
            qualtran.bloqs.arithmetic.permutation._PERMUTATION_DOC,
            qualtran.bloqs.arithmetic.permutation._PERMUTATION_CYCLE_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Bitwise Operations',
        module=qualtran.bloqs.arithmetic.bitwise,
        bloq_specs=[
            qualtran.bloqs.arithmetic.bitwise._XOR_DOC,
            qualtran.bloqs.arithmetic.bitwise._BITWISE_NOT_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Trigonometric Functions',
        module=qualtran.bloqs.arithmetic.trigonometric,
        bloq_specs=[qualtran.bloqs.arithmetic.trigonometric.arcsin._ARCSIN_DOC],
    ),
    NotebookSpecV2(
        title='List Functions',
        module=qualtran.bloqs.arithmetic.lists,
        bloq_specs=[
            qualtran.bloqs.arithmetic.lists.sort_in_place._SORT_IN_PLACE_DOC,
            qualtran.bloqs.arithmetic.lists.symmetric_difference._SYMMETRIC_DIFFERENCE_DOC,
            qualtran.bloqs.arithmetic.lists.has_duplicates._HAS_DUPLICATES_DOC,
        ],
    ),
]

MOD_ARITHMETIC = [
    NotebookSpecV2(
        title='Modular Addition',
        module=qualtran.bloqs.mod_arithmetic.mod_addition,
        bloq_specs=[
            qualtran.bloqs.mod_arithmetic.mod_addition._MOD_ADD_DOC,
            qualtran.bloqs.mod_arithmetic.mod_addition._MOD_ADD_K_DOC,
            qualtran.bloqs.mod_arithmetic.mod_addition._C_MOD_ADD_DOC,
            qualtran.bloqs.mod_arithmetic.mod_addition._C_MOD_ADD_K_DOC,
            qualtran.bloqs.mod_arithmetic.mod_addition._CTRL_SCALE_MOD_ADD_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Modular Subtraction',
        module=qualtran.bloqs.mod_arithmetic.mod_subtraction,
        bloq_specs=[
            qualtran.bloqs.mod_arithmetic.mod_subtraction._MOD_NEG_DOC,
            qualtran.bloqs.mod_arithmetic.mod_subtraction._CMOD_NEG_DOC,
            qualtran.bloqs.mod_arithmetic.mod_subtraction._MOD_SUB_DOC,
            qualtran.bloqs.mod_arithmetic.mod_subtraction._CMOD_SUB_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Modular Multiplication',
        module=qualtran.bloqs.mod_arithmetic.mod_multiplication,
        bloq_specs=[
            qualtran.bloqs.mod_arithmetic.mod_multiplication._MOD_DBL_DOC,
            qualtran.bloqs.mod_arithmetic.mod_multiplication._C_MOD_MUL_K_DOC,
            qualtran.bloqs.mod_arithmetic.mod_multiplication._DIRTY_OUT_OF_PLACE_MONTGOMERY_MOD_MUL_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Modular Divison',
        module=qualtran.bloqs.mod_arithmetic.mod_division,
        bloq_specs=[qualtran.bloqs.mod_arithmetic.mod_division._KALISKI_MOD_INVERSE_DOC],
    ),
    NotebookSpecV2(
        title='Factoring RSA',
        module=qualtran.bloqs.cryptography.rsa,
        bloq_specs=[
            qualtran.bloqs.cryptography.rsa.rsa_phase_estimate._RSA_PE_BLOQ_DOC,
            qualtran.bloqs.cryptography.rsa.rsa_mod_exp._RSA_MODEXP_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Elliptic Curve Addition',
        module=qualtran.bloqs.cryptography.ecc.ec_add,
        bloq_specs=[qualtran.bloqs.cryptography.ecc.ec_add._EC_ADD_DOC],
    ),
    NotebookSpecV2(
        title='Elliptic Curve Cryptography',
        module=qualtran.bloqs.cryptography.ecc,
        bloq_specs=[
            qualtran.bloqs.cryptography.ecc.find_ecc_private_key._ECC_BLOQ_DOC,
            qualtran.bloqs.cryptography.ecc.ec_phase_estimate_r._EC_PE_BLOQ_DOC,
            qualtran.bloqs.cryptography.ecc.ec_add_r._ECC_ADD_R_BLOQ_DOC,
            qualtran.bloqs.cryptography.ecc.ec_add_r._EC_WINDOW_ADD_BLOQ_DOC,
        ],
    ),
]

GF_ARITHMETIC = [
    # --------------------------------------------------------------------------
    # -----   Galois Fields (GF) Arithmetic    ---------------------------------
    # --------------------------------------------------------------------------
    NotebookSpecV2(
        title='GF($2^m$) Multiplication',
        module=qualtran.bloqs.gf_arithmetic.gf2_multiplication,
        bloq_specs=[
            qualtran.bloqs.gf_arithmetic.gf2_multiplication._GF2_MULTIPLICATION_DOC,
            qualtran.bloqs.gf_arithmetic.gf2_multiplication._MULTIPLY_BY_CONSTANT_MOD_DOC,
            qualtran.bloqs.gf_arithmetic.gf2_multiplication._MULTIPLY_POLY_BY_ONE_PLUS_XK_DOC,
            qualtran.bloqs.gf_arithmetic.gf2_multiplication._BINARY_POLYNOMIAL_MULTIPLICATION_DOC,
            qualtran.bloqs.gf_arithmetic.gf2_multiplication._GF2_SHIFT_RIGHT_MOD_DOC,
            qualtran.bloqs.gf_arithmetic.gf2_multiplication._GF2_MUL_DOC,
        ],
    ),
    NotebookSpecV2(
        title='GF($2^m$) Addition',
        module=qualtran.bloqs.gf_arithmetic.gf2_addition,
        bloq_specs=[qualtran.bloqs.gf_arithmetic.gf2_addition._GF2_ADDITION_DOC],
    ),
    NotebookSpecV2(
        title='GF($2^m$) Add Constant',
        module=qualtran.bloqs.gf_arithmetic.gf2_add_k,
        bloq_specs=[qualtran.bloqs.gf_arithmetic.gf2_add_k._GF2_ADD_K_DOC],
    ),
    NotebookSpecV2(
        title='GF($2^m$) Square',
        module=qualtran.bloqs.gf_arithmetic.gf2_square,
        bloq_specs=[qualtran.bloqs.gf_arithmetic.gf2_square._GF2_SQUARE_DOC],
    ),
    NotebookSpecV2(
        title='GF($2^m$) Inverse',
        module=qualtran.bloqs.gf_arithmetic.gf2_inverse,
        bloq_specs=[qualtran.bloqs.gf_arithmetic.gf2_inverse._GF2_INVERSE_DOC],
    ),
]

GF_POLY_ARITHMETIC = [
    # --------------------------------------------------------------------------
    # -----   Polynomials defined over Galois Fields (GF) Arithmetic    --------
    # --------------------------------------------------------------------------
    NotebookSpecV2(
        title='Polynomials over GF($p^m$) - Split and Join',
        module=qualtran.bloqs.gf_poly_arithmetic.gf_poly_split_and_join,
        bloq_specs=[
            qualtran.bloqs.gf_poly_arithmetic.gf_poly_split_and_join._GF_POLY_SPLIT_DOC,
            qualtran.bloqs.gf_poly_arithmetic.gf_poly_split_and_join._GF_POLY_JOIN_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Polynomials over GF($2^m$) - Add Constant',
        module=qualtran.bloqs.gf_poly_arithmetic.gf2_poly_add_k,
        bloq_specs=[qualtran.bloqs.gf_poly_arithmetic.gf2_poly_add_k._GF2_POLY_ADD_K_DOC],
    ),
]

ROT_QFT_PE = [
    # --------------------------------------------------------------------------
    # -----   Rotations    -----------------------------------------------------
    # --------------------------------------------------------------------------
    NotebookSpecV2(
        title='Basic Rotation Gates',
        module=qualtran.bloqs.basic_gates.rotation,
        bloq_specs=[
            qualtran.bloqs.basic_gates.rotation._Z_POW_DOC,
            qualtran.bloqs.basic_gates.rotation._CZ_POW_DOC,
            qualtran.bloqs.basic_gates.rotation._RZ_DOC,
            qualtran.bloqs.basic_gates.rotation._CRZ_DOC,
            qualtran.bloqs.basic_gates.rotation._X_POW_DOC,
            qualtran.bloqs.basic_gates.rotation._Y_POW_DOC,
        ],
    ),
    NotebookSpecV2(
        title='SU2 Rotation',
        module=qualtran.bloqs.basic_gates.su2_rotation,
        bloq_specs=[qualtran.bloqs.basic_gates.su2_rotation._SU2_ROTATION_GATE_DOC],
    ),
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
    NotebookSpecV2(
        title='Rotations via Phase Gradients',
        module=qualtran.bloqs.rotations.phase_gradient,
        bloq_specs=[
            qualtran.bloqs.rotations.phase_gradient._PHASE_GRADIENT_UNITARY_DOC,
            qualtran.bloqs.rotations.phase_gradient._PHASE_GRADIENT_STATE_DOC,
            qualtran.bloqs.rotations.phase_gradient._ADD_INTO_PHASE_GRAD_DOC,
            qualtran.bloqs.rotations.phase_gradient._ADD_SCALED_VAL_INTO_PHASE_REG_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Z Rotations via Hamming Weight Phasing',
        module=qualtran.bloqs.rotations.hamming_weight_phasing,
        bloq_specs=[
            qualtran.bloqs.rotations.hamming_weight_phasing._HAMMING_WEIGHT_PHASING_DOC,
            qualtran.bloqs.rotations.hamming_weight_phasing._HAMMING_WEIGHT_PHASING_VIA_PHASE_GRADIENT_DOC,
        ],
    ),
    NotebookSpecV2(
        title='ZPow Rotation via Phase Gradient',
        module=qualtran.bloqs.rotations.zpow_via_phase_gradient,
        bloq_specs=[
            qualtran.bloqs.rotations.zpow_via_phase_gradient._ZPOW_CONST_VIA_PHASE_GRADIENT_DOC
        ],
    ),
    NotebookSpecV2(
        title='Rz Rotation via Phase Gradient',
        module=qualtran.bloqs.rotations.rz_via_phase_gradient,
        bloq_specs=[qualtran.bloqs.rotations.rz_via_phase_gradient._RZ_VIA_PHASE_GRADIENT_DOC],
    ),
    NotebookSpecV2(
        title='Programmable Rotation Gate Array',
        module=qualtran.bloqs.rotations.programmable_rotation_gate_array,
        bloq_specs=[
            qualtran.bloqs.rotations.programmable_rotation_gate_array._PROGRAMMABLE_ROTATAION_GATE_ARRAY_DOC
        ],
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
    NotebookSpecV2(
        title='Textbook QFT',
        module=qualtran.bloqs.qft.qft_text_book,
        bloq_specs=[qualtran.bloqs.qft.qft_text_book._QFT_TEXT_BOOK_DOC],
    ),
    NotebookSpecV2(
        title='Phase Gradient QFT',
        module=qualtran.bloqs.qft.qft_phase_gradient,
        bloq_specs=[qualtran.bloqs.qft.qft_phase_gradient._QFT_PHASE_GRADIENT_DOC],
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
            qualtran.bloqs.phase_estimation.qpe_window_state._CC_RECTANGULAR_WINDOW_STATE_DOC,
            qualtran.bloqs.phase_estimation.text_book_qpe._CC_TEXTBOOK_PHASE_ESTIMATION_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Kaiser Window State for Quantum Phase Estimation',
        module=qualtran.bloqs.phase_estimation.kaiser_window_state,
        bloq_specs=[
            qualtran.bloqs.phase_estimation.kaiser_window_state._CC_KAISER_WINDOW_STATE_DOC
        ],
    ),
    NotebookSpecV2(
        title='Qubitization Walk Operator',
        module=qualtran.bloqs.qubitization.qubitization_walk_operator,
        bloq_specs=[qualtran.bloqs.qubitization.qubitization_walk_operator._QUBITIZATION_WALK_DOC],
    ),
    NotebookSpecV2(
        title='Qubitization Phase Estimation',
        module=qualtran.bloqs.phase_estimation.qubitization_qpe,
        bloq_specs=[qualtran.bloqs.phase_estimation.qubitization_qpe._QUBITIZATION_QPE_DOC],
    ),
]

# --------------------------------------------------------------------------
# -----   Block Encoding   ----------------------------------------------------------
# --------------------------------------------------------------------------
BLOCK_ENCODING: List[NotebookSpecV2] = [
    NotebookSpecV2(
        title='Block Encoding Interface',
        module=qualtran.bloqs.block_encoding,
        bloq_specs=[qualtran.bloqs.block_encoding.block_encoding_base._BLOCK_ENCODING_DOC],
    ),
    NotebookSpecV2(
        title='Unitary',
        module=qualtran.bloqs.block_encoding.unitary,
        bloq_specs=[qualtran.bloqs.block_encoding.unitary._UNITARY_DOC],
    ),
    NotebookSpecV2(
        title='Tensor Product',
        module=qualtran.bloqs.block_encoding.tensor_product,
        bloq_specs=[qualtran.bloqs.block_encoding.tensor_product._TENSOR_PRODUCT_DOC],
    ),
    NotebookSpecV2(
        title='Product',
        module=qualtran.bloqs.block_encoding.product,
        bloq_specs=[qualtran.bloqs.block_encoding.product._PRODUCT_DOC],
    ),
    NotebookSpecV2(
        title='Phase',
        module=qualtran.bloqs.block_encoding.phase,
        bloq_specs=[qualtran.bloqs.block_encoding.phase._PHASE_DOC],
    ),
    NotebookSpecV2(
        title='Linear Combination',
        module=qualtran.bloqs.block_encoding.linear_combination,
        bloq_specs=[qualtran.bloqs.block_encoding.linear_combination._LINEAR_COMBINATION_DOC],
    ),
    NotebookSpecV2(
        title='Sparse Matrix',
        module=qualtran.bloqs.block_encoding.sparse_matrix,
        bloq_specs=[qualtran.bloqs.block_encoding.sparse_matrix._SPARSE_MATRIX_DOC],
    ),
    NotebookSpecV2(
        title='Sparse Matrix Hermitian',
        module=qualtran.bloqs.block_encoding.sparse_matrix_hermitian,
        bloq_specs=[
            qualtran.bloqs.block_encoding.sparse_matrix_hermitian._SPARSE_MATRIX_HERMITIAN_DOC
        ],
    ),
    NotebookSpecV2(
        title='Chebyshev Polynomial',
        module=qualtran.bloqs.block_encoding.chebyshev_polynomial,
        bloq_specs=[
            qualtran.bloqs.block_encoding.chebyshev_polynomial._CHEBYSHEV_BLOQ_DOC,
            qualtran.bloqs.block_encoding.chebyshev_polynomial._SCALED_CHEBYSHEV_BLOQ_DOC,
        ],
    ),
    NotebookSpecV2(
        title='LCU Select/Prepare Oracles',
        module=qualtran.bloqs.block_encoding.lcu_block_encoding,
        bloq_specs=[
            qualtran.bloqs.block_encoding.lcu_block_encoding._SELECT_BLOCK_ENCODING_DOC,
            qualtran.bloqs.block_encoding.lcu_block_encoding._LCU_BLOCK_ENCODING_DOC,
            qualtran.bloqs.multiplexers.select_base._SELECT_ORACLE_DOC,
            qualtran.bloqs.state_preparation.prepare_base._PREPARE_ORACLE_DOC,
            qualtran.bloqs.multiplexers.black_box_select._BLACK_BOX_SELECT_DOC,
            qualtran.bloqs.state_preparation.black_box_prepare._BLACK_BOX_PREPARE_DOC,
        ],
    ),
]

# --------------------------------------------------------------------------
# -----   Optimization   ---------------------------------------------------
# --------------------------------------------------------------------------
OPTIMIZATION: List[NotebookSpecV2] = [
    NotebookSpecV2(
        title='Planted Noisy kXOR - Kikuchi Guiding State',
        module=qualtran.bloqs.optimization.k_xor_sat.kikuchi_guiding_state,
        bloq_specs=[
            qualtran.bloqs.optimization.k_xor_sat.kikuchi_guiding_state._SIMPLE_GUIDING_STATE_DOC,
            qualtran.bloqs.optimization.k_xor_sat.kikuchi_guiding_state._GUIDING_STATE_DOC,
        ],
    )
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
        title='Qubitized Hubbard Model',
        module=qualtran.bloqs.chemistry.hubbard_model.qubitization,
        path_stem='hubbard_model',
        bloq_specs=[
            qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard._SELECT_HUBBARD,
            qualtran.bloqs.chemistry.hubbard_model.qubitization.prepare_hubbard._PREPARE_HUBBARD,
        ],
    ),
    NotebookSpecV2(
        title='Apply to Lth Target',
        module=qualtran.bloqs.multiplexers.apply_gate_to_lth_target,
        bloq_specs=[qualtran.bloqs.multiplexers.apply_gate_to_lth_target._APPLY_TO_LTH_TARGET_DOC],
    ),
    NotebookSpecV2(
        title='Apply Lth Bloq',
        module=qualtran.bloqs.multiplexers.apply_lth_bloq,
        bloq_specs=[qualtran.bloqs.multiplexers.apply_lth_bloq._APPLY_LTH_BLOQ_DOC],
    ),
    NotebookSpecV2(
        title='QROM',
        module=qualtran.bloqs.data_loading.qrom,
        bloq_specs=[
            qualtran.bloqs.data_loading.qrom_base._QROM_BASE_DOC,
            qualtran.bloqs.data_loading.qrom._QROM_DOC,
        ],
    ),
    NotebookSpecV2(
        title='SelectSwapQROM',
        module=qualtran.bloqs.data_loading.select_swap_qrom,
        bloq_specs=[
            qualtran.bloqs.data_loading.qrom_base._QROM_BASE_DOC,
            qualtran.bloqs.data_loading.select_swap_qrom._SELECT_SWAP_QROM_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Advanced QROM (aka QROAM) using clean ancilla',
        module=qualtran.bloqs.data_loading.qroam_clean,
        bloq_specs=[
            qualtran.bloqs.data_loading.qrom_base._QROM_BASE_DOC,
            qualtran.bloqs.data_loading.qroam_clean._QROAM_CLEAN_DOC,
        ],
    ),
    NotebookSpecV2(
        title='Reflections',
        module=qualtran.bloqs.reflections,
        bloq_specs=[
            qualtran.bloqs.reflections.prepare_identity._PREPARE_IDENTITY_DOC,
            qualtran.bloqs.reflections.reflection_using_prepare._REFL_USING_PREP_DOC,
        ],
        directory=f'{SOURCE_DIR}/bloqs/reflections',
    ),
    NotebookSpecV2(
        title='Multi-Paulis',
        module=qualtran.bloqs.mcmt,
        bloq_specs=[
            qualtran.bloqs.mcmt.multi_target_cnot._C_MULTI_NOT_DOC,
            qualtran.bloqs.mcmt.multi_control_pauli._CC_PAULI_DOC,
        ],
        directory=f'{SOURCE_DIR}/bloqs/mcmt/',
        path_stem='multi_control_multi_target_pauli',
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
            qualtran.bloqs.state_preparation.state_preparation_alias_sampling._STATE_PREP_ALIAS_DOC,
            qualtran.bloqs.state_preparation.state_preparation_alias_sampling._SPARSE_STATE_PREP_ALIAS_DOC,
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
    NotebookSpecV2(
        title='Generalized Quantum Signal Processing',
        module=qualtran.bloqs.qsp.generalized_qsp,
        bloq_specs=[qualtran.bloqs.qsp.generalized_qsp._Generalized_QSP_DOC],
    ),
    NotebookSpecV2(
        title='Hamiltonian Simulation by Generalized Quantum Signal Processing',
        module=qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp,
        bloq_specs=[
            qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp._Hamiltonian_Simulation_by_GQSP_DOC
        ],
    ),
]

NB_BY_SECTION = [
    ('Basic Gates', BASIC_GATES),
    ('Chemistry', CHEMISTRY),
    ('Arithmetic', ARITHMETIC),
    ('Modular Arithmetic', MOD_ARITHMETIC),
    ('GF Arithmetic', GF_ARITHMETIC),
    ('Polynomials over Galois Fields', GF_POLY_ARITHMETIC),
    ('Rotations', ROT_QFT_PE),
    ('Block Encoding', BLOCK_ENCODING),
    ('Optimization', OPTIMIZATION),
    ('Other', OTHER),
]
