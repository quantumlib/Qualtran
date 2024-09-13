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

import pytest

import qualtran.testing as qlt_testing
from qualtran import BloqExample


def assert_bloq_example_make_for_pytest(bloq_ex: BloqExample):
    """Wrap `assert_bloq_example_make`.

    Anything other than PASS is a test failure.
    """
    try:
        qlt_testing.assert_bloq_example_make(bloq_ex)
    except qlt_testing.BloqCheckException as bce:
        # No special skip logic
        raise bce from bce


def assert_bloq_example_decompose_for_pytest(bloq_ex: BloqExample):
    """Wrap `assert_bloq_example_decompose`.

    `NA` or `MISSING` result in the test being skipped.
    """
    try:
        qlt_testing.assert_bloq_example_decompose(bloq_ex)
    except qlt_testing.BloqCheckException as bce:
        if bce.check_result is qlt_testing.BloqCheckResult.NA:
            pytest.skip(bce.msg)
        if bce.check_result is qlt_testing.BloqCheckResult.MISSING:
            pytest.skip(bce.msg)

        raise bce from bce


def assert_equivalent_bloq_example_counts_for_pytest(bloq_ex: BloqExample):
    try:
        qlt_testing.assert_equivalent_bloq_example_counts(bloq_ex)
    except qlt_testing.BloqCheckException as bce:
        if bce.check_result in [
            qlt_testing.BloqCheckResult.UNVERIFIED,
            qlt_testing.BloqCheckResult.NA,
            qlt_testing.BloqCheckResult.MISSING,
        ]:
            pytest.skip(bce.msg)

        if bce.check_result == qlt_testing.BloqCheckResult.FAIL:
            pytest.xfail("We are not yet enforcing the 'counts' check.")

        raise bce from bce


def assert_bloq_example_serializes_for_pytest(bloq_ex: BloqExample):
    if bloq_ex.name in [
        'prep_sparse',
        'thc_prep',
        'modexp',
        'apply_z_to_odd',
        'select_pauli_lcu',
        'sel_hubb',
        'walk_op',
        'thc_walk_op',  # thc_prep does not serialize
        'qubitization_qpe_chem_thc',  # too slow
        'walk_op_chem_sparse',
        'qubitization_qpe_sparse_chem',  # too slow
        'trott_unitary',
        'symbolic_hamsim_by_gqsp',
        'gqsp_1d_ising',
        'auto_partition',
        'unitary_block_encoding',
        'unitary_block_encoding_properties',
        'tensor_product_block_encoding',
        'tensor_product_block_encoding_properties',
        'tensor_product_block_encoding_symb',
        'product_block_encoding',
        'product_block_encoding_properties',
        'product_block_encoding_symb',
        'apply_lth_bloq',
        'linear_combination_block_encoding',
        'phase_block_encoding',
        'state_prep_alias_symb',  # cannot serialize Shaped
        'sparse_matrix_block_encoding',
        'sparse_matrix_symb_block_encoding',
        'sparse_state_prep_alias_symb',  # cannot serialize Shaped
        'sparse_permutation',  # contains nested tuple of inhomogeneous shape
        'permutation_cycle_symb',  # cannot serialize Shaped
        'permutation_cycle_symb_N',  # sympy variable assumptions dropped by serialized
        'permutation_symb',  # cannot serialize shaped
        'permutation_symb_with_cycles',  # Object arrays cannot be saved when allow_pickle=False
        'sparse_permutation_with_symbolic_N',  # setting an array element with a sequence.
        'state_prep_via_rotation_symb',  # cannot serialize HasLength
        'state_prep_via_rotation_symb_phasegrad',  # cannot serialize Shaped
        'sparse_state_prep_via_rotations',  # cannot serialize Permutation
        'explicit_matrix_block_encoding',  # cannot serialize AutoPartition
        'symmetric_banded_matrix_block_encoding',  # cannot serialize AutoPartition
        'chebyshev_poly_even',
        'scaled_chebyshev_poly_even',
        'scaled_chebyshev_poly_odd',
        'black_box_select',  # cannot serialize AutoPartition
        'black_box_prepare',  # cannot serialize AutoPartition
        'kaiser_window_state_symbolic',  # Split cannot have a symbolic data type.
    ]:
        pytest.xfail("Skipping serialization test for bloq examples that cannot yet be serialized.")

    if bloq_ex.name in [
        'ecc',
        'ec_pe',
        'ec_pe_small',
        'ec_add_r',
        'ec_add_r_small',
        'ec_window_add',
        'ec_add',
    ]:
        pytest.xfail("Skipping serialization test for bloqs that use ECPoint.")

    try:
        qlt_testing.assert_bloq_example_serializes(bloq_ex)
    except qlt_testing.BloqCheckException as bce:
        raise bce from bce


def assert_bloq_example_qtyping_for_pytest(bloq_ex: BloqExample):
    try:
        qlt_testing.assert_bloq_example_qtyping(bloq_ex)
    except qlt_testing.BloqCheckException as bce:
        if bce.check_result is qlt_testing.BloqCheckResult.NA:
            pytest.skip(bce.msg)
        if bce.check_result is qlt_testing.BloqCheckResult.UNVERIFIED:
            pytest.skip(bce.msg)


_TESTFUNCS = [
    ('make', assert_bloq_example_make_for_pytest),
    ('decompose', assert_bloq_example_decompose_for_pytest),
    ('counts', assert_equivalent_bloq_example_counts_for_pytest),
    ('serialize', assert_bloq_example_serializes_for_pytest),
    ('qtyping', assert_bloq_example_qtyping_for_pytest),
]


@pytest.fixture(scope="module", params=_TESTFUNCS, ids=[name for name, func in _TESTFUNCS])
def bloq_autotester(request):
    name, func = request.param
    func.check_name = name
    return func
