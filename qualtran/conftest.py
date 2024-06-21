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
        'tensor_product_block_encoding',
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
