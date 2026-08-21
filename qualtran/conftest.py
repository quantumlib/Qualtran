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

import os

import pytest

import qualtran
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
    pytest.importorskip('google.protobuf')
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
        'qubitization_qpe_ising',
        'trott_unitary',
        'symbolic_hamsim_by_gqsp',
        'gf16_addition',  # cannot serialize QGF
        'gf2_addition_symbolic',  # cannot serialize QGF
        'gf16_add_k',  # cannot serialize QGF
        'gf2_add_k_symbolic',  # cannot serialize QGF
        'gf16_multiplication',  # cannot serialize QGF
        'gf2_multiplication_symbolic',  # cannot serialize QGF
        'gf16_square',  # cannot serialize QGF
        'gf2_square_symbolic',  # cannot serialize QGF
        'gf16_inverse',  # cannot serialize QGF
        'gf2_inverse_symbolic',  # cannot serialize QGF
        'gf_poly_split',  # cannot serialize QGF and QGFPoly
        'gf_poly_join',  # cannot serialize QGF and QGFPoly
        'gf2_poly_4_8_add_k',  # cannot serialize QGF and QGFPoly
        'gf2_poly_add_k_symbolic',  # cannot serialize QGF and QGFPoly
        'gf2_poly_4_8_add',  # cannot serialize QGF and QGFPoly
        'gf2_poly_add_symbolic',  # cannot serialize QGF and QGFPoly
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
        'sparse_matrix_hermitian_block_encoding',
        'sparse_matrix_symb_hermitian_block_encoding',
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
        'sparse_state_prep_via_rotations_with_large_target_bitsize',  # setting an array element with a sequence.
        'explicit_matrix_block_encoding',  # cannot serialize AutoPartition
        'symmetric_banded_matrix_block_encoding',  # cannot serialize AutoPartition
        'chebyshev_poly_even',
        'scaled_chebyshev_poly_even',
        'scaled_chebyshev_poly_odd',
        'black_box_select',  # cannot serialize AutoPartition
        'black_box_prepare',  # cannot serialize AutoPartition
        'kaiser_window_state_symbolic',  # Split cannot have a symbolic data type.
        'ctrl_on_symbolic_cv',  # cannot serialize Shaped
        'ctrl_on_symbolic_cv_multi',  # cannot serialize Shaped
        'ctrl_on_symbolic_n_ctrls',  # cannot serialize Shaped
        'has_duplicates_symb_len',  # cannot serialize HasLength
        'symm_diff_symb',  # round trip fail: sympy assumptions not serialized
        'symm_diff_equal_size_symb',  # round trip fail: sympy assumptions not serialized
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

    if bloq_ex.name in [
        'col_kth_nz',
        'col_kth_nz_symb',
        'kikuchi_nonzero_index',
        'kikuchi_nonzero_index_symb',
        'simple_guiding_state',
        'simple_guiding_state_symb',
        'guiding_state',
        'guiding_state_symb',
        'guiding_state_symb_c',
        'kikuchi_matrix_entry',
        'kikuchi_matrix_entry_symb',
        'kikuchi_matrix',
        'kikuchi_matrix_symb',
        'load_scopes',
        'load_scopes_symb',
        'guided_phase_estimate_symb',
        'guided_hamiltonian_symb',
        'solve_planted',
        'solve_planted_symbolic',
    ]:
        pytest.xfail("Skipping serialization test for bloqs that use KXorInstance.")

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


@pytest.fixture(autouse=True)
def add_qlt(doctest_namespace):
    # Make qualtran available (without explicit import) in doctests
    doctest_namespace['qualtran'] = qualtran


def get_available_cpu_count() -> int:
    """Returns the number of CPU cores available to the current process.

    This function respects active CPU limits such as process affinity and
    container limits.
    """
    if hasattr(os, "process_cpu_count"):  # Python 3.13+
        cpus = os.process_cpu_count() or 1
    elif hasattr(os, "sched_getaffinity"):  # Unix/Linux
        try:
            cpus = len(os.sched_getaffinity(0))
        except OSError:
            cpus = os.cpu_count() or 1
    else:  # Fallback for older Python on Windows/macOS
        cpus = os.cpu_count() or 1
    return cpus


def _config_set_xdist_worksteal(config) -> None:
    """Sets `--dist worksteal` as the default distribution mode if not
    explicitly overridden by the user."""
    num_workers = config.getoption("numprocesses")
    if num_workers in (None, 0, 1, "0", "1"):
        return

    if (
        hasattr(config, "option")
        and hasattr(config.option, "dist")
        and config.getoption("dist") in (None, "no", "load")
    ):
        # Check if the user explicitly provided a distribution option. If they
        # did, we shouldn't overwrite it. Since dist defaults to "load" when
        # -n is set, we check if --dist is explicitly passed.
        args = []
        if hasattr(config, "invocation_params") and config.invocation_params is not None:
            args.extend(config.invocation_params.args)
        try:
            addopts = config.getini("addopts")
            if isinstance(addopts, list):
                args.extend(addopts)
        except (ValueError, AttributeError):
            pass

        for arg in args:
            if arg.startswith("--dist") or arg == "-d":
                break
        else:
            # Checked all args and didn't find --dist or -d.
            config.option.dist = "worksteal"


def _config_set_thread_limits(config) -> None:
    """Limit number of threads to prevent oversubscription with pytest-xdist.

    This only influences parallelism in some core numerical libraries used in
    packages such as NumPy by setting certain environment variables. When
    pytest runs as many workers as CPUs, limiting the number of threads used by
    the libraries greatly improves overall test performance. Without the limit,
    numerical operations in some tests spawn as many parallel threads as CPUs,
    overwhelming host resources when pytest runs the tests in parallel.
    """
    num_workers = config.getoption("numprocesses")
    if num_workers is None or not isinstance(num_workers, (int, str)):
        num_workers = 1

    num_cpus = get_available_cpu_count()
    if isinstance(num_workers, str):
        if num_workers in ("auto", "logical"):
            num_workers = num_cpus
        else:
            try:
                num_workers = int(num_workers)
            except ValueError:
                num_workers = 1

    # Cap the number of threads when using multiple workers.
    if num_workers > 1 and num_cpus > 0:
        limit = max(1, num_cpus // num_workers)
        env_vars = [
            "MKL_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ]
        for var in env_vars:
            os.environ[var] = str(limit)


def pytest_configure(config):
    """Configure pytest environment settings, especially for pytest-xdist."""

    # Only run in the controlling process, before workers are started.
    if hasattr(config, "workerinput"):
        return
    try:
        config.getoption("numprocesses")
    except ValueError:
        # pytest-xdist is not being used.
        return

    _config_set_thread_limits(config)
    _config_set_xdist_worksteal(config)
