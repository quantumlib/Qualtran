#  Copyright 2026 Google LLC
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
import sympy

from qualtran import QInt
from qualtran.bloqs.arithmetic import Add
from qualtran.bloqs.basic_gates import CNOT
from qualtran.l1 import build_library_entry, BuildOutcome, library_qlt_path


def test_build_outcome_is_hard_failure():
    assert not BuildOutcome.SUCCESS.is_hard_failure
    assert not BuildOutcome.EXECUTION_WITH_PROBLEMS.is_hard_failure
    assert not BuildOutcome.SKIPPED.is_hard_failure
    for outcome in [
        BuildOutcome.CONSTRUCT_FAILED,
        BuildOutcome.COMPILE_FAILED,
        BuildOutcome.RELOAD_FAILED,
        BuildOutcome.EXECUTION_CATASTROPHIC,
        BuildOutcome.TIMEOUT,
    ]:
        assert outcome.is_hard_failure


def test_library_qlt_path_from_class_and_instance(tmp_path):
    expected = tmp_path / 'lib' / 'qualtran' / 'bloqs' / 'basic_gates' / 'CNOT' / 'cnot.qlt'
    # `_class_name_in_pkg_` is a classmethod, so a class or an instance both work.
    assert library_qlt_path(CNOT, 'cnot', tmp_path, subdir='lib') == expected
    assert library_qlt_path(CNOT(), 'cnot', tmp_path, subdir='lib') == expected

    partial = tmp_path / 'partial' / 'qualtran' / 'bloqs' / 'basic_gates' / 'CNOT' / 'cnot.qlt'
    assert library_qlt_path(CNOT(), 'cnot', tmp_path, subdir='partial') == partial


def test_build_library_entry_success_lands_in_lib(tmp_path):
    result = build_library_entry(CNOT(), 'cnot', tmp_path)

    assert result.outcome is BuildOutcome.SUCCESS
    assert result.n_problems == 0
    assert result.n_atoms == 1

    lib_file = tmp_path / 'lib' / 'qualtran' / 'bloqs' / 'basic_gates' / 'CNOT' / 'cnot.qlt'
    assert result.qlt_path == str(lib_file)
    assert lib_file.exists()

    # A success leaves nothing under partial/.
    assert not list(tmp_path.glob('partial/**/*.qlt'))


def test_build_library_entry_reuse_is_idempotent(tmp_path):
    first = build_library_entry(CNOT(), 'cnot', tmp_path)
    # Second run finds the lib/ file and reuses it (no recompile) but still
    # reloads + executes, yielding the same successful outcome and path.
    second = build_library_entry(CNOT(), 'cnot', tmp_path)

    assert second.outcome is BuildOutcome.SUCCESS
    assert second.qlt_path == first.qlt_path

    # `--regenerate` recompiles but still succeeds.
    regenerated = build_library_entry(CNOT(), 'cnot', tmp_path, regenerate=True)
    assert regenerated.outcome is BuildOutcome.SUCCESS
    assert regenerated.qlt_path == first.qlt_path


def test_build_library_entry_soft_failure_stays_in_partial(tmp_path):
    # A symbolic `Add` compiles and reloads fine, but the VM cannot execute it
    # (it reaches the VM as an unsupported atomic bloq), so it is a soft failure.
    n = sympy.Symbol('n')
    result = build_library_entry(Add(QInt(n)), 'add_symb', tmp_path)

    assert result.outcome is BuildOutcome.EXECUTION_WITH_PROBLEMS
    assert not result.outcome.is_hard_failure
    assert result.n_problems and result.n_problems >= 1

    partial_file = (
        tmp_path / 'partial' / 'qualtran' / 'bloqs' / 'arithmetic' / 'Add' / 'add_symb.qlt'
    )
    assert result.qlt_path == str(partial_file)
    assert partial_file.exists()

    # A soft failure never lands in lib/.
    assert not list(tmp_path.glob('lib/**/*.qlt'))


def test_symbolic_name_pattern():
    import importlib.util
    import sys
    from pathlib import Path

    dev_tools_dir = str(Path(__file__).parents[2] / 'dev_tools')
    if dev_tools_dir not in sys.path:
        sys.path.insert(0, dev_tools_dir)

    spec = importlib.util.spec_from_file_location(
        "build_l1_library", "dev_tools/build-l1-library.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    SYMBOLIC_NAME_PATTERN = mod.SYMBOLIC_NAME_PATTERN

    should_match = [
        'bloq_ex_symb',
        'bloq_ex_symb_small',
        'bloq_ex_symbolic',
        'symbolic_qft',
        'symb_bloq_ex',
    ]
    should_not_match = ['bloq_ex_symbiotic', 'bloq_symbiotic_ex', 'asymb_bloq', 'asymbolic_bloq']

    for name in should_match:
        assert SYMBOLIC_NAME_PATTERN.search(name), f'Expected match for {name}'
    for name in should_not_match:
        assert not SYMBOLIC_NAME_PATTERN.search(name), f'Expected no match for {name}'
