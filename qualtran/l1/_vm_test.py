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
"""Tests for the architecture-agnostic virtual machine in `_vm.py`."""

import numpy as np
import pytest

from qualtran import BloqBuilder, QAny, Register
from qualtran.bloqs.basic_gates import CNOT, CSwap, TGate, XGate
from qualtran.bloqs.bookkeeping import Split
from qualtran.bloqs.mcmt import MultiAnd
from qualtran.l1._vm import (
    _mock_return_values,
    Problem,
    StackFrame,
    StandardQualtranArchitectureAgnosticVirtualMachine,
    UnsupportedAtomicBloqProblem,
)


def _two_tgates_cbloq():
    """A composite bloq that applies two T gates (both ISA atoms) to one qubit."""
    bb = BloqBuilder()
    q = bb.add_register('q', 1)
    q = bb.add(TGate(), q=q)
    q = bb.add(TGate(), q=q)
    return bb.finalize(q=q)


# ---------------------------------------------------------------------------
# _mock_return_values
# ---------------------------------------------------------------------------


def test_mock_return_values_scalar_register():
    soqs = _mock_return_values([Register('q', QAny(2))])
    assert set(soqs) == {'q'}
    # A shape-less register maps to a single opaque object.
    assert not isinstance(soqs['q'], np.ndarray)


def test_mock_return_values_shaped_register():
    soqs = _mock_return_values([Register('q', QAny(1), shape=(2, 3))])
    arr = soqs['q']
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 3)
    assert arr.dtype == object


# ---------------------------------------------------------------------------
# StackFrame / Problem
# ---------------------------------------------------------------------------


def test_stack_frame_defaults():
    frame = StackFrame(bloq_str='Foo')
    assert frame.bloq_str == 'Foo'
    assert frame.qlocals == {}


def test_unsupported_atomic_problem_summary():
    problem = UnsupportedAtomicBloqProblem(MultiAnd(cvs=(1, 1, 1)))
    assert isinstance(problem, Problem)
    summary = problem.get_summary()
    assert 'not a supported atomic bloq' in summary
    assert 'MultiAnd' in summary


# ---------------------------------------------------------------------------
# bloq_in_isa
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('bloq', [TGate(), CNOT(), XGate(), Split(QAny(3))])
def test_bloq_in_isa_true(bloq):
    vm = StandardQualtranArchitectureAgnosticVirtualMachine()
    assert vm.bloq_in_isa(bloq) is True


@pytest.mark.parametrize('bloq', [MultiAnd(cvs=(1, 1, 1))])
def test_bloq_in_isa_false(bloq):
    vm = StandardQualtranArchitectureAgnosticVirtualMachine()
    assert vm.bloq_in_isa(bloq) is False


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


def test_execute_isa_atom_counts_atom():
    vm = StandardQualtranArchitectureAgnosticVirtualMachine()
    vm.execute(TGate())
    assert vm.n_atoms == 1
    assert vm.n_calls == 0
    assert vm.problems == []


def test_execute_unsupported_atom_records_problem():
    vm = StandardQualtranArchitectureAgnosticVirtualMachine()
    result = vm.execute(MultiAnd(cvs=(1, 1, 1)))
    assert result == {}
    assert vm.n_atoms == 0
    assert len(vm.problems) == 1
    assert isinstance(vm.problems[0], UnsupportedAtomicBloqProblem)


def test_execute_non_bloq_raises():
    vm = StandardQualtranArchitectureAgnosticVirtualMachine()
    with pytest.raises(TypeError, match='Unexpected `execute` type'):
        vm.execute(42)  # type: ignore[arg-type]


def test_execute_composite_recurses():
    vm = StandardQualtranArchitectureAgnosticVirtualMachine()
    vm.execute(_two_tgates_cbloq())
    assert vm.n_calls == 1
    assert vm.n_atoms == 2
    assert vm.problems == []


# ---------------------------------------------------------------------------
# qcall
# ---------------------------------------------------------------------------


def test_qcall_manages_frames():
    vm = StandardQualtranArchitectureAgnosticVirtualMachine()
    vm.qcall(_two_tgates_cbloq())
    # The frame stack is balanced (pushed then popped).
    assert vm.frames == []
    assert vm.n_calls == 1
    assert vm.n_atoms == 2


def test_qcall_nested_composite():
    inner = _two_tgates_cbloq()
    bb = BloqBuilder()
    q = bb.add_register('q', 1)
    q = bb.add(inner, q=q)
    q = bb.add(TGate(), q=q)
    outer = bb.finalize(q=q)

    vm = StandardQualtranArchitectureAgnosticVirtualMachine()
    vm.execute(outer)
    # Two qcalls (outer + inner), three T gates total.
    assert vm.n_calls == 2
    assert vm.n_atoms == 3
    assert vm.problems == []


def test_execute_composite_with_unsupported_subbloq():
    # CSwap is neither an ISA atom nor a CompositeBloq, so walking a composite
    # that contains it should record a problem (rather than recursing or
    # counting it as an atom).
    bb = BloqBuilder()
    ctrl = bb.add_register('ctrl', 1)
    x = bb.add_register('x', 2)
    y = bb.add_register('y', 2)
    ctrl, x, y = bb.add(CSwap(bitsize=2), ctrl=ctrl, x=x, y=y)
    cbloq = bb.finalize(ctrl=ctrl, x=x, y=y)

    vm = StandardQualtranArchitectureAgnosticVirtualMachine()
    vm.execute(cbloq)
    assert vm.n_calls == 1
    assert vm.n_atoms == 0
    assert len(vm.problems) == 1
    assert isinstance(vm.problems[0], UnsupportedAtomicBloqProblem)
    assert vm.problems[0].bloq == CSwap(bitsize=2)


# ---------------------------------------------------------------------------
# print_execution_summary
# ---------------------------------------------------------------------------


def test_print_execution_summary_success(capsys):
    vm = StandardQualtranArchitectureAgnosticVirtualMachine()
    vm.execute(_two_tgates_cbloq())
    vm.print_execution_summary()
    out = capsys.readouterr().out
    assert '2 ISA operations' in out
    assert '1 subroutine calls' in out
    assert 'Execution completed successfully.' in out


def test_print_execution_summary_with_problems(capsys):
    vm = StandardQualtranArchitectureAgnosticVirtualMachine()
    vm.execute(MultiAnd(cvs=(1, 1, 1)))
    vm.execute(MultiAnd(cvs=(1, 1, 1)))
    vm.print_execution_summary()
    out = capsys.readouterr().out
    assert 'the following problems were encountered' in out
    # The two identical problems are aggregated with a 2x count.
    assert '2x' in out
    assert 'not a supported atomic bloq' in out
