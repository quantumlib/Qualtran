#  Copyright 2025 Google LLC
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

"""Tests for qualtran.simulation.verification."""

from __future__ import annotations

import io
from functools import cached_property

import attrs
import numpy as np
import pytest

from qualtran import (
    Bloq,
    BloqBuilder,
    DecomposeNotImplementedError,
    DecomposeTypeError,
    QBit,
    QUInt,
    Register,
    Side,
    Signature,
    SoquetT,
)
from qualtran.bloqs.basic_gates import CNOT, XGate
from qualtran.simulation.classical_sim import QCDTypeDomainError
from qualtran.simulation.verification import (
    _assert_results_equal,
    _should_use_fastsim,
    ClassicalSimTestCase,
    generate_input_vals,
    MissingClassicalValsError,
    print_verification_summary,
    simulate_via_subbloq_classical_vals,
    VerificationResult,
    VerificationStatus,
    verify_structural_induction,
)

# ===========================================================================
# Test fixtures: small custom bloqs with controlled behaviour
# ===========================================================================


@pytest.fixture(autouse=True)
def default_python_verification_engine(monkeypatch, request):
    """By default, run verification tests with Python engine (`QLT_USE_FASTSIM=0`).

    Mock verification bloqs (`_AtomicNot`, `_DomainConstrainedBloq`, etc.) are not
    supported by the `rsqualtran` compiled L1 engine. Tests that explicitly exercise
    the Rust fastsim engine bypass this via `qlt_use_fastsim` parameterization or fixture.
    """
    if 'qlt_use_fastsim' not in request.fixturenames:
        monkeypatch.setenv('QLT_USE_FASTSIM', '0')


@pytest.fixture
def qlt_use_fastsim(request, monkeypatch):
    monkeypatch.setenv('QLT_USE_FASTSIM', request.param)
    return request.param


#
# These are organised by role:
#
# 1. Basic building blocks (identity, NOT, double-NOT variants)
# 2. Atomic leaf bloqs (raise DecomposeTypeError — can't decompose further)
# 3. Composite bloqs for structural-induction tests (diamonds, 3-level trees)
# 4. Domain-constrained / edge-case bloqs
# 5. Shaped- and large-register bloqs for input-generation tests


# --- 1. Basic building blocks ------------------------------------------------


@attrs.frozen
class _IdentityBloq(Bloq):
    """Single-qubit identity: has on_classical_vals, no decomposition."""

    @cached_property
    def signature(self):
        return Signature([Register('q', QBit())])

    def on_classical_vals(self, q):
        return {'q': q}


@attrs.frozen
class _NotGate(Bloq):
    """Single-qubit NOT: has on_classical_vals, no decomposition."""

    @cached_property
    def signature(self):
        return Signature([Register('q', QBit())])

    def on_classical_vals(self, q):
        return {'q': (q + 1) % 2}


@attrs.frozen
class _DoubleNot(Bloq):
    """Applies NOT twice (identity). Has both reference and decomposition."""

    @cached_property
    def signature(self):
        return Signature([Register('q', QBit())])

    def on_classical_vals(self, q):
        return {'q': q}

    def build_composite_bloq(self, bb: BloqBuilder, q: SoquetT) -> dict[str, SoquetT]:
        q = bb.add(XGate(), q=q)
        q = bb.add(XGate(), q=q)
        return {'q': q}


@attrs.frozen
class _BuggyDoubleNot(Bloq):
    """Correct decomposition (XX = identity), but WRONG on_classical_vals (NOT).

    A tautological verifier using call_classically() on both sides would
    silently fall back to decomposition for the parent too, comparing
    decomposition-vs-decomposition and never noticing the bug. The real
    verifier must call on_classical_vals DIRECTLY and catch the mismatch.
    """

    @cached_property
    def signature(self):
        return Signature([Register('q', QBit())])

    def on_classical_vals(self, q):
        # WRONG: claims to be NOT.
        return {'q': (q + 1) % 2}

    def build_composite_bloq(self, bb: BloqBuilder, q: SoquetT) -> dict[str, SoquetT]:
        # CORRECT: XX = identity.
        q = bb.add(XGate(), q=q)
        q = bb.add(XGate(), q=q)
        return {'q': q}


@attrs.frozen
class _PartiallyWrongBloq(Bloq):
    """on_classical_vals is CORRECT for input 0 but WRONG for input 1.

    A verifier that only tests input 0 (or stops after the first success)
    would miss the bug. The verifier must test both inputs for QBit
    (exhaustive enumeration) and detect the failure on input 1.
    """

    @cached_property
    def signature(self):
        return Signature([Register('q', QBit())])

    def on_classical_vals(self, q):
        if q == 0:
            return {'q': 0}  # CORRECT: identity on 0
        return {'q': 0}  # WRONG: should return 1 for identity

    def build_composite_bloq(self, bb: BloqBuilder, q: SoquetT) -> dict[str, SoquetT]:
        # Decomposition: identity (no gates).
        return {'q': q}


@attrs.frozen
class _NoClassicalValsBloq(Bloq):
    """Leaf bloq with no on_classical_vals and no decomposition."""

    @cached_property
    def signature(self):
        return Signature([Register('q', QBit())])

    # on_classical_vals not overridden → returns NotImplemented.


@attrs.frozen
class _NoReferenceBloq(Bloq):
    """Has a decomposition but a sub-bloq lacks on_classical_vals."""

    @cached_property
    def signature(self):
        return Signature([Register('q', QBit())])

    def build_composite_bloq(self, bb: BloqBuilder, q: SoquetT) -> dict[str, SoquetT]:
        q = bb.add(_NoClassicalValsBloq(), q=q)
        return {'q': q}


@attrs.frozen
class _DecomposableNoReference(Bloq):
    """Has a decomposition (into _DoubleNot) but no on_classical_vals.

    The parent-reference check must be skipped for this bloq. If the verifier
    accidentally used call_classically (which falls back to decomposition),
    it would be comparing the decomposition against itself — tautological.
    """

    @cached_property
    def signature(self):
        return Signature([Register('q', QBit())])

    # No on_classical_vals — returns NotImplemented by default.

    def build_composite_bloq(self, bb: BloqBuilder, q: SoquetT) -> dict[str, SoquetT]:
        q = bb.add(_DoubleNot(), q=q)
        return {'q': q}


# --- 2. Atomic leaf bloqs ----------------------------------------------------
# These raise DecomposeTypeError to signal they are primitive operations.


@attrs.frozen
class _AtomicNot(Bloq):
    """Atomic NOT gate (leaf node). Correct on_classical_vals."""

    @cached_property
    def signature(self):
        return Signature([Register('q', QBit())])

    def on_classical_vals(self, q):
        return {'q': (q + 1) % 2}

    def decompose_bloq(self):
        raise DecomposeTypeError(f"{self} is atomic")


@attrs.frozen
class _AtomicBuggyIdentity(Bloq):
    """Atomic leaf whose on_classical_vals is WRONG: claims identity
    instead of NOT.

    Used to inject bugs deep in decomposition trees. A parent that expects
    two NOTs (= identity) will get NOT + identity = NOT, causing a mismatch.
    """

    @cached_property
    def signature(self):
        return Signature([Register('q', QBit())])

    def on_classical_vals(self, q):
        # WRONG: claims identity instead of NOT.
        return {'q': q}

    def decompose_bloq(self):
        raise DecomposeTypeError(f"{self} is atomic")


# --- 3. Composite bloqs for structural-induction tests -----------------------


@attrs.frozen
class _ParentBloq(Bloq):
    """Decomposes into two _DoubleNot instances (shared child class)."""

    @cached_property
    def signature(self):
        return Signature([Register('x', QBit()), Register('y', QBit())])

    def on_classical_vals(self, x, y):
        return {'x': x, 'y': y}

    def build_composite_bloq(self, bb: BloqBuilder, x: SoquetT, y: SoquetT) -> dict[str, SoquetT]:
        x = bb.add(_DoubleNot(), q=x)
        y = bb.add(_DoubleNot(), q=y)
        return {'x': x, 'y': y}


@attrs.frozen
class _ParentWithBuggySubBloqs(Bloq):
    """Parent is correct (identity), decomposition uses two _AtomicBuggyIdentity.

    Each sub-bloq claims identity (wrong — should be NOT), so two
    "identities" = identity, matching the parent's reference. The parent
    passes, but _AtomicBuggyIdentity would fail its own leaf verification
    if it had a decomposition to cross-check against.

    This demonstrates that consistently-wrong sub-bloqs can mask each other.
    """

    @cached_property
    def signature(self):
        return Signature([Register('q', QBit())])

    def on_classical_vals(self, q):
        return {'q': q}

    def build_composite_bloq(self, bb: BloqBuilder, q: SoquetT) -> dict[str, SoquetT]:
        q = bb.add(_AtomicBuggyIdentity(), q=q)
        q = bb.add(_AtomicBuggyIdentity(), q=q)
        return {'q': q}


@attrs.frozen
class _ParentWithOneBuggySubBloq(Bloq):
    """Parent is correct (identity via NOT + NOT). Decomposition uses one
    correct _AtomicNot and one buggy _AtomicBuggyIdentity.

    simulate_via_subbloq_classical_vals computes: NOT(input) via _AtomicNot, then identity
    via _AtomicBuggyIdentity = NOT(input). Parent says identity = input.
    MISMATCH — sub-bloq bugs propagate to the parent.
    """

    @cached_property
    def signature(self):
        return Signature([Register('q', QBit())])

    def on_classical_vals(self, q):
        return {'q': q}

    def build_composite_bloq(self, bb: BloqBuilder, q: SoquetT) -> dict[str, SoquetT]:
        q = bb.add(_AtomicNot(), q=q)
        q = bb.add(_AtomicBuggyIdentity(), q=q)
        return {'q': q}


@attrs.frozen
class _DiamondMid(Bloq):
    """Middle node in a diamond-shaped DAG. Decomposes into one correct
    _AtomicNot and one buggy _AtomicBuggyIdentity.

    on_classical_vals: identity (two NOTs should cancel).
    simulate_via_subbloq_classical_vals: NOT + identity = NOT. MISMATCH.
    """

    @cached_property
    def signature(self):
        return Signature([Register('q', QBit())])

    def on_classical_vals(self, q):
        return {'q': q}

    def build_composite_bloq(self, bb: BloqBuilder, q: SoquetT) -> dict[str, SoquetT]:
        q = bb.add(_AtomicNot(), q=q)
        q = bb.add(_AtomicBuggyIdentity(), q=q)
        return {'q': q}


@attrs.frozen
class _DiamondTopBloq(Bloq):
    """Diamond DAG top node. Decomposes into two _DiamondMid instances
    on separate registers, creating a diamond-shaped dependency graph.

    BFS must discover _DiamondMid and its leaves. The bug in
    _AtomicBuggyIdentity causes _DiamondMid to fail.
    """

    @cached_property
    def signature(self):
        return Signature([Register('x', QBit()), Register('y', QBit())])

    def on_classical_vals(self, x, y):
        return {'x': x, 'y': y}

    def build_composite_bloq(self, bb: BloqBuilder, x: SoquetT, y: SoquetT) -> dict[str, SoquetT]:
        x = bb.add(_DiamondMid(), q=x)
        y = bb.add(_DiamondMid(), q=y)
        return {'x': x, 'y': y}


@attrs.frozen
class _Level1Mid(Bloq):
    """Level 1 in a 3-level hierarchy. Decomposes into _AtomicNot +
    _AtomicBuggyIdentity. on_classical_vals says identity (two NOTs
    should cancel) but the buggy leaf causes a mismatch.
    """

    @cached_property
    def signature(self):
        return Signature([Register('q', QBit())])

    def on_classical_vals(self, q):
        return {'q': q}

    def build_composite_bloq(self, bb: BloqBuilder, q: SoquetT) -> dict[str, SoquetT]:
        q = bb.add(_AtomicNot(), q=q)
        q = bb.add(_AtomicBuggyIdentity(), q=q)
        return {'q': q}


@attrs.frozen
class _Level2Top(Bloq):
    """3-level hierarchy top. Decomposes into two _Level1Mid instances.

    BFS must traverse 3 levels deep to discover the bug in
    _AtomicBuggyIdentity via _Level1Mid.
    """

    @cached_property
    def signature(self):
        return Signature([Register('q', QBit())])

    def on_classical_vals(self, q):
        return {'q': q}

    def build_composite_bloq(self, bb: BloqBuilder, q: SoquetT) -> dict[str, SoquetT]:
        q = bb.add(_Level1Mid(), q=q)
        q = bb.add(_Level1Mid(), q=q)
        return {'q': q}


# --- 4. Domain-constrained / edge-case bloqs ---------------------------------


@attrs.frozen
class _DomainConstrainedBloq(Bloq):
    """Raises ValueError on odd inputs. Only even inputs are checked.

    n_inputs_checked must be strictly less than total domain size.
    """

    n: int = 4

    @cached_property
    def signature(self):
        return Signature([Register('x', QUInt(self.n))])

    def on_classical_vals(self, x):
        if x % 2 != 0:
            raise QCDTypeDomainError(f"Only even inputs allowed, got {x}")
        return {'x': x}

    def build_composite_bloq(self, bb: BloqBuilder, x: SoquetT) -> dict[str, SoquetT]:
        return {'x': x}


@attrs.frozen
class _AllRaisingBloq(Bloq):
    """on_classical_vals raises on ALL inputs.

    The verifier must NOT report 'passed' when nothing was actually
    verified. Instead it should report 'no_valid_inputs'.
    """

    @cached_property
    def signature(self):
        return Signature([Register('q', QBit())])

    def on_classical_vals(self, q):
        raise QCDTypeDomainError("No valid inputs")

    def build_composite_bloq(self, bb: BloqBuilder, q: SoquetT) -> dict[str, SoquetT]:
        q = bb.add(XGate(), q=q)
        return {'q': q}


@attrs.frozen
class _TypeErrorInOnClassicalVals(Bloq):
    """on_classical_vals has a genuine bug: calls len() on an int.

    This is NOT a domain constraint — it's a programming error. The verifier
    must NOT silently skip it as a "domain-constrained" input. It should
    propagate as an ERROR.
    """

    @cached_property
    def signature(self):
        return Signature([Register('q', QBit())])

    def on_classical_vals(self, q):
        return {'q': len(q)}  # Bug: q is an int, not a sequence

    def build_composite_bloq(self, bb: BloqBuilder, q: SoquetT) -> dict[str, SoquetT]:
        return {'q': q}


@attrs.frozen
class _KeyErrorInOnClassicalVals(Bloq):
    """on_classical_vals has a genuine bug: accesses a nonexistent dict key.

    This is NOT a domain constraint — it's a programming error. The verifier
    must NOT silently skip it.
    """

    @cached_property
    def signature(self):
        return Signature([Register('q', QBit())])

    def on_classical_vals(self, q):
        d: dict[str, int] = {}
        return {'q': d['missing_key']}  # Bug: KeyError

    def build_composite_bloq(self, bb: BloqBuilder, q: SoquetT) -> dict[str, SoquetT]:
        return {'q': q}


@attrs.frozen
class _TypeErrorLeafBloq(Bloq):
    """Atomic leaf bloq with a genuine TypeError bug in on_classical_vals.

    The verifier must NOT mark this LEAF_VERIFIED — the exception is not a
    domain constraint, it's a real bug.
    """

    @cached_property
    def signature(self):
        return Signature([Register('q', QBit())])

    def on_classical_vals(self, q):
        return {'q': len(q)}  # Bug: q is an int

    def decompose_bloq(self):
        raise DecomposeTypeError(f"{self} is atomic")


@attrs.frozen
class _ParentOfBuggyLeaf(Bloq):
    """Parent that decomposes into _TypeErrorLeafBloq.

    When the verifier runs simulate_via_subbloq_classical_vals on this parent, it invokes
    _TypeErrorLeafBloq.on_classical_vals, which raises TypeError. With the
    narrowed except clauses, this TypeError must propagate as ERROR rather
    than being silently swallowed.
    """

    @cached_property
    def signature(self):
        return Signature([Register('q', QBit())])

    def on_classical_vals(self, q):
        return {'q': q}  # Correct identity

    def build_composite_bloq(self, bb: BloqBuilder, q: SoquetT) -> dict[str, SoquetT]:
        q = bb.add(_TypeErrorLeafBloq(), q=q)
        return {'q': q}


# --- 5. Shaped- and large-register bloqs ------------------------------------


@attrs.frozen
class _LargeDomainBloq(Bloq):
    """Uses QUInt(16) — domain size 65,536. Forces random sampling
    when n_samples is small.
    """

    @cached_property
    def signature(self):
        return Signature([Register('x', QUInt(16))])

    def on_classical_vals(self, x):
        return {'x': x}


@attrs.frozen
class _ShapedRegBloq(Bloq):
    """Has a shaped register — shape=(5,) of QBit."""

    @cached_property
    def signature(self):
        return Signature([Register('arr', QBit(), shape=(5,))])

    def on_classical_vals(self, arr):
        return {'arr': arr}


@attrs.frozen
class _MixedRegBloq(Bloq):
    """Has both scalar (QBit) and shaped (QBit, shape=(3,)) registers."""

    @cached_property
    def signature(self):
        return Signature([Register('scalar', QBit()), Register('array', QBit(), shape=(3,))])

    def on_classical_vals(self, scalar, array):
        return {'scalar': scalar, 'array': array}


# ===========================================================================
# Tests: StrictClassicalSimState
# ===========================================================================


def test_reference_sim_raises_on_missing_on_classical_vals():
    """StrictClassicalSimState raises MissingClassicalValsError when
    on_classical_vals is not implemented on a sub-bloq."""
    bloq = _NoReferenceBloq()
    with pytest.raises(MissingClassicalValsError):
        simulate_via_subbloq_classical_vals(bloq, q=0)


def test_reference_sim_succeeds_with_reference():
    """StrictClassicalSimState works when all sub-bloqs have on_classical_vals."""
    bloq = _DoubleNot()
    assert simulate_via_subbloq_classical_vals(bloq, q=0) == (0,)
    assert simulate_via_subbloq_classical_vals(bloq, q=1) == (1,)


# ===========================================================================
# Tests: simulate_via_subbloq_classical_vals
# ===========================================================================


def test_call_no_decomposition_raises():
    """Bloqs without decomposition raise DecomposeNotImplementedError."""
    bloq = _IdentityBloq()
    with pytest.raises(DecomposeNotImplementedError):
        simulate_via_subbloq_classical_vals(bloq, q=0)


# ===========================================================================
# Tests: generate_input_vals
# ===========================================================================


def test_generate_exhaustive_for_small_domain():
    """When n_samples >= domain size, enumerate exhaustively."""
    cnot = CNOT()
    rng = np.random.default_rng(0)
    vals = generate_input_vals(cnot, n_samples=100, rng=rng)
    assert len(vals) == 4
    combos = {(v['ctrl'], v['target']) for v in vals}
    assert combos == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_generate_random_sampling():
    """When n_samples < domain size, return exactly n_samples."""
    from qualtran.bloqs.arithmetic import Add

    add = Add(QUInt(8))
    rng = np.random.default_rng(42)
    vals = generate_input_vals(add, n_samples=10, rng=rng)
    assert len(vals) == 10
    for v in vals:
        assert 'a' in v and 'b' in v


def test_generate_no_inputs():
    """Bloqs with no LEFT registers return a single empty dict."""
    from qualtran.bloqs.bookkeeping import Allocate

    alloc = Allocate(QBit())
    rng = np.random.default_rng(0)
    vals = generate_input_vals(alloc, n_samples=10, rng=rng)
    assert vals == [{}]


def test_generate_reproducible_with_same_seed():
    from qualtran.bloqs.arithmetic import Add

    add = Add(QUInt(4))
    vals1 = generate_input_vals(add, 5, np.random.default_rng(123))
    vals2 = generate_input_vals(add, 5, np.random.default_rng(123))
    assert vals1 == vals2


def test_generate_large_domain_values_in_range():
    """QUInt(16) has domain size 65,536 > n_samples=20, so random
    sampling is used. All generated values must be valid (0 <= x < 2^16).
    """
    bloq = _LargeDomainBloq()
    rng = np.random.default_rng(42)
    vals = generate_input_vals(bloq, n_samples=20, rng=rng)
    assert len(vals) == 20
    for v in vals:
        assert 'x' in v
        assert 0 <= v['x'] < 2**16, f"Value {v['x']} out of range"


def test_generate_large_domain_values_are_integers():
    """Sampled values for QUInt(16) should be integers (not floats)."""
    bloq = _LargeDomainBloq()
    rng = np.random.default_rng(42)
    vals = generate_input_vals(bloq, n_samples=10, rng=rng)
    for v in vals:
        assert isinstance(v['x'], (int, np.integer)), f"Expected integer type, got {type(v['x'])}"


def test_generate_large_domain_has_variety():
    """Random sampling should produce variety, not all the same value."""
    bloq = _LargeDomainBloq()
    rng = np.random.default_rng(42)
    vals = generate_input_vals(bloq, n_samples=20, rng=rng)
    unique_values = {v['x'] for v in vals}
    assert len(unique_values) > 1, "Random sampling produced only one unique value"


def test_generate_shaped_register_correct_shape():
    """Shaped register (shape=(5,)) should produce NDArrays with shape (5,)."""
    bloq = _ShapedRegBloq()
    rng = np.random.default_rng(42)
    vals = generate_input_vals(bloq, n_samples=10, rng=rng)
    assert len(vals) == 10
    for v in vals:
        assert 'arr' in v
        arr = v['arr']
        assert isinstance(arr, np.ndarray), f"Expected ndarray, got {type(arr)}"
        assert arr.shape == (5,), f"Expected shape (5,), got {arr.shape}"


def test_generate_shaped_register_values_in_domain():
    """Each element of a shaped QBit register should be 0 or 1."""
    bloq = _ShapedRegBloq()
    rng = np.random.default_rng(42)
    vals = generate_input_vals(bloq, n_samples=10, rng=rng)
    for v in vals:
        arr = v['arr']
        assert isinstance(arr, np.ndarray)
        for elem in arr.flat:
            assert elem in (0, 1), f"QBit element {elem} not in {{0, 1}}"


def test_generate_shaped_register_forces_sampling():
    """Shaped registers force random sampling even when n_samples
    is large. Should return exactly n_samples dicts, not exhaustive.
    """
    bloq = _ShapedRegBloq()
    rng = np.random.default_rng(42)
    vals = generate_input_vals(bloq, n_samples=50, rng=rng)
    assert len(vals) == 50


def test_generate_mixed_scalar_and_shaped():
    """A bloq with both scalar QBit and shaped QBit registers.

    The scalar register should be an integer, and the shaped
    register should be an NDArray with the correct shape.
    """
    bloq = _MixedRegBloq()
    rng = np.random.default_rng(42)
    vals = generate_input_vals(bloq, n_samples=15, rng=rng)
    assert len(vals) == 15
    for v in vals:
        # Scalar register.
        assert 'scalar' in v
        assert isinstance(v['scalar'], (int, np.integer))
        assert v['scalar'] in (0, 1)
        # Shaped register.
        assert 'array' in v
        arr = v['array']
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3,)
        for elem in arr.flat:
            assert elem in (0, 1)


def test_generate_mixed_register_forces_sampling():
    """Presence of shaped register forces random sampling for all."""
    bloq = _MixedRegBloq()
    rng = np.random.default_rng(42)
    vals = generate_input_vals(bloq, n_samples=30, rng=rng)
    assert len(vals) == 30


def test_generate_output_only_bloq_returns_single_empty_dict():
    """Bloqs with no LEFT registers (only RIGHT) should return [{}]."""

    @attrs.frozen
    class _OutputOnlyBloq(Bloq):
        @cached_property
        def signature(self):
            return Signature([Register('out', QBit(), side=Side.RIGHT)])

        def on_classical_vals(self):
            return {'out': 0}

    bloq = _OutputOnlyBloq()
    rng = np.random.default_rng(42)
    vals = generate_input_vals(bloq, n_samples=10, rng=rng)
    assert vals == [{}]


# ===========================================================================
# Tests: _assert_results_equal
# ===========================================================================


def test_assert_equal_scalars():
    _assert_results_equal((1, 2, 3), (1, 2, 3), _IdentityBloq(), {})


def test_assert_unequal_scalars():
    with pytest.raises(AssertionError, match="Mismatch"):
        _assert_results_equal((1,), (2,), _IdentityBloq(), {'q': 0})


def test_assert_length_mismatch():
    with pytest.raises(AssertionError, match="length mismatch"):
        _assert_results_equal((1, 2), (1,), _IdentityBloq(), {})


def test_assert_equal_arrays():
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3])
    _assert_results_equal((a,), (b,), _IdentityBloq(), {})


def test_assert_unequal_arrays():
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 4])
    with pytest.raises(AssertionError, match="Mismatch"):
        _assert_results_equal((a,), (b,), _IdentityBloq(), {'q': 0})


def test_assert_nan_scalar_fails():
    """NaN != NaN, so two results containing NaN should raise.

    float('nan') != float('nan') evaluates to True in Python (meaning
    they are NOT equal). The comparison function should report a mismatch.
    """
    nan = float('nan')
    with pytest.raises(AssertionError, match='Mismatch'):
        _assert_results_equal((nan,), (nan,), _IdentityBloq(), {'q': 0})


def test_assert_nan_in_array_fails():
    """NaN in numpy arrays causes comparison failure."""
    a = np.array([1.0, float('nan')])
    b = np.array([1.0, float('nan')])
    with pytest.raises(AssertionError, match='Mismatch'):
        _assert_results_equal((a,), (b,), _IdentityBloq(), {'q': 0})


def test_assert_int_vs_np_int64_equal():
    """Python int(1) and np.int64(1) should compare equal."""
    _assert_results_equal((int(1),), (np.int64(1),), _IdentityBloq(), {'q': 1})


def test_assert_int_vs_np_int_different_values():
    """int(1) vs np.int64(0) should fail."""
    with pytest.raises(AssertionError, match='Mismatch'):
        _assert_results_equal((int(1),), (np.int64(0),), _IdentityBloq(), {'q': 0})


def test_assert_empty_tuples_equal():
    """Two empty tuples should compare equal (bloq with no RIGHT registers)."""
    _assert_results_equal((), (), _IdentityBloq(), {})


def test_assert_array_vs_scalar_mismatch():
    """Comparing ndarray vs scalar when shapes differ should fail."""
    with pytest.raises(AssertionError, match='Mismatch'):
        _assert_results_equal((np.array([1]),), (1,), _IdentityBloq(), {'q': 0})


def test_assert_multi_element_tuple():
    """Multi-element tuples with mixed types."""
    _assert_results_equal(
        (0, np.array([1, 2]), 3), (0, np.array([1, 2]), 3), _IdentityBloq(), {'q': 0}
    )


def test_assert_multi_element_partial_mismatch():
    """Only one element in a multi-element tuple differs."""
    with pytest.raises(AssertionError, match='output index 1'):
        _assert_results_equal(
            (0, np.array([1, 2]), 3), (0, np.array([1, 9]), 3), _IdentityBloq(), {'q': 0}
        )


# ===========================================================================
# Tests: verify_structural_induction
# ===========================================================================


def test_verify_passing_bloq():
    """A correct bloq with decomposition produces 'passed' status."""
    cases = [ClassicalSimTestCase(bloq=_DoubleNot(), name='double_not')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=10, log=log)
    statuses = {r.status for r in results}
    assert VerificationStatus.PASSED in statuses


def test_verify_catches_wrong_on_classical_vals():
    """verify_structural_induction must catch a bloq whose on_classical_vals
    disagrees with its decomposition.

    _BuggyDoubleNot has on_classical_vals returning NOT, but its decomposition
    (XX) correctly computes identity. The verifier should detect this mismatch
    and report status='failed'.
    """
    bloq = _BuggyDoubleNot()
    cases = [ClassicalSimTestCase(bloq=bloq, name='buggy')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=10, log=log)
    root_result = results[0]
    assert root_result.bloq.__class__.__name__ == '_BuggyDoubleNot'
    assert root_result.status == VerificationStatus.FAILED
    assert root_result.error_message is not None
    assert 'Mismatch' in root_result.error_message

    # Confirm the mismatch directly: on_classical_vals says NOT,
    # decomposition says identity.
    assert bloq.on_classical_vals(q=0) == {'q': 1}
    assert simulate_via_subbloq_classical_vals(bloq, q=0) == (0,)


def test_verify_correct_on_classical_vals_passes():
    """A bloq whose on_classical_vals agrees with its decomposition should
    produce status='passed'."""
    cases = [ClassicalSimTestCase(bloq=_DoubleNot(), name='correct')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=10, log=log)
    root_result = results[0]
    assert root_result.bloq.__class__.__name__ == '_DoubleNot'
    assert root_result.status == VerificationStatus.PASSED


def test_verify_no_on_classical_vals_still_cross_checks():
    """A bloq without on_classical_vals: the parent check should be
    skipped gracefully, not cause an error."""
    cases = [ClassicalSimTestCase(bloq=_NoReferenceBloq(), name='no_ref')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=10, log=log)
    root_result = results[0]
    # Should be 'missing_classical_vals', not 'error'.
    assert root_result.status == VerificationStatus.MISSING_CLASSICAL_VALS


def test_verify_skips_parent_check_when_no_reference():
    """When a bloq has a decomposition but no on_classical_vals, the verifier
    must NOT fall back to call_classically (which would silently use the
    decomposition, comparing it against itself).

    _DecomposableNoReference decomposes into _DoubleNot (which is correct),
    but has no on_classical_vals. The verifier should report
    'missing_classical_vals' — NOT 'passed' from a tautological self-comparison.
    """
    cases = [ClassicalSimTestCase(bloq=_DecomposableNoReference(), name='no_ref_parent')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=10, log=log)
    root_result = results[0]
    assert root_result.bloq.__class__.__name__ == '_DecomposableNoReference'
    assert root_result.status == VerificationStatus.MISSING_CLASSICAL_VALS


def test_verify_missing_classical_vals():
    """A bloq whose children lack on_classical_vals → 'missing_classical_vals'."""
    cases = [ClassicalSimTestCase(bloq=_NoReferenceBloq(), name='no_ref')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=10, log=log)
    root_result = results[0]
    assert root_result.status == VerificationStatus.MISSING_CLASSICAL_VALS


def test_verify_partial_correctness_caught():
    """A bloq correct on input 0 but wrong on input 1 must be caught.

    The verifier must test both inputs for QBit (exhaustive enumeration).
    """
    bloq = _PartiallyWrongBloq()
    cases = [ClassicalSimTestCase(bloq=bloq, name='partial_bug')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=10, log=log)
    root = results[0]
    assert root.status == VerificationStatus.FAILED, "Verifier should detect the bug on input q=1."
    assert root.error_message is not None
    assert 'Mismatch' in root.error_message

    # Confirm partial correctness directly.
    assert bloq.on_classical_vals(q=0) == {'q': 0}
    assert simulate_via_subbloq_classical_vals(bloq, q=0) == (0,)
    assert bloq.on_classical_vals(q=1) == {'q': 0}
    assert simulate_via_subbloq_classical_vals(bloq, q=1) == (1,)


def test_verify_sub_bloq_bug_propagates():
    """A parent with one correct and one buggy sub-bloq.

    The parent's on_classical_vals is correct (identity). But the
    decomposition uses a buggy sub-bloq that claims identity instead
    of NOT. simulate_via_subbloq_classical_vals computes the wrong answer, resulting
    in a 'failed' status.
    """
    bloq = _ParentWithOneBuggySubBloq()
    cases = [ClassicalSimTestCase(bloq=bloq, name='buggy_sub')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=10, log=log)
    root = results[0]
    assert (
        root.status == VerificationStatus.FAILED
    ), "Sub-bloq's buggy on_classical_vals should cause parent failure."
    assert root.error_message is not None
    assert 'Mismatch' in root.error_message


def test_verify_consistent_sub_bloq_bugs_can_mask():
    """Two consistently-wrong sub-bloqs can mask each other.

    _ParentWithBuggySubBloqs uses two _AtomicBuggyIdentity instances.
    Each claims identity (wrong — should be NOT), so two identities =
    identity, matching the parent's reference. The parent passes, but
    _AtomicBuggyIdentity would fail its own check if it had a decomposition.
    """
    bloq = _ParentWithBuggySubBloqs()
    cases = [ClassicalSimTestCase(bloq=bloq, name='consistent_bugs')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=10, log=log)
    root = results[0]
    assert root.status == VerificationStatus.PASSED


def test_verify_domain_constrained_n_inputs_checked():
    """Skipped inputs must NOT count as checked.

    _DomainConstrainedBloq raises on odd inputs. For QUInt(4), roughly
    half the inputs should be skipped. n_inputs_checked must reflect
    only the inputs where both sides could be compared.
    """
    bloq = _DomainConstrainedBloq(n=4)
    cases = [ClassicalSimTestCase(bloq=bloq, name='constrained')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=100, log=log)
    root = results[0]
    assert root.status == VerificationStatus.PASSED
    # QUInt(4) has 16 values. With n_samples=100 >= 16, exhaustive.
    # Only even values (0,2,4,6,8,10,12,14) = 8 pass the domain check.
    assert root.n_inputs_checked == 8, (
        f"Expected 8 checked inputs (even values 0-14), " f"got {root.n_inputs_checked}."
    )


def test_verify_deduplication():
    """The same bloq instance appearing multiple times is verified only once."""
    cases = [
        ClassicalSimTestCase(bloq=_DoubleNot(), name='dn1'),
        ClassicalSimTestCase(bloq=_DoubleNot(), name='dn2'),
    ]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=10, log=log)
    bloq_classes = [r.bloq.__class__.__name__ for r in results]
    assert bloq_classes.count('_DoubleNot') == 1


def test_verify_children_populated():
    """Results for decomposable bloqs have children populated."""
    cases = [ClassicalSimTestCase(bloq=_DoubleNot(), name='dn')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=10, log=log)
    dn_result = [r for r in results if r.bloq.__class__.__name__ == '_DoubleNot'][0]
    assert len(dn_result.children) > 0
    child_classes = [c.__class__.__name__ for c in dn_result.children]
    assert 'XGate' in child_classes


def test_verify_leaf_bloqs_have_no_children():
    """Leaf bloqs (no decomposition) have empty children."""
    cases = [ClassicalSimTestCase(bloq=_DoubleNot(), name='dn')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=10, log=log)
    leaf_results = [r for r in results if r.status == VerificationStatus.LEAF_VERIFIED]
    for r in leaf_results:
        assert r.children == ()


def test_verify_diamond_dag_catches_deep_bug():
    """Diamond DAG with a buggy deep child.

    _DiamondTopBloq decomposes into two _DiamondMid instances, each
    of which decomposes into _AtomicNot (correct NOT) and
    _AtomicBuggyIdentity (wrong — identity instead of NOT). The bug
    causes _DiamondMid's verification to fail.
    """
    bloq = _DiamondTopBloq()
    cases = [ClassicalSimTestCase(bloq=bloq, name='diamond')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=10, log=log)

    # Find _DiamondMid result — it should fail.
    mid_results = [r for r in results if r.bloq.__class__.__name__ == '_DiamondMid']
    assert len(mid_results) == 1, "BFS should discover _DiamondMid"
    assert (
        mid_results[0].status == VerificationStatus.FAILED
    ), "Buggy deep child should cause mid-level bloq to fail."
    assert mid_results[0].error_message is not None
    assert 'Mismatch' in mid_results[0].error_message


def test_verify_diamond_dag_discovers_all_nodes():
    """BFS traversal of diamond DAG discovers all unique bloq classes."""
    bloq = _DiamondTopBloq()
    cases = [ClassicalSimTestCase(bloq=bloq, name='diamond')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=10, log=log)
    classes = {r.bloq.__class__.__name__ for r in results}
    assert '_DiamondTopBloq' in classes
    assert '_DiamondMid' in classes
    assert '_AtomicNot' in classes
    assert '_AtomicBuggyIdentity' in classes


def test_verify_diamond_mid_deduplication():
    """Two instances of _DiamondMid (from x and y paths) should
    be deduplicated — verified only once."""
    bloq = _DiamondTopBloq()
    cases = [ClassicalSimTestCase(bloq=bloq, name='diamond')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=10, log=log)
    mid_results = [r for r in results if r.bloq.__class__.__name__ == '_DiamondMid']
    assert len(mid_results) == 1, "Identical _DiamondMid instances should be deduplicated."


def test_verify_3_level_deep_bug_discovered():
    """3-level decomposition with a buggy leaf at the bottom.

    _Level2Top → _Level1Mid → _AtomicNot + _AtomicBuggyIdentity.
    BFS must discover all 3+ levels and detect that _Level1Mid
    fails due to the buggy leaf.
    """
    bloq = _Level2Top()
    cases = [ClassicalSimTestCase(bloq=bloq, name='deep')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=10, log=log)

    # Discover all levels.
    classes = {r.bloq.__class__.__name__ for r in results}
    assert '_Level2Top' in classes
    assert '_Level1Mid' in classes
    assert '_AtomicNot' in classes
    assert '_AtomicBuggyIdentity' in classes

    # _Level1Mid should fail due to the buggy leaf.
    mid_results = [r for r in results if r.bloq.__class__.__name__ == '_Level1Mid']
    assert len(mid_results) == 1
    assert mid_results[0].status == VerificationStatus.FAILED
    assert mid_results[0].error_message is not None
    assert 'Mismatch' in mid_results[0].error_message


def test_verify_n_inputs_checked_for_qbit():
    """For a QBit bloq with exhaustive enumeration, n_inputs_checked
    should be exactly 2 (for inputs 0 and 1)."""
    bloq = _Level2Top()
    cases = [ClassicalSimTestCase(bloq=bloq, name='deep')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=10, log=log)

    top = [r for r in results if r.bloq.__class__.__name__ == '_Level2Top'][0]
    assert top.n_inputs_checked == 2


def test_verify_all_raising_reports_no_valid_inputs():
    """A bloq whose on_classical_vals raises on ALL inputs gets
    status 'no_valid_inputs' with n_inputs_checked=0.

    The verifier must NOT report 'passed' when nothing was actually
    verified. Instead it reports 'no_valid_inputs' to flag that the
    verification was inconclusive.
    """
    bloq = _AllRaisingBloq()
    cases = [ClassicalSimTestCase(bloq=bloq, name='all_raising')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=10, log=log)
    root = results[0]
    assert (
        root.status == VerificationStatus.NO_VALID_INPUTS
    ), "All inputs raised — should be 'no_valid_inputs', not 'passed'."
    assert root.n_inputs_checked == 0, "All inputs raised — none should be counted as checked."
    assert root.error_message is not None
    assert 'raised on all inputs' in root.error_message


def test_verify_multi_level_status_propagation():
    """Full pipeline with 3-level hierarchy checks status propagation."""
    bloq = _Level2Top()
    cases = [ClassicalSimTestCase(bloq=bloq, name='multi_level')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=10, log=log)

    statuses = {r.bloq.__class__.__name__: r.status for r in results}
    # Level2Top should pass (identity matches).
    assert statuses['_Level2Top'] == VerificationStatus.PASSED
    # Level1Mid should fail (buggy leaf corrupts decomposition).
    assert statuses['_Level1Mid'] == VerificationStatus.FAILED
    # Atomic leaf bloqs that raise DecomposeTypeError but are NOT in the ISA
    # get 'missing_decomposition' — they are custom test bloqs, not standard
    # Qualtran gates.
    assert statuses['_AtomicNot'] == VerificationStatus.MISSING_DECOMPOSITION
    assert statuses['_AtomicBuggyIdentity'] == VerificationStatus.MISSING_DECOMPOSITION


def test_verify_constrained_inputs_log():
    """Domain-constrained bloq produces valid log output."""
    bloq = _DomainConstrainedBloq(n=4)
    cases = [ClassicalSimTestCase(bloq=bloq, name='constrained')]
    log = io.StringIO()
    verify_structural_induction(cases, n_samples=100, log=log)

    log_text = log.getvalue()
    assert len(log_text) > 0
    tab_lines = [l for l in log_text.split('\n') if '\t' in l]
    assert len(tab_lines) > 0


def test_verify_bfs_discovers_all_descendants():
    """BFS discovers bloqs at all levels of the decomposition tree."""
    cases = [ClassicalSimTestCase(bloq=_ParentBloq(), name='parent')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=10, log=log)
    classes = {r.bloq.__class__.__name__ for r in results}
    assert '_ParentBloq' in classes
    assert '_DoubleNot' in classes
    assert 'XGate' in classes


def test_verify_default_rng_is_deterministic():
    """Without explicit rng, results are reproducible."""
    cases = [ClassicalSimTestCase(bloq=_DoubleNot(), name='dn')]
    r1 = verify_structural_induction(cases, n_samples=10, log=None)
    r2 = verify_structural_induction(cases, n_samples=10, log=None)
    assert len(r1) == len(r2)
    for a, b in zip(r1, r2):
        assert a.status == b.status
        assert a.n_inputs_checked == b.n_inputs_checked


def test_verify_log_is_none():
    """Passing log=None suppresses output without errors."""
    cases = [ClassicalSimTestCase(bloq=_DoubleNot(), name='dn')]
    results = verify_structural_induction(cases, n_samples=10, log=None)
    assert len(results) > 0


def test_verify_log_format_tab_separated():
    """Log output is tab-separated with 4 columns per line."""
    cases = [ClassicalSimTestCase(bloq=_DoubleNot(), name='dn')]
    log = io.StringIO()
    verify_structural_induction(cases, n_samples=10, log=log)
    log_text = log.getvalue()
    for line in log_text.rstrip('\n').split('\n'):
        fields = line.split('\t')
        assert len(fields) == 4, f"Expected 4 tab-separated fields, got {len(fields)}: {line!r}"


def test_verify_log_no_non_ascii():
    """Log output contains no non-ASCII characters."""
    cases = [ClassicalSimTestCase(bloq=_DoubleNot(), name='dn')]
    log = io.StringIO()
    verify_structural_induction(cases, n_samples=10, log=log)
    log_text = log.getvalue()
    for char in log_text:
        assert ord(char) < 128, f"Non-ASCII character in log: {char!r} (U+{ord(char):04X})"


def test_verify_log_contains_objectstring():
    """Log uses dump_objectstring for bloq identification."""
    cases = [ClassicalSimTestCase(bloq=_DoubleNot(), name='dn')]
    log = io.StringIO()
    verify_structural_induction(cases, n_samples=10, log=log)
    assert '_DoubleNot' in log.getvalue()


# ===========================================================================
# Tests: VerificationResult (attrs frozen)
# ===========================================================================


def test_verification_result_is_frozen():
    r = VerificationResult(bloq=_IdentityBloq(), status=VerificationStatus.PASSED)
    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        r.status = VerificationStatus.FAILED  # type: ignore[misc]


def test_verification_result_defaults():
    r = VerificationResult(bloq=_IdentityBloq(), status=VerificationStatus.PASSED)
    assert r.n_inputs_checked == 0
    assert r.error_message is None
    assert r.children == ()


def test_verification_result_children_is_tuple():
    r = VerificationResult(
        bloq=_IdentityBloq(), status=VerificationStatus.PASSED, children=(_IdentityBloq(),)
    )
    assert isinstance(r.children, tuple)


# ===========================================================================
# Tests: ClassicalSimTestCase (attrs frozen)
# ===========================================================================


def test_classical_sim_test_case_is_frozen():
    tc = ClassicalSimTestCase(bloq=_IdentityBloq(), name='test')
    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        tc.name = 'other'  # type: ignore[misc]


# ===========================================================================
# Tests: print_verification_summary
# ===========================================================================


def _make_tree_results() -> list[VerificationResult]:
    """Build a small result set with known structure.

    Structure:
        _ParentBloq (passed)
          └─ _DoubleNot (passed)
               └─ XGate (leaf_verified)
    """
    xgate = XGate()
    double_not = _DoubleNot()
    parent = _ParentBloq()
    return [
        VerificationResult(
            bloq=parent,
            status=VerificationStatus.PASSED,
            n_inputs_checked=4,
            children=(double_not,),
        ),
        VerificationResult(
            bloq=double_not, status=VerificationStatus.PASSED, n_inputs_checked=2, children=(xgate,)
        ),
        VerificationResult(bloq=xgate, status=VerificationStatus.LEAF_VERIFIED, n_inputs_checked=2),
    ]


def test_summary_contains_header():
    results = _make_tree_results()
    buf = io.StringIO()
    print_verification_summary(results, file=buf)
    assert 'VERIFICATION SUMMARY' in buf.getvalue()


def test_summary_uses_class_names():
    results = _make_tree_results()
    buf = io.StringIO()
    print_verification_summary(results, file=buf)
    text = buf.getvalue()
    assert '_ParentBloq' in text
    assert '_DoubleNot' in text
    assert 'XGate' in text


def test_summary_tree_indentation():
    """Child nodes are indented further than their parents."""
    results = _make_tree_results()
    buf = io.StringIO()
    print_verification_summary(results, file=buf)
    lines = buf.getvalue().split('\n')
    parent_line = [l for l in lines if '_ParentBloq' in l][0]
    child_line = [l for l in lines if '_DoubleNot' in l][0]
    # The child's class name should start at a greater column offset.
    assert child_line.index('_DoubleNot') > parent_line.index('_ParentBloq')


def test_summary_dag_deduplication_see_above():
    """When the same class appears under multiple parents, subsequent
    appearances show '(ClassName) [icon see above]'."""
    xgate = XGate()
    dn = _DoubleNot()
    parent = _ParentBloq()

    @attrs.frozen
    class _AnotherParent(Bloq):
        @cached_property
        def signature(self):
            return Signature([Register('q', QBit())])

        def on_classical_vals(self, q):
            return {'q': q}

    another = _AnotherParent()

    results = [
        VerificationResult(
            bloq=parent, status=VerificationStatus.PASSED, n_inputs_checked=4, children=(dn,)
        ),
        VerificationResult(
            bloq=another, status=VerificationStatus.PASSED, n_inputs_checked=2, children=(dn,)
        ),
        VerificationResult(
            bloq=dn, status=VerificationStatus.PASSED, n_inputs_checked=2, children=(xgate,)
        ),
        VerificationResult(bloq=xgate, status=VerificationStatus.LEAF_VERIFIED, n_inputs_checked=2),
    ]

    buf = io.StringIO()
    print_verification_summary(results, file=buf)
    text = buf.getvalue()
    assert 'see above' in text
    assert '(_DoubleNot)' in text


def test_summary_omits_leaf_verified_already_printed():
    """Leaf-verified nodes that have already been printed are completely
    omitted from subsequent tree appearances (no 'see above' line).

    Non-leaf-verified nodes still show the '(ClassName) [see above]'
    back-reference when they appear under multiple parents.
    """
    xgate = XGate()

    @attrs.frozen
    class _Parent1(Bloq):
        @cached_property
        def signature(self):
            return Signature([Register('q', QBit())])

        def on_classical_vals(self, q):
            return {'q': q}

    @attrs.frozen
    class _Parent2(Bloq):
        @cached_property
        def signature(self):
            return Signature([Register('q', QBit())])

        def on_classical_vals(self, q):
            return {'q': q}

    p1 = _Parent1()
    p2 = _Parent2()

    results = [
        VerificationResult(
            bloq=p1, status=VerificationStatus.PASSED, n_inputs_checked=2, children=(xgate,)
        ),
        VerificationResult(
            bloq=p2, status=VerificationStatus.PASSED, n_inputs_checked=2, children=(xgate,)
        ),
        VerificationResult(bloq=xgate, status=VerificationStatus.LEAF_VERIFIED, n_inputs_checked=2),
    ]

    buf = io.StringIO()
    print_verification_summary(results, file=buf)
    text = buf.getvalue()

    # XGate is leaf_verified: it should appear exactly once (under _Parent1)
    # and be completely omitted under _Parent2 — no '(XGate)' back-reference.
    xgate_lines = [l for l in text.split('\n') if 'XGate' in l]
    assert (
        len(xgate_lines) == 1
    ), f"Expected XGate to appear exactly once, got {len(xgate_lines)}: {xgate_lines}"
    assert (
        '(XGate)' not in text
    ), "Leaf-verified XGate should be omitted, not shown as '(XGate) [see above]'"


def test_summary_status_totals():
    """Footer contains correct status counts."""
    results = _make_tree_results()
    buf = io.StringIO()
    print_verification_summary(results, file=buf)
    text = buf.getvalue()
    assert 'Total: 3' in text
    assert 'passed: 2' in text
    assert 'leaf_verified: 1' in text


def test_summary_class_name_deduplication():
    """Multiple results for the same class merge into one tree node."""
    xgate = XGate()
    dn = _DoubleNot()

    results = [
        VerificationResult(
            bloq=dn, status=VerificationStatus.PASSED, n_inputs_checked=2, children=(xgate,)
        ),
        VerificationResult(bloq=xgate, status=VerificationStatus.LEAF_VERIFIED, n_inputs_checked=2),
    ]

    buf = io.StringIO()
    print_verification_summary(results, file=buf)
    text = buf.getvalue()

    # Filter to tree body lines (exclude header/footer/totals).
    tree_lines = [
        l
        for l in text.split('\n')
        if l.strip()
        and not l.startswith('=')
        and not l.startswith('-')
        and 'Total' not in l
        and 'SUMMARY' not in l
    ]
    dn_lines = [l for l in tree_lines if '_DoubleNot' in l]
    xgate_lines = [l for l in tree_lines if 'XGate' in l]
    assert len(dn_lines) == 1
    assert len(xgate_lines) == 1


def test_summary_worst_status_propagation():
    """When merging by class name, the worst status wins."""
    dn1 = _DoubleNot()
    dn2 = _DoubleNot()

    results = [
        VerificationResult(bloq=dn1, status=VerificationStatus.PASSED, n_inputs_checked=2),
        VerificationResult(
            bloq=dn2,
            status=VerificationStatus.ERROR,
            n_inputs_checked=0,
            error_message='something broke',
        ),
    ]

    buf = io.StringIO()
    print_verification_summary(results, file=buf)
    text = buf.getvalue()
    # The merged node should show the worse status.
    assert '[error]' in text


def test_summary_empty_results():
    """Printing with no results does not crash."""
    buf = io.StringIO()
    print_verification_summary([], file=buf)
    assert 'Total: 0' in buf.getvalue()


def test_summary_root_identification():
    """Bloqs not appearing as children of others are displayed as roots."""
    dn = _DoubleNot()
    xgate = XGate()

    results = [
        VerificationResult(
            bloq=dn, status=VerificationStatus.PASSED, n_inputs_checked=2, children=(xgate,)
        ),
        VerificationResult(bloq=xgate, status=VerificationStatus.LEAF_VERIFIED, n_inputs_checked=2),
    ]

    buf = io.StringIO()
    print_verification_summary(results, file=buf)
    lines = buf.getvalue().split('\n')

    # _DoubleNot is a root: no tree connector before its name.
    dn_line = [l for l in lines if '_DoubleNot' in l][0]
    prefix = dn_line.split('_DoubleNot')[0]
    assert '├' not in prefix
    assert '└' not in prefix

    # XGate is a child: has a tree connector.
    xgate_line = [l for l in lines if 'XGate' in l][0]
    prefix = xgate_line.split('XGate')[0]
    assert '└' in prefix or '├' in prefix


def test_summary_inputs_aggregated():
    """n_inputs_checked is summed across results of the same class."""
    dn1 = _DoubleNot()
    dn2 = _DoubleNot()

    results = [
        VerificationResult(bloq=dn1, status=VerificationStatus.PASSED, n_inputs_checked=10),
        VerificationResult(bloq=dn2, status=VerificationStatus.PASSED, n_inputs_checked=15),
    ]

    buf = io.StringIO()
    print_verification_summary(results, file=buf)
    text = buf.getvalue()
    # Total should be 10 + 15 = 25.
    assert '(25 inputs)' in text


# ===========================================================================
# Integration test
# ===========================================================================


def test_end_to_end_with_add():
    """Full pipeline with a real Qualtran bloq (Add)."""
    from qualtran.bloqs.arithmetic import Add

    add = Add(QUInt(3))
    cases = [ClassicalSimTestCase(bloq=add, name='add_3bit')]
    log = io.StringIO()
    results = verify_structural_induction(
        cases, n_samples=20, rng=np.random.default_rng(0), log=log
    )

    # Should have at least the root + some children.
    assert len(results) >= 1

    # Root should be Add.
    root = results[0]
    assert root.bloq.__class__.__name__ == 'Add'
    assert root.status in (VerificationStatus.PASSED, VerificationStatus.MISSING_CLASSICAL_VALS)

    # Log should be parseable.
    log_text = log.getvalue()
    assert len(log_text) > 0
    for line in log_text.rstrip('\n').split('\n'):
        assert len(line.split('\t')) == 4

    # Summary should not crash.
    buf = io.StringIO()
    print_verification_summary(results, file=buf)
    summary = buf.getvalue()
    assert 'VERIFICATION SUMMARY' in summary
    assert 'Add' in summary


# ===========================================================================
# Tests: broad-except hardening
# ===========================================================================
# These tests verify that genuine bugs (TypeError, KeyError, etc.) in
# on_classical_vals are NOT silently swallowed as "domain-constrained"
# inputs. Before the fix, these would have been silently skipped.


def test_verify_typeerror_in_on_classical_vals_not_swallowed():
    """A TypeError in on_classical_vals is a real bug, not a domain constraint.

    Before the fix, `except Exception` would skip this input silently.
    If ALL inputs raise TypeError, the old code would report NO_VALID_INPUTS
    (misleading). The narrowed catch should let TypeError propagate as ERROR.
    """
    bloq = _TypeErrorInOnClassicalVals()
    cases = [ClassicalSimTestCase(bloq=bloq, name='type_error_bug')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=10, log=log)
    root = results[0]
    # Must NOT be PASSED or NO_VALID_INPUTS — those would mean the TypeError
    # was silently swallowed.
    assert root.status == VerificationStatus.ERROR, (
        f"Expected ERROR for TypeError bug, got {root.status}. "
        f"The broad 'except Exception' would have hidden this."
    )
    assert root.error_message is not None
    assert 'TypeError' in root.error_message


def test_verify_keyerror_in_on_classical_vals_not_swallowed():
    """A KeyError in on_classical_vals is a real bug, not a domain constraint.

    Before the fix, `except Exception` would skip this input silently.
    """
    bloq = _KeyErrorInOnClassicalVals()
    cases = [ClassicalSimTestCase(bloq=bloq, name='key_error_bug')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=10, log=log)
    root = results[0]
    assert root.status == VerificationStatus.ERROR, (
        f"Expected ERROR for KeyError bug, got {root.status}. "
        f"The broad 'except Exception' would have hidden this."
    )
    assert root.error_message is not None
    assert 'KeyError' in root.error_message


def test_verify_typeerror_leaf_not_silently_verified():
    """A leaf sub-bloq with a TypeError bug must cause its parent to ERROR.

    _ParentOfBuggyLeaf decomposes into _TypeErrorLeafBloq, whose
    on_classical_vals raises TypeError. When the verifier runs
    simulate_via_subbloq_classical_vals on the parent, this TypeError propagates.

    Before the fix, the broad `except Exception` on the inner loop would
    have caught the TypeError and silently skipped the input, potentially
    reporting PASSED or NO_VALID_INPUTS instead of ERROR.
    """
    bloq = _ParentOfBuggyLeaf()
    cases = [ClassicalSimTestCase(bloq=bloq, name='buggy_leaf_parent')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=10, log=log)

    parent = [r for r in results if r.bloq.__class__.__name__ == '_ParentOfBuggyLeaf']
    assert len(parent) == 1
    assert parent[0].status == VerificationStatus.ERROR, (
        f"Expected ERROR for TypeError in leaf sub-bloq, got {parent[0].status}. "
        f"The broad 'except Exception' would have hidden this."
    )
    assert parent[0].error_message is not None
    assert 'TypeError' in parent[0].error_message


def test_verify_domain_constraint_error_skipped():
    """QCDTypeDomainError in on_classical_vals is the canonical way to
    signal that an input is outside the bloq's valid domain.

    The verifier must skip these inputs gracefully.
    """
    bloq = _DomainConstrainedBloq(n=4)
    cases = [ClassicalSimTestCase(bloq=bloq, name='constrained')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=100, log=log)
    root = results[0]
    assert root.status == VerificationStatus.PASSED
    assert root.n_inputs_checked == 8  # only even values 0-14


def test_verify_bare_valueerror_is_not_a_domain_constraint():
    """Bare ValueError in on_classical_vals is NOT treated as a domain constraint.

    Bloqs must use QCDTypeDomainError for domain validation. A bare ValueError
    is treated as a real bug and reported as ERROR.
    """

    @attrs.frozen
    class _BareValueErrorBloq(Bloq):
        @cached_property
        def signature(self):
            return Signature([Register('q', QBit())])

        def on_classical_vals(self, q):
            if q == 1:
                raise ValueError("Not using QCDTypeDomainError")
            return {'q': q}

        def build_composite_bloq(self, bb: BloqBuilder, q: SoquetT) -> dict[str, SoquetT]:
            return {'q': q}

    bloq = _BareValueErrorBloq()
    cases = [ClassicalSimTestCase(bloq=bloq, name='bare_valueerror')]
    log = io.StringIO()
    results = verify_structural_induction(cases, n_samples=10, log=log)
    root = results[0]
    # Bare ValueError must NOT be silently swallowed — it's a bug.
    assert root.status == VerificationStatus.ERROR, (
        f"Expected ERROR for bare ValueError, got {root.status}. "
        f"Bloqs must use QCDTypeDomainError for domain constraints."
    )


def test_domain_constraint_error_is_a_valueerror():
    """QCDTypeDomainError subclasses ValueError for backward compatibility."""
    err = QCDTypeDomainError("test")
    assert isinstance(err, ValueError)
    assert isinstance(err, QCDTypeDomainError)


def test_should_use_fastsim_env_var(monkeypatch):
    """Explicitly test _should_use_fastsim evaluation logic across QLT_USE_FASTSIM options."""
    # Test forcing disable via environment
    for false_str in ('0', 'false', 'no', 'off', 'FALSE', ' 0 '):
        monkeypatch.setenv('QLT_USE_FASTSIM', false_str)
        assert _should_use_fastsim() is False

    # Test forcing enable via environment
    for true_str in ('1', 'true', 'yes', 'on', 'TRUE', ' 1 '):
        monkeypatch.setenv('QLT_USE_FASTSIM', true_str)
        try:
            assert _should_use_fastsim() is True
        except ImportError:
            pass

    # Test invalid boolean environment string
    monkeypatch.setenv('QLT_USE_FASTSIM', 'not_a_bool')
    with pytest.raises(ValueError, match="Invalid value for QLT_USE_FASTSIM"):
        _should_use_fastsim()

    # Test unset variable falls back to testing import rsqualtran
    monkeypatch.delenv('QLT_USE_FASTSIM', raising=False)
    try:
        import rsqualtran  # type: ignore[import-untyped,import-not-found] # noqa: F401 # pylint: disable=unused-import

        assert _should_use_fastsim() is True
    except ImportError:
        assert _should_use_fastsim() is False


@pytest.mark.parametrize('qlt_use_fastsim', ['0', '1'], indirect=True)
def test_verify_structural_induction_engine_code_paths(qlt_use_fastsim, capsys):
    """Explicitly verify structural induction under both Python (0) and Rust fastsim (1) engines.

    Ensures that real arithmetic bloqs execute and produce PASSED verification results
    under both exact simulation code paths (`simulate_via_subbloq_classical_vals` vs
    `QLTFastsim.call_classically`), and verifying that the active engine style is reported to stderr.
    """
    if qlt_use_fastsim == '1':
        try:
            import rsqualtran  # type: ignore[import-untyped,import-not-found] # noqa: F401 # pylint: disable=unused-import
        except ImportError:
            pytest.skip("rsqualtran (`QLTFastsim`) is not installed in this environment.")

    from qualtran.bloqs.arithmetic.bitwise import Xor

    bloq = Xor(QUInt(3))
    cases = [ClassicalSimTestCase(bloq=bloq, name='xor_test')]
    log = io.StringIO()

    results = verify_structural_induction(cases, n_samples=20, log=log)
    assert results[0].status == VerificationStatus.PASSED

    captured = capsys.readouterr()
    expected_style = (
        "QLTFastsim (Rust)"
        if qlt_use_fastsim == '1'
        else "Python (simulate_via_subbloq_classical_vals)"
    )
    assert f"[verify_structural_induction] Simulation style: {expected_style}" in captured.err
