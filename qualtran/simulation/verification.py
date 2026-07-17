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

"""Functional verification of bloqs via classical simulation cross-checks.

This module verifies that a bloq's decomposition is functionally equivalent
to its hand-written `on_classical_vals`.

The key problem: `Bloq.call_classically()` silently falls back to recursive
decomposition when `on_classical_vals` is not implemented.  Comparing
`bloq.call_classically()` vs `bloq.decompose_bloq().call_classically()`
can therefore be tautological — both sides may execute the exact same code
path.

To avoid this, we provide:

- `StrictClassicalSimState`: a `ClassicalSimState` subclass that raises
  `MissingClassicalValsError` instead of recursing when
  `on_classical_vals` returns `NotImplemented`.

- `simulate_via_subbloq_classical_vals`: decomposes a bloq once, then
  simulates using each sub-bloq's `on_classical_vals`.  Used by
  `verify_structural_induction` as one side of the cross-check (the other
  side being the parent bloq's own `on_classical_vals`).

- `generate_input_vals`: automatic generation of classical input values
  from `QDType.get_classical_domain()`, with random sampling or exhaustive
  enumeration.

- `verify_structural_induction`: BFS over the decomposition tree, verifying
  each bloq encountered via structural induction.
"""

from __future__ import annotations

import enum
import functools
import itertools
import logging
import math
import os
import sys
from collections import Counter, deque, OrderedDict
from typing import Any, Callable, IO, Sequence

import attrs
import numpy as np

from qualtran import Bloq, DecomposeNotImplementedError, DecomposeTypeError, Register
from qualtran.l1 import dump_objectstring, StandardQualtranArchitectureAgnosticVirtualMachine
from qualtran.simulation.classical_sim import ClassicalSimState, ClassicalValT, QCDTypeDomainError


class MissingClassicalValsError(Exception):
    """Raised when a bloq does not implement `on_classical_vals`.

    This means there is no hand-written classical simulation to compare
    against, so the verification cannot proceed for this bloq.
    """


class VerificationStatus(enum.StrEnum):
    """Status of a single bloq verification.

    Using `StrEnum` so members are valid strings — backward-compatible
    with code that compares against plain string literals.
    """

    PASSED = 'passed'
    FAILED = 'failed'
    MISSING_DECOMPOSITION = 'missing_decomposition'
    LEAF_VERIFIED = 'leaf_verified'
    LEAF_NO_CLASSICAL_VALS = 'leaf_no_classical_vals'
    MISSING_CLASSICAL_VALS = 'missing_classical_vals'
    NO_VALID_INPUTS = 'no_valid_inputs'
    ERROR = 'error'


class StrictClassicalSimState(ClassicalSimState):
    """A `ClassicalSimState` that requires `on_classical_vals` on every bloq.

    Unlike the standard `ClassicalSimState` which silently falls back to
    recursive decomposition when `on_classical_vals` returns `NotImplemented`,
    this subclass raises `MissingClassicalValsError`. This guarantees that every
    sub-bloq in the simulation has a hand-written `on_classical_vals`, and
    we are never accidentally comparing simulation against itself.

    Implementation: overrides `_recurse` (the fallback called from `step()`
    when `on_classical_vals` returns `NotImplemented`). The base class's
    `step()` method — including `basis_state_phase` handling — is inherited
    unchanged.
    """

    def _recurse(self, binst, in_vals):
        """Raise instead of recursing into decomposition."""
        bloq = binst.bloq
        raise MissingClassicalValsError(
            f"{bloq.__class__.__name__} (instance: {bloq}) does not implement "
            f"on_classical_vals."
        )


def simulate_via_subbloq_classical_vals(bloq: Bloq, **vals: ClassicalValT) -> tuple:
    """Decompose `bloq` once, then simulate using each sub-bloq's `on_classical_vals`.

    Every sub-bloq in the decomposition must implement `on_classical_vals`;
    if any is missing, raises `MissingClassicalValsError`.

    Args:
        bloq: The bloq to decompose. Must have a decomposition.
        **vals: Classical input values keyed by LEFT register names.

    Returns:
        A tuple of output classical values, ordered by RIGHT registers.

    Raises:
        MissingClassicalValsError: If any sub-bloq lacks `on_classical_vals`.
        DecomposeNotImplementedError: If the bloq has no decomposition.
    """
    cbloq = bloq.decompose_bloq()
    sim = StrictClassicalSimState.from_cbloq(cbloq, vals=vals)
    final_vals = sim.simulate()
    return tuple(final_vals[reg.name] for reg in bloq.signature.rights())


_MAX_DOMAIN_MATERIALIZE = 2**16  # Don't materialize domains larger than this.


def _domain_size_for_register(reg: Register) -> int:
    """Get the size of the classical domain for a register's dtype."""
    # TODO: consider adding get_classical_domain_size() to QDType
    # so this doesn't have to assume 2**num_qubits.
    return 2**reg.dtype.num_qubits


def _sample_from_register(reg: Register, rng: np.random.Generator) -> ClassicalValT:
    """Sample a single random classical value for a register.

    Avoids materializing the full domain for large types by sampling an
    integer index directly.
    """
    domain_size = _domain_size_for_register(reg)

    if domain_size <= _MAX_DOMAIN_MATERIALIZE:
        # Small domain: materialize and index.
        try:
            domain = list(reg.dtype.get_classical_domain())
        except TypeError:  # dtype doesn't support get_classical_domain
            domain = list(range(domain_size))
        return domain[rng.integers(0, len(domain))]
    else:
        # Large domain: sample directly as an integer.
        # This works for QUInt, QInt, QAny — the most common large types.
        return int(rng.integers(0, domain_size))


def generate_input_vals(
    bloq: Bloq, n_samples: int, rng: np.random.Generator
) -> list[dict[str, ClassicalValT]]:
    """Generate classical input values for a bloq's LEFT registers.

    Uses `QDType.get_classical_domain()` to determine valid value ranges for
    each input register. If `n_samples` is >= the total domain size (product
    of all register domain sizes), switches to exhaustive enumeration.

    For registers with `shape` (multi-dimensional), each element is sampled
    independently from the domain.

    Args:
        bloq: The bloq whose signature determines the input types.
        n_samples: Number of random input samples to generate. If this exceeds
            the total domain size, exhaustive enumeration is used instead.
        rng: NumPy random generator for reproducible sampling.

    Returns:
        A list of dicts, each mapping LEFT register names to classical values.
    """
    left_regs = list(bloq.signature.lefts())

    if not left_regs:
        # No inputs — return a single empty dict.
        return [{}]

    # Compute per-register domain size (without materializing).
    domain_sizes: dict[str, int] = {}
    total_size: float = 1.0
    has_shaped = False

    for reg in left_regs:
        domain_sizes[reg.name] = _domain_size_for_register(reg)
        if reg.shape:
            has_shaped = True
            total_size = math.inf
        else:
            total_size *= domain_sizes[reg.name]

    if not has_shaped and n_samples >= total_size and math.isfinite(total_size):
        # Exhaustive enumeration: only when all registers are scalar and
        # total domain is small enough.
        domains = []
        for reg in left_regs:
            try:
                domains.append(list(reg.dtype.get_classical_domain()))
            except TypeError:  # dtype doesn't support get_classical_domain
                domains.append(list(range(domain_sizes[reg.name])))
        results = []
        for combo in itertools.product(*domains):
            vals = {r.name: v for r, v in zip(left_regs, combo)}
            results.append(vals)
        return results
    else:
        # Random sampling.
        results = []
        for _ in range(n_samples):
            sampled_vals: dict[str, ClassicalValT] = {}
            for reg in left_regs:
                if reg.shape:
                    n_elems = int(np.prod(reg.shape))
                    sampled_vals[reg.name] = np.array(
                        [_sample_from_register(reg, rng) for _ in range(n_elems)]
                    ).reshape(reg.shape)
                else:
                    sampled_vals[reg.name] = _sample_from_register(reg, rng)
            results.append(sampled_vals)
        return results


def _assert_results_equal(result_a: tuple, result_b: tuple, bloq: Bloq, input_vals: dict) -> None:
    """Assert two simulation results are equal, with informative error messages."""
    if len(result_a) != len(result_b):
        raise AssertionError(
            f"Result length mismatch for {bloq}: "
            f"on_classical_vals has {len(result_a)} outputs, "
            f"decomposition has {len(result_b)} outputs."
        )
    for i, (a, b) in enumerate(zip(result_a, result_b)):
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if not np.array_equal(a, b):
                raise AssertionError(
                    f"Mismatch for {bloq}, output index {i}.\n"
                    f"  Input: {input_vals}\n"
                    f"  on_classical_vals: {a}\n"
                    f"  decomposition:     {b}"
                )
        elif a != b:
            raise AssertionError(
                f"Mismatch for {bloq}, output index {i}.\n"
                f"  Input: {input_vals}\n"
                f"  on_classical_vals: {a}\n"
                f"  decomposition:     {b}"
            )


@attrs.frozen
class ClassicalSimTestCase:
    """A test case specification for classical simulation verification.

    Specifies a concrete bloq instance (with specific compile-time parameters
    like bitsizes) to be verified. Classical input values are generated
    automatically from the bloq's signature and QDType domains.

    Attributes:
        bloq: A concrete (non-symbolic) Bloq instance.
        name: Human-readable name for this test case.
    """

    bloq: Bloq
    name: str


TestCaseProvider = Callable[[], list[ClassicalSimTestCase]]
"""A zero-argument callable returning `list[ClassicalSimTestCase]` for one bloq class.

Classical verification works by cross-checking two independent computations of the same
function: a bloq's hand-written `on_classical_vals` (the reference) and the
result of decomposing the bloq and running each subbloq's `on_classical_vals`
(structural induction). If both sides agree on all inputs, the decomposition
is functionally correct. The structural induction framework walks the full decomposition tree via
BFS, so every subbloq is also verified.

## How to add verification for a bloq

1. **Implement `on_classical_vals`** on your bloq class. This is the reference
   implementation that the decomposition is checked against. It must be a
   direct, independent computation.

2. **Provide a decomposition** via `build_composite_bloq`.

3. **Write a `TestCaseProvider`** in the bloq's module that returns
   `ClassicalSimTestCase` instances for desired parameter
   combinations. Example:

       def _get_add_classical_sim_test_cases() -> list['ClassicalSimTestCase']:
           from qualtran.simulation.verification import ClassicalSimTestCase
           return [
               ClassicalSimTestCase(
                   bloq=Add(a_bitsize=a, b_bitsize=b),
                   name=f"Add(a={a}, b={b})",
               )
               for a, b in itertools.product([2, 3, 4], repeat=2)
           ]

4. **Register** the provider in the `qualtran/simulation/do-classical-verification.py` script
    keyed by a test suite alias and run with
    `python do-classical-verification.py --[your testsuite alias]`.
"""


def validate_test_cases(name: str, cases: list[ClassicalSimTestCase]) -> None:
    """Validate that all test cases from a provider target the same bloq class.

    Call this after invoking a :data:`TestCaseProvider` to verify the
    structural invariant: each provider must return test cases for exactly
    one bloq class.

    Args:
        name: The registry name of the provider (used in error messages).
        cases: The test cases returned by a provider function.

    Raises:
        ValueError: If ``cases`` contains bloqs from more than one class.
    """
    if not cases:
        return

    class_names = {tc.bloq.__class__.__name__ for tc in cases}
    if len(class_names) > 1:
        raise ValueError(
            f"Test case provider '{name}' returned test cases for multiple bloq "
            f"classes: {sorted(class_names)}. Each provider function must return "
            f"test cases for exactly one bloq class."
        )


@attrs.frozen
class VerificationResult:
    """The result of verifying one bloq.

    Attributes:
        bloq: The bloq that was verified.
        status: The verification outcome.
        n_inputs_checked: Number of input values tested.
        error_message: Description of the failure, if any.
        children: Bloqs discovered by decomposing this bloq.
    """

    bloq: Bloq
    status: VerificationStatus
    n_inputs_checked: int = 0
    error_message: str | None = None
    children: tuple[Bloq, ...] = ()


def _log_result(bloq: Bloq, result: VerificationResult, log: IO[str] | None) -> None:
    """Write a single tab-separated log record for a verification result.

    Columns: objectstring \t status \t n_inputs_checked \t error_message
    """
    if log is None:
        return
    objectstring = dump_objectstring(bloq)
    detail = result.error_message or ''
    log.write(f"{objectstring}\t{result.status}\t{result.n_inputs_checked}\t{detail}\n")
    log.flush()


def _verify_leaf_bloq(
    bloq: Bloq, input_vals_list: list[dict[str, ClassicalValT]]
) -> VerificationResult:
    """Verify a leaf bloq (no decomposition) by exercising `on_classical_vals`.

    Since there is no decomposition to cross-check against, we verify that
    `on_classical_vals` is implemented and does not crash on any generated
    input.
    """
    if not input_vals_list:
        return VerificationResult(bloq, VerificationStatus.LEAF_VERIFIED, n_inputs_checked=0)

    n_ok = 0
    implemented: bool | None = None  # None = not yet determined
    try:
        for input_vals in input_vals_list:
            try:
                out = bloq.on_classical_vals(**input_vals)
            except QCDTypeDomainError:
                # Domain-constrained input; skip but keep trying others.
                continue

            if out is NotImplemented:
                if n_ok > 0:
                    # Inconsistent: returned real values for earlier inputs.
                    raise MissingClassicalValsError(
                        f"on_classical_vals returned NotImplemented for "
                        f"inputs {input_vals} after succeeding on earlier inputs"
                    )
                implemented = False
                break

            implemented = True
            if not isinstance(out, dict):
                raise TypeError(f"on_classical_vals returned {type(out).__name__}, expected dict")
            n_ok += 1

        if implemented is False:
            return VerificationResult(
                bloq,
                VerificationStatus.LEAF_NO_CLASSICAL_VALS,
                error_message="Leaf bloq does not implement on_classical_vals.",
            )

        return VerificationResult(bloq, VerificationStatus.LEAF_VERIFIED, n_inputs_checked=n_ok)

    except Exception as e:  # pylint: disable=broad-exception-caught
        return VerificationResult(
            bloq, VerificationStatus.ERROR, error_message=f"Leaf verification failed: {e}"
        )


logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _get_qlt_fastsim_cls() -> Any | None:
    """Memoize attempting to import `rsqualtran.QLTFastsim` to avoid repetitive `sys.path` searches."""
    try:
        from rsqualtran import QLTFastsim  # type: ignore[import-untyped,import-not-found]

        return QLTFastsim
    except ImportError:
        return None


def _should_use_fastsim() -> bool:
    """Determine whether to use the Rust fastsim engine for classical verification.

    Checks the `QLT_USE_FASTSIM` environment variable first (`1`/`true` or
    `0`/`false`). If not set, checks whether `rsqualtran` can be imported and
    uses fastsim if available, otherwise falls back to Python.
    """
    env_val = os.environ.get('QLT_USE_FASTSIM')
    if env_val is not None:
        val = env_val.strip().lower()
        if val in ('1', 'true', 'yes', 'on'):
            cls = _get_qlt_fastsim_cls()
            if cls is not None:
                return True
            raise ImportError(
                "QLT_USE_FASTSIM is explicitly enabled in environment variables, "
                "but `rsqualtran` (`QLTFastsim`) could not be imported."
            )
        elif val in ('0', 'false', 'no', 'off'):
            return False
        else:
            raise ValueError(
                f"Invalid value for QLT_USE_FASTSIM environment variable: '{env_val}'. "
                f"Expected '1'/'0', 'true'/'false', or similar boolean string."
            )

    return _get_qlt_fastsim_cls() is not None


def _verify_decomposable_bloq(
    bloq: Bloq, input_vals_list: list[dict[str, ClassicalValT]], child_bloqs: tuple[Bloq, ...]
) -> VerificationResult:
    """Verify a decomposable bloq via structural induction.

    Cross-checks the parent bloq's `on_classical_vals` against
    `simulate_via_subbloq_classical_vals` (decompose the parent once, then
    run each sub-bloq's `on_classical_vals`).  If the parent does not
    implement `on_classical_vals`, reports `MISSING_CLASSICAL_VALS`.

    We call `on_classical_vals` directly, never `call_classically` or any
    method that falls back to recursive decomposition to avoid tautological
    comparisons.
    """
    use_fastsim = _should_use_fastsim()
    sim = None
    if use_fastsim:
        fastsim_cls = _get_qlt_fastsim_cls()
        if fastsim_cls is not None:
            try:
                sim = fastsim_cls.from_bloq(bloq)
            except Exception:  # pylint: disable=broad-exception-caught
                # Fall back to Python simulation for mock or custom test bloqs.
                sim = None

    try:
        n_checked = 0
        parent_has_classical_vals = True

        for input_vals in input_vals_list:
            # Try on_classical_vals on auto-generated inputs.
            # QCDTypeDomainError indicates an out-of-domain input and is skipped.
            # All non-domain errors propagate immediately as verification bugs.
            try:
                parent_out = bloq.on_classical_vals(**input_vals)
            except QCDTypeDomainError:
                continue

            if parent_out is NotImplemented:
                parent_has_classical_vals = False
                break

            if sim is not None:
                decomp_result = sim.call_classically(**input_vals)
            else:
                decomp_result = simulate_via_subbloq_classical_vals(bloq, **input_vals)

            right_regs = list(bloq.signature.rights())
            parent_tuple = tuple(parent_out[reg.name] for reg in right_regs)
            _assert_results_equal(parent_tuple, decomp_result, bloq, input_vals)
            n_checked += 1

        if not parent_has_classical_vals:
            return VerificationResult(
                bloq,
                VerificationStatus.MISSING_CLASSICAL_VALS,
                error_message='Bloq does not implement on_classical_vals.',
                children=child_bloqs,
            )
        elif n_checked == 0:
            return VerificationResult(
                bloq,
                VerificationStatus.NO_VALID_INPUTS,
                n_inputs_checked=0,
                error_message='on_classical_vals raised on all inputs; '
                'nothing was actually verified.',
                children=child_bloqs,
            )
        else:
            return VerificationResult(
                bloq, VerificationStatus.PASSED, n_inputs_checked=n_checked, children=child_bloqs
            )

    except MissingClassicalValsError as e:
        return VerificationResult(
            bloq,
            VerificationStatus.MISSING_CLASSICAL_VALS,
            error_message=str(e),
            children=child_bloqs,
        )

    except AssertionError as e:
        return VerificationResult(
            bloq, VerificationStatus.FAILED, error_message=str(e), children=child_bloqs
        )

    except Exception as e:  # pylint: disable=broad-exception-caught
        return VerificationResult(
            bloq,
            VerificationStatus.ERROR,
            error_message=f"{type(e).__name__}: {e}",
            children=child_bloqs,
        )


def verify_structural_induction(
    root_cases: Sequence[ClassicalSimTestCase],
    n_samples: int = 100,
    rng: np.random.Generator | None = None,
    log: IO[str] | None = sys.stdout,
) -> list[VerificationResult]:
    """BFS over the decomposition tree, verifying each bloq encountered.

    For each bloq in the traversal:

    1. Attempt to decompose it. If it can't decompose, it's a leaf node —
       verify that `on_classical_vals` can at least be called.
    2. Enqueue its sub-bloqs (callees) for subsequent verification.
    3. Generate input values (random or exhaustive).
    4. Cross-check: the parent bloq's `on_classical_vals` (its declared
       behavior) must agree with `simulate_via_subbloq_classical_vals`
       (decompose once, run each sub-bloq's `on_classical_vals`).

    This implements structural induction: if every bloq's `on_classical_vals`
    agrees with its decomposition, then the entire computation tree is
    functionally correct.

    Args:
        root_cases: Test cases specifying the root bloq(s) to start from.
        n_samples: Number of random input samples per bloq. If this exceeds
            the total domain size, exhaustive enumeration is used.
        rng: NumPy random generator. Defaults to a seeded generator for
            reproducibility.
        log: Writable text stream for the complete verification log.
            Defaults to `sys.stdout`. Pass an open file to capture
            the log, or `open(os.devnull, 'w')` to suppress output.

    Returns:
        A list of `VerificationResult` for every bloq encountered.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    use_fastsim = _should_use_fastsim()
    sim_style = (
        "QLTFastsim (Rust)" if use_fastsim else "Python (simulate_via_subbloq_classical_vals)"
    )
    sys.stderr.write(f"[verify_structural_induction] Simulation style: {sim_style}\n")
    logger.info("[verify_structural_induction] Simulation style: %s", sim_style)

    queue: deque[Bloq] = deque()
    visited: set[Bloq] = set()
    results: list[VerificationResult] = []
    vm = StandardQualtranArchitectureAgnosticVirtualMachine()

    for tc in root_cases:
        queue.append(tc.bloq)

    while queue:
        bloq = queue.popleft()

        # attrs @frozen classes have __eq__ and __hash__, so deduplication works.
        if bloq in visited:
            continue
        visited.add(bloq)

        # Generate input values.
        try:
            input_vals_list = generate_input_vals(bloq, n_samples, rng)
        except Exception as e:  # pylint: disable=broad-exception-caught
            result = VerificationResult(
                bloq, VerificationStatus.ERROR, error_message=f"Input generation failed: {e}"
            )
            _log_result(bloq, result, log)
            results.append(result)
            continue

        # Try to decompose.
        try:
            cbloq = bloq.decompose_bloq()
        except (DecomposeNotImplementedError, DecomposeTypeError):
            if vm.bloq_in_isa(bloq):
                # Valid leaf node: in the ISA, no decomposition needed.
                # Verify on_classical_vals is at least implemented and callable.
                result = _verify_leaf_bloq(bloq, input_vals_list)
            else:
                # Not in the ISA and no decomposition — this bloq should
                # decompose but doesn't. Flag it as a warning.
                result = VerificationResult(
                    bloq,
                    VerificationStatus.MISSING_DECOMPOSITION,
                    error_message="Bloq has no decomposition and is not in the ISA.",
                )
            _log_result(bloq, result, log)
            results.append(result)
            continue

        # Collect unique child bloqs from the decomposition.
        seen_children: set[Bloq] = set()
        child_bloqs_list: list[Bloq] = []
        for binst in cbloq.bloq_instances:
            if binst.bloq not in seen_children:
                seen_children.add(binst.bloq)
                child_bloqs_list.append(binst.bloq)
        child_bloqs = tuple(child_bloqs_list)

        # Enqueue callees (sub-bloqs from the decomposition) for BFS.
        for child_bloq in child_bloqs:
            if child_bloq not in visited:
                queue.append(child_bloq)

        result = _verify_decomposable_bloq(bloq, input_vals_list, child_bloqs)
        _log_result(bloq, result, log)
        results.append(result)

    return results


_STATUS_ICONS: dict[VerificationStatus, str] = {
    VerificationStatus.PASSED: '✓',
    VerificationStatus.FAILED: '✗',
    VerificationStatus.MISSING_DECOMPOSITION: '⊘',
    VerificationStatus.LEAF_VERIFIED: '●',
    VerificationStatus.LEAF_NO_CLASSICAL_VALS: '○',
    VerificationStatus.MISSING_CLASSICAL_VALS: '⚠',
    VerificationStatus.NO_VALID_INPUTS: '◇',
    VerificationStatus.ERROR: '⚡',
}

_STATUS_RANK: dict[VerificationStatus, int] = {
    VerificationStatus.PASSED: 0,
    VerificationStatus.LEAF_VERIFIED: 1,
    VerificationStatus.LEAF_NO_CLASSICAL_VALS: 2,
    VerificationStatus.MISSING_CLASSICAL_VALS: 3,
    VerificationStatus.NO_VALID_INPUTS: 4,
    VerificationStatus.MISSING_DECOMPOSITION: 5,
    VerificationStatus.ERROR: 6,
    VerificationStatus.FAILED: 7,
}


@attrs.frozen
class _TreeNode:
    """A node in the verification summary tree.

    Represents a single bloq class with aggregated verification results
    from all instances of that class.
    """

    class_name: str
    status: VerificationStatus
    n_inputs_checked: int
    error_message: str | None
    child_class_names: tuple[str, ...]


def print_verification_summary(
    results: list[VerificationResult], file: IO[str] = sys.stdout
) -> None:
    """Print a tree-view summary of the verification results.

    The decomposition DAG is displayed as a tree with increasing indentation
    for children. Since the decomposition graph is a DAG (not a tree), each
    bloq class is assigned to one parent for its primary display. If the same
    class appears as a descendant of another node, it is shown in parentheses
    indicating that its analysis is summarized elsewhere.

    Nodes are de-duplicated by bloq class name: multiple instances of the same
    class are collapsed into a single tree node.

    Args:
        results: The list of `VerificationResult` from
            `verify_structural_induction`.
        file: Writable text stream for the summary. Defaults to stdout.
    """

    def _write(msg: str = '') -> None:
        file.write(msg + '\n')

    # --- De-duplicate results by class name. ---------------------------------
    # For each class name, merge all VerificationResult instances into one
    # representative entry: aggregate n_inputs_checked, collect unique child
    # class names, pick the worst status, and keep the first error message.
    groups: OrderedDict[str, list[VerificationResult]] = OrderedDict()
    for r in results:
        cls_name = r.bloq.__class__.__name__
        groups.setdefault(cls_name, []).append(r)

    nodes: dict[str, _TreeNode] = {}
    for cls_name, group in groups.items():
        total_inputs = sum(r.n_inputs_checked for r in group)
        worst = max(group, key=lambda r: _STATUS_RANK.get(r.status, 99))
        # Collect unique child class names, preserving first-seen order.
        seen_child_names: dict[str, None] = {}
        for r in group:
            for child in r.children:
                child_cls = child.__class__.__name__
                if child_cls not in seen_child_names:
                    seen_child_names[child_cls] = None
        nodes[cls_name] = _TreeNode(
            class_name=cls_name,
            status=worst.status,
            n_inputs_checked=total_inputs,
            error_message=worst.error_message,
            child_class_names=tuple(seen_child_names),
        )

    # --- Build tree structure. -----------------------------------------------
    # Identify root class names: those not appearing as a child of any node.
    all_child_names: set[str] = set()
    for node in nodes.values():
        all_child_names.update(node.child_class_names)
    root_names = [name for name in nodes if name not in all_child_names]

    # Track which class names have been fully printed.
    printed: set[str] = set()

    def _node_line(node: _TreeNode) -> str:
        icon = _STATUS_ICONS.get(node.status, '?')
        parts = [f"{icon} {node.class_name} [{node.status}]"]
        if node.n_inputs_checked:
            parts.append(f"({node.n_inputs_checked} inputs)")
        if node.error_message:
            parts.append(f"-- {node.error_message}")
        return ' '.join(parts)

    def _print_tree(name: str, prefix: str, connector: str) -> None:
        node = nodes.get(name)

        if name in printed:
            # Omit leaf-verified nodes entirely — they are typically low-level
            # primitives that appear many times and add noise to the summary.
            if node is not None and node.status == VerificationStatus.LEAF_VERIFIED:
                return
            icon = _STATUS_ICONS.get(node.status, '?') if node else '?'
            _write(f"{prefix}{connector}({name}) [{icon} see above]")
            return

        printed.add(name)

        if node is not None:
            _write(f"{prefix}{connector}{_node_line(node)}")
            children = node.child_class_names
        else:
            _write(f"{prefix}{connector}{name} [no result]")
            children = ()

        for i, child_name in enumerate(children):
            is_last = i == len(children) - 1
            child_connector = '└── ' if is_last else '├── '
            child_prefix = prefix + ('    ' if connector in ('└── ', '') else '│   ')
            _print_tree(child_name, child_prefix, child_connector)

    _write()
    _write('=' * 70)
    _write('VERIFICATION SUMMARY (tree view)')
    _write('=' * 70)

    for i, root_name in enumerate(root_names):
        _print_tree(root_name, '', '')
        if i < len(root_names) - 1:
            _write()

    # Status totals.
    status_counts: Counter[VerificationStatus] = Counter(r.status for r in results)
    _write('-' * 70)
    for status, count in sorted(status_counts.items()):
        icon = _STATUS_ICONS.get(status, '?')
        _write(f"  {icon} {status}: {count}")
    _write(f"  Total: {len(results)}")
    _write('=' * 70)
