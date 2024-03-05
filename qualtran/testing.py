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

import itertools
import traceback
from enum import Enum
from pathlib import Path
from typing import List, Tuple

from qualtran import (
    Bloq,
    BloqError,
    BloqExample,
    CompositeBloq,
    DanglingT,
    DecomposeNotImplementedError,
    DecomposeTypeError,
    LeftDangle,
    RightDangle,
    Side,
    Soquet,
)
from qualtran._infra.composite_bloq import _get_flat_dangling_soqs


def assert_registers_match_parent(bloq: Bloq) -> CompositeBloq:
    """Check that the registers following decomposition match those of the original bloq.

    This is a strict condition of the `decompose_bloq()` protocol. A decomposition is only
    valid if it takes exactly the same inputs and outputs.

    This returns the decomposed bloq for further checking.
    """
    cbloq = bloq.decompose_bloq()

    if bloq.signature != cbloq.signature:
        err = "Parent registers do not match registers"
        for reg, dreg in itertools.zip_longest(bloq.signature, cbloq.signature):
            if reg != dreg:
                raise BloqError(f'{err}: {reg} != {dreg}')

        raise BloqError(f'{err}: {bloq}')

    return cbloq


def assert_registers_match_dangling(cbloq: CompositeBloq):
    """Check that connections to LeftDangle and RightDangle match the declared registers.

    All Soquets must be consumed exactly once by a subsequent subbloq or connected explicitly
    to either `LeftDangle` or `RightDangle` to indicate the soquet's status as an input
    or output, respectively.
    """
    lefts = frozenset(_get_flat_dangling_soqs(cbloq.signature, right=False))
    seen_lefts = set()
    rights = frozenset(_get_flat_dangling_soqs(cbloq.signature, right=True))
    seen_rights = set()

    for cxn in cbloq.connections:
        if isinstance(cxn.left.binst, DanglingT):
            if cxn.left.binst is not LeftDangle:
                raise BloqError(
                    f"The left side of a connection is connected to a "
                    f"dangling type other than LeftDangle: {cxn}"
                )

            # cxn.left is LeftDangle
            if cxn.left not in lefts:
                raise BloqError(f"{cxn}'s LeftDangle does not match the registers of the bloq.")
            if cxn.left in seen_lefts:
                raise BloqError(f"{cxn}'s LeftDangle was already connected to something else!")

            seen_lefts.add(cxn.left)

        if isinstance(cxn.right.binst, DanglingT):
            if cxn.right.binst is not RightDangle:
                raise BloqError(
                    f"The right side of a connection is connected to a "
                    f"dangling type other than RightDangle: {cxn}"
                )

            # cxn.right is RightDangle
            if cxn.right not in rights:
                raise BloqError(f"{cxn}'s RightDangle does not match the registers of the bloq.")
            if cxn.right in seen_rights:
                raise BloqError(f"{cxn}'s RightDangle was already connected to something else!")

            seen_rights.add(cxn.right)


def assert_connections_compatible(cbloq: CompositeBloq):
    """Check that all connections are between compatible registers.

    We check that register bitsize are equal and that LEFT and RIGHT registers are only
    used as such.
    """
    for cxn in cbloq.connections:
        lr = cxn.left.reg
        rr = cxn.right.reg

        if lr.bitsize != rr.bitsize:
            raise BloqError(f"{cxn}'s bitsizes are incompatible: {lr} -> {rr}")

        # Check the left side of the connection relative to the `Register.side`.
        if cxn.left.binst is LeftDangle:
            lr_side_should_be = Side.LEFT
        else:
            # internal connection -- left side should be output from a RIGHT register
            lr_side_should_be = Side.RIGHT

        if not (lr.side & lr_side_should_be):
            raise BloqError(f"{cxn}'s left side is associated with a register with side {lr.side}")

        # And the right side
        if cxn.right.binst is RightDangle:
            rr_side_should_be = Side.RIGHT
        else:
            # internal connection -- right side should input into a LEFT register
            rr_side_should_be = Side.LEFT
        if not (rr.side & rr_side_should_be):
            raise BloqError(f"{cxn}'s right side is associated with a register with side {rr.side}")


def assert_soquets_belong_to_registers(cbloq: CompositeBloq):
    """Check that all soquet's registers make sense.

    We check that any indexed soquets fit within the bounds of the register and that the
    register actually exists on the bloq.
    """
    for soq in cbloq.all_soquets:
        reg = soq.reg

        if len(soq.idx) != len(reg.shape):
            raise BloqError(f"{soq} has an idx of the wrong shape for {reg}")

        for soq_i, reg_max in zip(soq.idx, reg.shape):
            if soq_i >= reg_max:
                raise BloqError(f"{soq}'s index exceeds the bounds provided by {reg}'s shape.")

        if isinstance(soq.binst, DanglingT):
            continue

        if soq.reg not in soq.binst.bloq.signature:
            raise BloqError(f"{soq}'s register doesn't exist on its bloq {soq.binst.bloq}")


def assert_soquets_used_exactly_once(cbloq: CompositeBloq):
    """Check that all soquets are used once and only once.

    Each bloq's register produces prod(reg.shape) soquets which must be consumed
    once and only once.
    """
    produced = set()
    consumed = set()
    for cxn in cbloq.connections:
        if cxn.left in produced:
            raise BloqError(f"{cxn}'s left side had already been produced by a different bloq.")
        produced.add(cxn.left)

        if cxn.right in consumed:
            raise BloqError(f"{cxn}'s right side had already been consumed by a different bloq")
        consumed.add(cxn.right)

    diff1 = produced - cbloq.all_soquets
    if diff1:
        raise BloqError(f"Some soquets were not consumed: {diff1}")
    diff2 = consumed - cbloq.all_soquets
    if diff2:
        raise BloqError(f"Some soquets were not produced: {diff2}")


def assert_valid_cbloq(cbloq: CompositeBloq):
    """Perform all composite-bloq validity assertions."""
    assert_registers_match_dangling(cbloq)
    assert_connections_compatible(cbloq)
    assert_soquets_belong_to_registers(cbloq)
    assert_soquets_used_exactly_once(cbloq)


def assert_valid_bloq_decomposition(bloq: Bloq) -> CompositeBloq:
    """Check the validity of a bloq decomposition.

    Importantly, this does not do any correctness checking -- for that you likely
    need to use the simulation utilities provided by the library.

    This returns the decomposed, composite bloq on which you can do further testing.
    """
    cbloq = assert_registers_match_parent(bloq)
    assert_valid_cbloq(cbloq)
    return cbloq


def assert_wire_symbols_match_expected(bloq: Bloq, expected_ws: List[str]):
    """Assert a bloq's wire symbols match the expected ones.

    Args:
        bloq: the bloq whose wire symbols we want to check.
        expected_ws: A list of the expected wire symbols.
    """
    ws = []
    regs = bloq.signature
    for i, r in enumerate(regs):
        # note this will only work if shape = ().
        # See: https://github.com/quantumlib/Qualtran/issues/608
        ws.append(bloq.wire_symbol(Soquet(i, r)).text)

    assert ws == expected_ws


def execute_notebook(name: str):
    """Execute a jupyter notebook in the caller's directory.

    Args:
        name: The name of the notebook without extension.

    """
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    # Assumes that the notebook is in the same path from where the function was called,
    # which may be different from `__file__`.
    notebook_path = Path(traceback.extract_stack()[-2].filename).parent / f"{name}.ipynb"
    with notebook_path.open() as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb)


class BloqCheckResult(Enum):
    """The status result of the `check_bloq_example_xxx` functions."""

    PASS = 0
    """The check passed and is an unqualified success."""

    FAIL = 1
    """The check failed with a broken assertion or invariant."""

    MISSING = 2
    """The check did not pass because the required functionality is missing."""

    NA = 3
    """The check is not applicable in the current context."""

    UNVERIFIED = 4
    """The bloq protocol has provided a value, but some functionality is missing so we can't
    verify the result."""

    ERROR = 5
    """An unexpected error occurred during execution of the check."""


class BloqCheckException(AssertionError):
    """An exception raised by the `assert_bloq_example_xxx` functions in this module.

    These exceptions correspond to known failures due to assertion errors, non-applicable checks,
    or unverified protocols.

    Consider using the factory class methods `BloqCheckException.{fail, missing, na, unverified}`
    for convenience.

    Args:
        check_result: The BloqCheckResult.
        msg: A message providing details for the exception.
    """

    def __init__(self, check_result: BloqCheckResult, msg: str):
        super().__init__(msg)
        self._check_result = check_result
        self._msg = msg

    @property
    def check_result(self) -> BloqCheckResult:
        """The BloqCheckResult."""
        return self._check_result

    @property
    def msg(self) -> str:
        """A message providing details for the exception."""
        return self._msg

    @classmethod
    def fail(cls, msg: str) -> 'BloqCheckException':
        """Create an exception with a FAIL check result."""
        return cls(BloqCheckResult.FAIL, msg=msg)

    @classmethod
    def missing(cls, msg: str) -> 'BloqCheckException':
        """Create an exception with a MISSING check result."""
        return cls(BloqCheckResult.MISSING, msg=msg)

    @classmethod
    def na(cls, msg: str) -> 'BloqCheckException':
        """Create an exception with a NA check result."""
        return cls(BloqCheckResult.NA, msg=msg)

    @classmethod
    def unverified(cls, msg: str) -> 'BloqCheckException':
        """Create an exception with an UNVERIFIED check result."""
        return cls(BloqCheckResult.UNVERIFIED, msg=msg)


def assert_bloq_example_make(bloq_ex: BloqExample) -> None:
    """Assert that the BloqExample returns the desired bloq.

    Returns:
        None if the assertions are satisfied.

    Raises:
        BloqCheckException if any assertions are violated.
    """
    bloq = bloq_ex.make()
    if not isinstance(bloq, Bloq):
        raise BloqCheckException.fail(f'{bloq} is not an instance of Bloq')
    if not isinstance(bloq, bloq_ex.bloq_cls):
        raise BloqCheckException.fail(f'{bloq} is not an instance of {bloq_ex.bloq_cls}')
    return


def check_bloq_example_make(bloq_ex: BloqExample) -> Tuple[BloqCheckResult, str]:
    """Check that the BloqExample returns the desired bloq.

    Returns:
        result: The `BloqCheckResult`.
        msg: A message providing details from the check.
    """
    try:
        assert_bloq_example_make(bloq_ex)
    except BloqCheckException as bce:
        return bce.check_result, bce.msg
    except Exception as e:  # pylint: disable=broad-except
        return BloqCheckResult.ERROR, f'{bloq_ex.name}: {e}'

    return BloqCheckResult.PASS, ''


def assert_bloq_example_decompose(bloq_ex: BloqExample) -> None:
    """Assert that the BloqExample has a valid decomposition.

    This will use `assert_valid_decomposition` which has a variety of sub-checks. A failure
    in any of those checks will result in `FAIL`. `DecomposeTypeError` results in a
    not-applicable `NA` status. `DecomposeNotImplementedError` results in a `MISSING` status.

    Returns:
        None if the assertions are satisfied.

    Raises:
        BloqCheckException if any assertions are violated, not applicable, or missing.
    """
    try:
        bloq = bloq_ex.make()
        assert_valid_bloq_decomposition(bloq)
        return
    except DecomposeTypeError as e:
        raise BloqCheckException.na(str(e)) from e
    except DecomposeNotImplementedError as e:
        raise BloqCheckException.missing(str(e)) from e
    except BloqError as e:
        raise BloqCheckException.fail(str(e)) from e


def check_bloq_example_decompose(bloq_ex: BloqExample) -> Tuple[BloqCheckResult, str]:
    """Check that the BloqExample has a valid decomposition.

    This will use `assert_valid_decomposition` which has a variety of sub-checks. A failure
    in any of those checks will result in `FAIL`. `DecomposeTypeError` results in a
    not-applicable `NA` status. `DecomposeNotImplementedError` results in a `MISSING` status.

    Returns:
        result: The `BloqCheckResult`.
        msg: A message providing details from the check.
    """
    try:
        assert_bloq_example_decompose(bloq_ex)
    except BloqCheckException as bce:
        return bce.check_result, bce.msg
    except Exception as e:  # pylint: disable=broad-except
        return BloqCheckResult.ERROR, f'{bloq_ex.name}: {e}'

    return BloqCheckResult.PASS, ''


def assert_equivalent_bloq_example_counts(bloq_ex: BloqExample) -> None:
    """Assert that the BloqExample has consistent bloq counts.

    Bloq counts can be annotated directly via the `Bloq.build_call_graph` override.
    They can be inferred from a bloq's decomposition. This function asserts that both
    data sources are present and that they produce the same values.

    If both sources are present, and they disagree, that results in a `FAIL`. If only one source
    is present, an `UNVERIFIED` exception is raised. If neither are present, a `MISSING` result
    is raised.

    Returns:
        None if the assertions are satisfied.

    Raises:
        BloqCheckException if any assertions are violated or unverifiable due to partial
        or missing information.
    """
    bloq = bloq_ex.make()
    generalizer = bloq_ex.generalizer

    has_manual_counts: bool
    has_decomp_counts: bool
    manual_counts = None
    decomp_counts = None

    # Notable implementation detail: since `bloq.build_call_graph` has a default fallback
    # that uses the decomposition, we could accidentally be comparing two identical code paths
    # which isn't much of a check at all.
    #
    # To determine whether we have an independent source of bloq counts, we test whether
    # the `build_call_graph` method was overriden or not. This is not foolproof! The override
    # could itself rely on the decomposition, and we wouldn't actually have two independent sources
    # of data to compare against each other.
    if bloq.build_call_graph.__qualname__.startswith('Bloq.'):
        has_manual_counts = False
    else:
        has_manual_counts = True
        manual_counts = bloq.bloq_counts(generalizer=generalizer)
        if manual_counts == {bloq: 1}:
            has_manual_counts = False

    try:
        cbloq = bloq.decompose_bloq()
        decomp_counts = cbloq.bloq_counts(generalizer=generalizer)
        has_decomp_counts = True
    except (DecomposeTypeError, DecomposeNotImplementedError):
        has_decomp_counts = False

    if (not has_decomp_counts) and (not has_manual_counts):
        raise BloqCheckException.missing('No block counts')

    if has_manual_counts and has_decomp_counts:
        if manual_counts == decomp_counts:
            return
        else:
            msg = [f'{bloq_ex.name} does not have equivalent bloq counts.']
            only_manual = set(manual_counts.keys()) - set(decomp_counts.keys())
            if only_manual:
                msg.append(f"Bloq's missing from decomposition: {only_manual}")
            only_decomp = set(decomp_counts.keys()) - set(manual_counts.keys())
            if only_decomp:
                msg.append(f"Bloq's missing from annotation: {only_decomp}")
            msg.append(f'Annotation: {manual_counts}')
            msg.append(f'Decomp:     {decomp_counts}')
            raise BloqCheckException.fail('\n'.join(msg))

    assert has_manual_counts or has_decomp_counts
    if has_manual_counts:
        raise BloqCheckException.unverified(f'{bloq_ex.name} only has counts from build_call_graph')
    if has_decomp_counts:
        raise BloqCheckException.unverified(f'{bloq_ex.name} only has counts from decomposition.')


def check_equivalent_bloq_example_counts(bloq_ex: BloqExample) -> Tuple[BloqCheckResult, str]:
    """Check that the BloqExample has consistent bloq counts.

    Bloq counts can be annotated directly via the `Bloq.build_call_graph` override.
    They can be inferred from a bloq's decomposition. This function checks that both
    data sources are present and that they produce the same values.

    If both sources are present, and they disagree, that results in a `FAIL`. If only one source
    is present, an `UNVERIFIED` result is returned. If neither are present, a `MISSING` result
    is returned.

    Returns:
        result: The `BloqCheckResult`.
        msg: A message providing details from the check.
    """
    try:
        assert_equivalent_bloq_example_counts(bloq_ex)
    except BloqCheckException as bce:
        return bce.check_result, bce.msg
    except Exception as e:  # pylint: disable=broad-except
        return BloqCheckResult.ERROR, f'{bloq_ex.name}: {e}'

    return BloqCheckResult.PASS, ''
