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
from pathlib import Path

from qualtran import Bloq, BloqError, CompositeBloq, DanglingT, LeftDangle, RightDangle, Side
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
