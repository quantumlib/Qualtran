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

"""Functionality for the `Bloq.call_classically(...)` protocol."""
import itertools
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import sympy
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    BloqInstance,
    Connection,
    DanglingT,
    LeftDangle,
    Register,
    RightDangle,
    Signature,
    Soquet,
)
from qualtran._infra.composite_bloq import _binst_to_cxns

ClassicalValT = Union[int, NDArray[np.integer]]


def bits_to_ints(bitstrings: Union[Sequence[int], NDArray[np.uint]]) -> NDArray[np.uint]:
    """Returns the integer specified by the given big-endian bitstrings.

    Args:
        bitstrings: A bitstring or array of bitstrings, each of which has the 1s bit (LSB) at the end.
    Returns:
        An array of integers; one for each bitstring.
    """
    bitstrings = np.atleast_2d(bitstrings)
    if bitstrings.shape[1] > 64:
        raise NotImplementedError()
    basis = 2 ** np.arange(bitstrings.shape[1] - 1, 0 - 1, -1, dtype=np.uint64)
    return np.sum(basis * bitstrings, axis=1)


def ints_to_bits(x: Union[int, Sequence[int], NDArray[np.uint]], w: int) -> NDArray[np.uint8]:
    """Returns the big-endian bitstrings specified by the given integers.

    Args:
        x: An integer or array of unsigned integers.
        w: The bit width of the returned bitstrings.
    """
    x = np.atleast_1d(x)
    if not np.issubdtype(x.dtype, np.uint):
        assert np.all(x >= 0)
        assert np.iinfo(x.dtype).bits <= 64
        x = x.astype(np.uint64)
    assert w <= np.iinfo(x.dtype).bits
    mask = 2 ** np.arange(w - 1, 0 - 1, -1, dtype=x.dtype).reshape((w, 1))
    return (x & mask).astype(bool).astype(np.uint8).T


def _get_in_vals(
    binst: BloqInstance, reg: Register, soq_assign: Dict[Soquet, ClassicalValT]
) -> ClassicalValT:
    """Pluck out the correct values from `soq_assign` for `reg` on `binst`."""
    if not reg.shape:
        return soq_assign[Soquet(binst, reg)]

    if reg.bitsize <= 8:
        dtype = np.uint8
    elif reg.bitsize <= 16:
        dtype = np.uint16
    elif reg.bitsize <= 32:
        dtype = np.uint32
    elif reg.bitsize <= 64:
        dtype = np.uint64
    else:
        raise NotImplementedError(
            "We currently only support up to 64-bit "
            "multi-dimensional registers in classical simulation."
        )

    arg = np.empty(reg.shape, dtype=dtype)
    for idx in reg.all_idxs():
        soq = Soquet(binst, reg, idx=idx)
        arg[idx] = soq_assign[soq]

    return arg


def _update_assign_from_vals(
    regs: Iterable[Register],
    binst: BloqInstance,
    vals: Dict[str, ClassicalValT],
    soq_assign: Dict[Soquet, ClassicalValT],
):
    """Update `soq_assign` using `vals`.

    This helper function is responsible for error checking. We use `regs` to make sure all the
    keys are present in the vals dictionary. We check the classical value shapes, types, and
    ranges.
    """
    for reg in regs:
        debug_str = f'{binst}.{reg.name}'
        try:
            val = vals[reg.name]
        except KeyError as e:
            raise ValueError(f"{binst} requires an input register named {reg.name}") from e

        if reg.shape:
            # `val` is an array
            val = np.asarray(val)
            if val.shape != reg.shape:
                raise ValueError(
                    f"Incorrect shape {val.shape} received for {debug_str}. " f"Want {reg.shape}."
                )
            reg.dtype.assert_valid_classical_val_array(val, debug_str)

            for idx in reg.all_idxs():
                soq = Soquet(binst, reg, idx=idx)
                soq_assign[soq] = val[idx]

        elif isinstance(val, sympy.Expr):
            # `val` is symbolic
            soq = Soquet(binst, reg)
            soq_assign[soq] = val

        else:
            # `val` is one value.
            reg.dtype.assert_valid_classical_val(val, debug_str)
            soq = Soquet(binst, reg)
            soq_assign[soq] = val


def _binst_on_classical_vals(
    binst: BloqInstance, pred_cxns: Iterable[Connection], soq_assign: Dict[Soquet, ClassicalValT]
):
    """Call `on_classical_vals` on a given binst.

    Args:
        binst: The bloq instance whose bloq we will call `on_classical_vals`.
        pred_cxns: Predecessor connections for the bloq instance.
        soq_assign: Current assignment of soquets to classical values.
    """

    # Track inter-Bloq name changes
    for cxn in pred_cxns:
        soq_assign[cxn.right] = soq_assign[cxn.left]

    def _in_vals(reg: Register):
        # close over binst and `soq_assign`
        return _get_in_vals(binst, reg, soq_assign=soq_assign)

    bloq = binst.bloq
    in_vals = {reg.name: _in_vals(reg) for reg in bloq.signature.lefts()}

    # Apply function
    out_vals = bloq.on_classical_vals(**in_vals)
    if not isinstance(out_vals, dict):
        raise TypeError(f"{bloq.__class__.__name__}.on_classical_vals should return a dictionary.")
    _update_assign_from_vals(bloq.signature.rights(), binst, out_vals, soq_assign)


def call_cbloq_classically(
    signature: Signature, vals: Dict[str, ClassicalValT], binst_graph: nx.DiGraph
) -> Tuple[Dict[str, ClassicalValT], Dict[Soquet, ClassicalValT]]:
    """Propagate `on_classical_vals` calls through a composite bloq's contents.

    While we're handling the plumbing, we also do error checking on the arguments; see
    `_update_assign_from_vals`.

    Args:
        signature: The cbloq's signature for validating inputs
        vals: Mapping from register name to classical values
        binst_graph: The cbloq's binst graph.

    Returns:
        final_vals: A mapping from register name to output classical values
        soq_assign: An assignment from each soquet to its classical value. Soquets
            corresponding to thru registers will be mapped to the *output* classical
            value.
    """
    # Keep track of each soquet's bit array. Initialize with LeftDangle
    soq_assign: Dict[Soquet, ClassicalValT] = {}
    _update_assign_from_vals(signature.lefts(), LeftDangle, vals, soq_assign)

    # Bloq-by-bloq application
    for binst in nx.topological_sort(binst_graph):
        if isinstance(binst, DanglingT):
            continue
        pred_cxns, succ_cxns = _binst_to_cxns(binst, binst_graph=binst_graph)
        _binst_on_classical_vals(binst, pred_cxns, soq_assign)

    # Track bloq-to-dangle name changes
    if len(list(signature.rights())) > 0:
        final_preds, _ = _binst_to_cxns(RightDangle, binst_graph=binst_graph)
        for cxn in final_preds:
            soq_assign[cxn.right] = soq_assign[cxn.left]

    # Formulate output with expected API
    def _f_vals(reg: Register):
        return _get_in_vals(RightDangle, reg, soq_assign)

    final_vals = {reg.name: _f_vals(reg) for reg in signature.rights()}
    return final_vals, soq_assign


def get_classical_truth_table(
    bloq: 'Bloq',
) -> Tuple[List[str], List[str], List[Tuple[Sequence[Any], Sequence[Any]]]]:
    """Get a 'truth table' for a classical-reversible bloq.

    Args:
        bloq: The classical-reversible bloq to create a truth table for.

    Returns:
        in_names: The names of the left, input registers to serve as truth table headings for
            the input side of the truth table.
        out_names: The names of the right, output registers to serve as truth table headings
            for the output side of the truth table.
        truth_table: A list of table entries. Each entry is a tuple of (in_vals, out_vals).
            The vals sequences are ordered according to the `in_names` and `out_names` return
            values.
    """
    for reg in bloq.signature.lefts():
        if reg.shape:
            raise NotImplementedError()

    in_names: List[str] = []
    iters = []
    for reg in bloq.signature.lefts():
        in_names.append(reg.name)
        iters.append(reg.dtype.get_classical_domain())
    out_names: List[str] = [reg.name for reg in bloq.signature.rights()]

    truth_table: List[Tuple[Sequence[Any], Sequence[Any]]] = []
    for in_val_tuple in itertools.product(*iters):
        in_val_d = {name: val for name, val in zip(in_names, in_val_tuple)}
        out_val_tuple = bloq.call_classically(**in_val_d)
        # out_val_d = {name: val for name, val in zip(out_names, out_val_tuple)}
        truth_table.append((in_val_tuple, out_val_tuple))
    return in_names, out_names, truth_table


def format_classical_truth_table(
    in_names: Sequence[str],
    out_names: Sequence[str],
    truth_table: Sequence[Tuple[Sequence[Any], Sequence[Any]]],
) -> str:
    """Get a formatted tabular representation of the classical truth table."""
    heading = '  '.join(in_names) + '  |  ' + '  '.join(out_names) + '\n'
    heading += '-' * len(heading)
    entries = [
        ', '.join(f'{v}' for v in invals) + ' -> ' + ', '.join(f'{v}' for v in outvals)
        for invals, outvals in truth_table
    ]
    return '\n'.join([heading] + entries)
