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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, Union

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

ClassicalValT = Union[int, np.integer, NDArray[np.integer]]


def _get_in_vals(
    binst: Union[DanglingT, BloqInstance], reg: Register, soq_assign: Dict[Soquet, ClassicalValT]
) -> ClassicalValT:
    """Pluck out the correct values from `soq_assign` for `reg` on `binst`."""
    if not reg.shape:
        return soq_assign[Soquet(binst, reg)]

    if reg.bitsize <= 8:
        dtype: Type = np.uint8
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
    binst: Union[DanglingT, BloqInstance],
    vals: Union[Dict[str, Union[sympy.Symbol, ClassicalValT]], Dict[str, ClassicalValT]],
    soq_assign: Union[
        Dict[Soquet, Union[sympy.Symbol, ClassicalValT]], Dict[Soquet, ClassicalValT]
    ],
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
            soq_assign[soq] = val  # type: ignore[assignment]

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
    signature: Signature,
    vals: Mapping[str, Union[sympy.Symbol, ClassicalValT]],
    binst_graph: nx.DiGraph,
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
    _update_assign_from_vals(signature.lefts(), LeftDangle, dict(vals), soq_assign)

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


def add_ints(a: int, b: int, *, num_bits: Optional[int] = None, is_signed: bool = False) -> int:
    r"""Performs addition modulo $2^\mathrm{num\_bits}$ of (un)signed in a reversible way.

    Addition of signed integers can result in an overflow. In most classical programming languages (e.g. C++)
    what happens when an overflow happens is left as an implementation detail for compiler designers. However,
    for quantum subtraction, the operation should be unitary and that means that the unitary of the bloq should
    be a permutation matrix.

    If we hold `a` constant then the valid range of values of $b \in [-2^{\mathrm{num\_bits}-1}, 2^{\mathrm{num\_bits}-1})$
    gets shifted forward or backward by `a`. To keep the operation unitary overflowing values wrap around. This is the same
    as moving the range $2^\mathrm{num\_bits}$ by the same amount modulo $2^\mathrm{num\_bits}$. That is add
    $2^{\mathrm{num\_bits}-1})$ before addition modulo and then remove it.

    Args:
        a: left operand of addition.
        b: right operand of addition.
        num_bits: optional num_bits. When specified addition is done in the interval [0, 2**num_bits) or
            [-2**(num_bits-1), 2**(num_bits-1)) based on the value of `is_signed`.
        is_signed: boolean whether the numbers are unsigned or signed ints. This value is only used when
            `num_bits` is provided.
    """
    c = a + b
    if num_bits is not None:
        N = 2**num_bits
        if is_signed:
            return (c + N // 2) % N - N // 2
        return c % N
    return c
