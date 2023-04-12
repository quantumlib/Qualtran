from typing import Dict, Iterable, Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from cirq_qubitization.quantum_graph.composite_bloq import _binst_to_cxns
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters
from cirq_qubitization.quantum_graph.quantum_graph import (
    BloqInstance,
    Connection,
    DanglingT,
    LeftDangle,
    RightDangle,
    Soquet,
)

ClassicalValT = Union[int, NDArray[int]]


def bits_to_ints(bitstrings):
    """Returns the big-endian integer specified by the given bits.
    Args:
        bits: Descending bits of the integer, with the 1s bit at the end.
    Returns:
        The integer.
    """
    bitstrings = np.atleast_2d(bitstrings)
    assert bitstrings.shape[1] <= 64
    basis = 2 ** np.arange(bitstrings.shape[1] - 1, 0 - 1, -1, dtype=np.uint64)
    return np.sum(basis * bitstrings, axis=1)


def ints_to_bits(x: NDArray[np.uint], w: int):
    assert np.issubdtype(x.dtype, np.uint)
    assert w <= np.iinfo(x.dtype).bits
    mask = 2 ** np.arange(w - 1, 0 - 1, -1, dtype=x.dtype).reshape((w, 1))
    return (x & mask).astype(bool).astype(np.uint8).T


def _get_in_vals(
    binst: BloqInstance, reg: FancyRegister, soq_assign: Dict[Soquet, ClassicalValT]
) -> ClassicalValT:
    """Pluck out the correct values from `soq_assign` for `reg` on `binst`."""
    if not reg.wireshape:
        return soq_assign[Soquet(binst, reg)]

    if reg.bitsize > 64:
        raise NotImplementedError("Come back later")

    arg = np.empty(reg.wireshape, dtype=np.uint64)
    for idx in reg.wire_idxs():
        soq = Soquet(binst, reg, idx=idx)
        arg[idx] = soq_assign[soq]

    return arg


def _update_assign_from_vals(
    regs: Iterable[FancyRegister],
    binst: BloqInstance,
    vals: Dict[str, ClassicalValT],
    soq_assign: Dict[Soquet, ClassicalValT],
):
    # TODO: note error checking happens here
    # TODO: check for positive values?
    for reg in regs:
        try:
            arr = vals[reg.name]
        except KeyError:
            raise ValueError(f"{binst} requires an input register named {reg.name}")

        if reg.wireshape:
            arr = np.asarray(arr)
            if arr.shape != reg.wireshape:
                raise ValueError(f"Incorrect shape {arr.shape} received for {binst}.{reg.name}")

            for idx in reg.wire_idxs():
                soq = Soquet(binst, reg, idx=idx)
                soq_assign[soq] = arr[idx]
        else:
            if not isinstance(arr, (int, np.uint)):
                raise ValueError(f"{binst}.{reg.name} should be an integer, not {arr!r}")
            soq = Soquet(binst, reg)
            soq_assign[soq] = arr


def _binst_apply_classical(
    binst: BloqInstance, pred_cxns: Iterable[Connection], soq_assign: Dict[Soquet, ClassicalValT]
):
    """Call `apply_classical` on a given binst."""

    # Track inter-Bloq name changes
    for cxn in pred_cxns:
        soq_assign[cxn.right] = soq_assign[cxn.left]

    def _in_vals(reg: FancyRegister):
        # close over binst and `soq_assign`
        return _get_in_vals(binst, reg, soq_assign=soq_assign)

    bloq = binst.bloq
    in_vals = {reg.name: _in_vals(reg) for reg in bloq.registers.lefts()}

    # Apply function
    out_vals = bloq.apply_classical(**in_vals)
    _update_assign_from_vals(bloq.registers.rights(), binst, out_vals, soq_assign)


def _cbloq_apply_classical(
    registers: FancyRegisters, vals: Dict[str, ClassicalValT], binst_graph: nx.DiGraph
) -> Tuple[Dict[str, ClassicalValT], Dict[Soquet, ClassicalValT]]:
    """Propagate `apply_classical` calls through a composite bloq's contents.

    Args:
        registers: The cbloq's registers for validating inputs
        vals: Mapping from register name to bit values
        binst_graph: The cbloq's binst graph.
    """
    # Keep track of each soquet's bit array. Initialize with LeftDangle
    soq_assign: Dict[Soquet, ClassicalValT] = {}
    _update_assign_from_vals(registers.lefts(), LeftDangle, vals, soq_assign)

    # Bloq-by-bloq application
    for binst in nx.topological_sort(binst_graph):
        if isinstance(binst, DanglingT):
            continue
        pred_cxns, succ_cxns = _binst_to_cxns(binst, binst_graph=binst_graph)
        _binst_apply_classical(binst, pred_cxns, soq_assign)

    # Track bloq-to-dangle name changes
    if len(list(registers.rights())) > 0:
        final_preds, _ = _binst_to_cxns(RightDangle, binst_graph=binst_graph)
        for cxn in final_preds:
            soq_assign[cxn.right] = soq_assign[cxn.left]

    # Formulate output with expected API
    def _f_vals(reg: FancyRegister):
        return _get_in_vals(RightDangle, reg, soq_assign)

    final_vals = {reg.name: _f_vals(reg) for reg in registers.rights()}
    return final_vals, soq_assign
