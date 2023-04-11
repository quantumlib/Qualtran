from typing import Any, Dict, Iterable, TypeVar

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

T = TypeVar('T')


def big_endian_bits_to_int_cirq(bits: Iterable[Any]) -> int:
    """Returns the big-endian integer specified by the given bits.
    Args:
        bits: Descending bits of the integer, with the 1s bit at the end.
    Returns:
        The integer.
    Examples:
        >>> cirq.big_endian_bits_to_int([0, 1])
        1
        >>> cirq.big_endian_bits_to_int([1, 0])
        2
        >>> cirq.big_endian_bits_to_int([0, 1, 0])
        2
        >>> cirq.big_endian_bits_to_int([1, 0, 0, 1, 0])
        18
    """
    result = 0
    for e in bits:
        result <<= 1
        if e:
            result |= 1
    return result


def big_endian_bits_to_int(bitstrings):
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


def int_to_bits_ref(x: int, w: int):
    return np.asarray([int(b) for b in f'{x:0{w}b}'])


def int_to_bits(x: int, w: int):
    assert x >= 0
    assert x.bit_length() <= 64
    mask = 2 ** np.arange(w - 1, 0 - 1, -1, dtype=np.uint64).reshape((1, w))
    return (x & mask).astype(bool).astype(int)[0]


def _get_in_vals(
    binst: BloqInstance, reg: FancyRegister, soq_assign: Dict[Soquet, NDArray[np.uint8]]
) -> NDArray[np.uint8]:
    """Pluck out the correct values from `soq_assign` for `reg` on `binst`."""
    # TODO: use for left dangle?

    if not reg.wireshape:
        return soq_assign[Soquet(binst, reg)]

    if reg.bitsize > 64:
        raise NotImplementedError("Come back later")

    arg = np.empty(reg.wireshape, dtype=np.uint64)
    for idx in reg.wire_idxs():
        soq = Soquet(binst, reg, idx=idx)
        arg[idx] = soq_assign[soq]

    return arg


def _binst_apply_classical(
    binst: BloqInstance,
    pred_cxns: Iterable[Connection],
    soq_assign: Dict[Soquet, NDArray[np.uint8]],
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

    # Use output
    for reg in bloq.registers.rights():
        arr = out_vals[reg.name]

        if reg.wireshape:
            arr = np.asarray(arr)
            for idx in reg.wire_idxs():
                soq = Soquet(binst, reg, idx=idx)
                soq_assign[soq] = arr[idx]
        else:
            soq = Soquet(binst, reg)
            soq_assign[soq] = arr


def _cbloq_apply_classical(
    registers: FancyRegisters, vals: Dict[str, NDArray], binst_graph: nx.DiGraph
) -> Dict[str, NDArray]:
    """Propogate `apply_classical` calls through a composite bloq's contents.

    Args:
        registers: The cbloq's registers for validating inputs
        vals: Mapping from register name to bit values
        binst_graph: The cbloq's binst graph.
    """

    # Keep track of each soquet's bit array.
    soq_assign: Dict[Soquet, NDArray[np.uint8]] = {}

    # LeftDangle assignment
    for reg in registers.lefts():
        arr = vals[reg.name]
        # arr = np.asarray(vals[reg.name]).astype(np.uint8, casting='safe', copy=False)
        # if arr.shape != reg.wireshape + (reg.bitsize,):
        #     raise ValueError(f"Classical values for {reg} are the wrong shape: {arr.shape}")

        if reg.wireshape:
            for idx in reg.wire_idxs():
                soq = Soquet(LeftDangle, reg, idx=idx)
                soq_assign[soq] = arr[idx]
        else:
            soq = Soquet(LeftDangle, reg)
            soq_assign[soq] = arr

    # Bloq-by-bloq application
    for binst in nx.topological_sort(binst_graph):
        if isinstance(binst, DanglingT):
            continue
        pred_cxns, succ_cxns = _binst_to_cxns(binst, binst_graph=binst_graph)
        _binst_apply_classical(binst, pred_cxns, soq_assign)

    # Track bloq-to-dangle name changes
    final_preds, _ = _binst_to_cxns(RightDangle, binst_graph=binst_graph)
    for cxn in final_preds:
        soq_assign[cxn.right] = soq_assign[cxn.left]

    # Formulate output with expected API
    def _f_vals(reg: FancyRegister):
        return _get_in_vals(RightDangle, reg, soq_assign)

    final_vals = {reg.name: _f_vals(reg) for reg in registers.rights()}
    return final_vals, soq_assign
