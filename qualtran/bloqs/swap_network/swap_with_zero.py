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

from functools import cached_property
from typing import cast, Dict, Iterable, Iterator, Tuple, TYPE_CHECKING, Union

import attrs
import cirq
import numpy as np
from numpy.typing import NDArray

from qualtran import (
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    BQUInt,
    GateWithRegisters,
    QAny,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox
from qualtran.drawing import TextBox, WireSymbol
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.symbolics import is_symbolic, prod, SymbolicInt

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


def _to_tuple(x: Union[SymbolicInt, Iterable[SymbolicInt]]) -> Tuple[SymbolicInt, ...]:
    if isinstance(x, np.ndarray):
        return _to_tuple(x.tolist())
    if isinstance(x, Iterable):
        return tuple(x)
    return (x,)


def _swap_with_zero_swap_sequence(
    selection_bitsizes: Tuple[int, ...], target_shape: Tuple[int, ...], idx: Tuple[int, ...] = ()
) -> Iterator[Tuple[int, int, Tuple[int, ...], Tuple[int, ...]]]:
    """Yields tuples of indices that should be swapped in that order to realize a swap with zero.

    The method recursively iterates over all combinations of `S = np.prod(selection_bitsizes)`
    indices, where each index is multidimensional with `n=len(selection_bitsizes)` dimensions.
    The method yields a sequence of O(S) pairs of target indices which should be swapped controlled
    on a qubit from one of the `n` selection registers being True.

    Args:
        selection_bitsizes: Bitsizes of the selection index.
        target_shape: Shape of the target register.
        idx: A tuple representing a prefix of n-dimensional selection index,
            used as part of the recursion.

    Yields:
        - i: Integer in range `[0, n)`, representing index of the selection register from which a qubit
            should be used as a control.
        - sel_idx: Little endian index of the qubit in the `i`'th selection register that should be
            used as a control for the swap.
        - idx_one: An n-dimensional tuple representing a unique selection index that should be swapped.
        - idx_two: An n-dimensional tuple representing a unique selection index that should be swapped.
    """
    if len(idx) == len(selection_bitsizes):
        return
    idx_len = len(idx)
    dim = target_shape[idx_len]
    for x in range(dim):
        # First make sure that value to be searched is present at the LEFT most position
        # of the composite index by recursively swapping the subtrees attached on leaf nodes of
        # the current segment tree.
        yield from _swap_with_zero_swap_sequence(selection_bitsizes, target_shape, idx + (x,))
    mid_point = 1
    for sel_idx in range(selection_bitsizes[idx_len]):
        # Imagine a complete binary tree of depth `logN` with `N` leaves, each denoting a target
        # register. If the selection register stores index `r`, we want to bring the value stored
        # in leaf indexed `r` to the leaf indexed `0`. At each node of the binary tree, the left
        # subtree contains node with current bit 0 and right subtree contains nodes with current
        # bit 1. Thus, leaf indexed `0` is the leftmost node in the tree.
        # Start iterating from the root of the tree. If the j'th bit is set in the selection
        # register (i.e. the control would be activated); we know that the value we are searching
        # for is in the right subtree. In order to (eventually) bring the desired value to node
        # 0; we swap all values in the right subtree with all values in the left subtree. This
        # takes (N / (2 ** (j + 1)) swaps at level `j`.
        # Therefore, in total, we need $\sum_{j=0}^{logN-1} \frac{N}{2 ^ {j + 1}}$ controlled swaps.
        for x in range(0, dim - mid_point, 2 * mid_point):
            zero_pad = (0,) * (len(selection_bitsizes) - idx_len - 1)
            idx_one = idx + (x,) + zero_pad
            idx_two = idx + (x + mid_point,) + zero_pad
            yield idx_len, sel_idx, idx_one, idx_two
        mid_point <<= 1


@attrs.frozen
class SwapWithZero(GateWithRegisters):
    r"""Swaps $|\Psi_0\rangle$ with $|\Psi_x\rangle$ if selection register stores index `x`.

    Implements the unitary
    $$
    U |x\rangle |\Psi_0\rangle |\Psi_1\rangle \dots \Psi_{M-1}\rangle \rightarrow
      |x\rangle |\Psi_x\rangle |\text{Rest of } \Psi\rangle$
    $$
    Note that the state of $|\text{Rest of } \Psi\rangle$ is allowed to be anything and
    should not be depended upon.

    Also supports the multidimensional version where $|x\rangle$ can be an n-dimensional index
    of the form $|x_1\rangle|x_2\rangle \dots |x_n\rangle$

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis](https://arxiv.org/abs/1812.00954).
        Low, Kliuchnikov, Schaeffer. 2018.
    """

    selection_bitsizes: Tuple[SymbolicInt, ...] = attrs.field(converter=_to_tuple)
    target_bitsize: SymbolicInt
    n_target_registers: Tuple[SymbolicInt, ...] = attrs.field(converter=_to_tuple)
    uncompute: bool = False

    def __attrs_post_init__(self):
        assert len(self.n_target_registers) == len(self.selection_bitsizes)

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        types = [
            BQUInt(sb, l)
            for sb, l in zip(self.selection_bitsizes, self.n_target_registers)
            if is_symbolic(sb) or sb > 0
        ]
        if len(types) == 1:
            return (Register('selection', types[0]),)
        return tuple(Register(f'selection{i}_', qdtype) for i, qdtype in enumerate(types))

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return (
            Register('targets', QAny(bitsize=self.target_bitsize), shape=self.n_target_registers),
        )

    @cached_property
    def signature(self) -> Signature:
        return Signature([*self.selection_registers, *self.target_registers])

    @cached_property
    def cswap_n(self) -> 'CSwapApprox':
        return CSwapApprox(self.target_bitsize)

    @cached_property
    def _swap_sequence(self) -> Tuple[Tuple[int, int, Tuple[int, ...], Tuple[int, ...]], ...]:
        if is_symbolic(*self.selection_bitsizes) or is_symbolic(*self.n_target_registers):
            raise ValueError(f"Cannot produce swap sequence for symbolic {self=}")
        selection_bitsizes = cast(Tuple[int, ...], self.selection_bitsizes)
        n_target_registers = cast(Tuple[int, ...], self.n_target_registers)
        ret = [*_swap_with_zero_swap_sequence(selection_bitsizes, n_target_registers)]
        return tuple(ret[::-1] if self.uncompute else ret)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', targets: NDArray['Soquet'], **sel: 'Soquet'  # type: ignore[type-var]
    ) -> Dict[str, 'SoquetT']:
        sel_soqs = [bb.split(sel[reg.name]) for reg in self.selection_registers]
        for i, sel_idx_small, idx_one, idx_two in self._swap_sequence:
            sel_idx_big = self.selection_bitsizes[i] - sel_idx_small - 1
            sel_soqs[i][sel_idx_big], targets[idx_one], targets[idx_two] = bb.add(
                self.cswap_n, ctrl=sel_soqs[i][sel_idx_big], x=targets[idx_one], y=targets[idx_two]
            )
        sel = {
            reg.name: bb.join(sel_soq) for sel_soq, reg in zip(sel_soqs, self.selection_registers)
        }
        return sel | {'targets': targets}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        num_swaps = prod(x for x in self.n_target_registers) - 1
        return {self.cswap_n: num_swaps}

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        from qualtran.cirq_interop._bloq_to_cirq import _wire_symbol_to_cirq_diagram_info

        return _wire_symbol_to_cirq_diagram_info(self, args)

    def wire_symbol(self, reg: Register, idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return super().wire_symbol(reg, idx)
        name = reg.name
        if 'selection' in name:
            return TextBox('@(r⇋0)')
        elif name == 'targets':
            subscript = "".join(f"_{i}" for i in idx)
            return TextBox(f'swap{subscript}')
        raise ValueError(f'Unrecognized register name {name}')

    def on_classical_vals(
        self, *, targets: 'ClassicalValT', **selection: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        assert isinstance(targets, np.ndarray)
        selection_idx = tuple(selection.values())
        for i, sel_idx_small, idx_one, idx_two in self._swap_sequence:
            if selection_idx[i] & (1 << sel_idx_small) != 0:
                targets[idx_one], targets[idx_two] = targets[idx_two], targets[idx_one]
        return selection | {'targets': targets}

    def adjoint(self) -> 'SwapWithZero':
        return attrs.evolve(self, uncompute=not self.uncompute)

    def __str__(self) -> str:
        return 'SwapWithZero†' if self.uncompute else 'SwapWithZero'


@bloq_example(generalizer=ignore_split_join)
def _swz() -> SwapWithZero:
    swz = SwapWithZero(selection_bitsizes=8, target_bitsize=32, n_target_registers=4)
    return swz


@bloq_example(generalizer=ignore_split_join)
def _swz_small() -> SwapWithZero:
    # A small version on four bits.
    swz_small = SwapWithZero(selection_bitsizes=3, target_bitsize=2, n_target_registers=2)
    return swz_small


@bloq_example(generalizer=ignore_split_join)
def _swz_multi_dimensional() -> SwapWithZero:
    swz_multi_dimensional = SwapWithZero(
        selection_bitsizes=(2, 2), target_bitsize=2, n_target_registers=(3, 4)
    )
    return swz_multi_dimensional


_SWZ_DOC = BloqDocSpec(bloq_cls=SwapWithZero, examples=(_swz, _swz_small, _swz_multi_dimensional))
