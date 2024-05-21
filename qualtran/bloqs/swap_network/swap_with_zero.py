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
from typing import Dict, Iterable, Set, Tuple, TYPE_CHECKING, Union

import attrs
import cirq
import numpy as np
from numpy.typing import NDArray

from qualtran import (
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    BoundedQUInt,
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
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


def _to_tuple(x: Union[SymbolicInt, Iterable[SymbolicInt]]) -> Tuple[SymbolicInt, ...]:
    if isinstance(x, np.ndarray):
        return _to_tuple(x.tolist())
    if isinstance(x, Iterable):
        return tuple(x)
    return (x,)


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

    def __attrs_post_init__(self):
        assert len(self.n_target_registers) == len(self.selection_bitsizes)

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        types = [
            BoundedQUInt(sb, l)
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

    def build_via_tree(
        self,
        bb: 'BloqBuilder',
        sel: Dict[str, 'Soquet'],
        targets: NDArray['Soquet'],  # type: ignore[type-var]
        idx: Tuple[int, ...],
    ) -> None:
        sel_idx = len(idx)
        if sel_idx == len(self.selection_bitsizes):
            return

        n_target_registers = self.n_target_registers[sel_idx]
        assert isinstance(n_target_registers, int)
        for i in range(n_target_registers):
            # First make sure that value to be searched is present at the LEFT most position
            # of the composite index by recursively swapping the subtrees attached on leaf nodes of
            # the current segment tree.
            self.build_via_tree(bb, sel, targets, idx + (i,))

        sel_reg = self.selection_registers[sel_idx]  # type: ignore[type-var]
        sel_soqs = bb.split(sel[sel_reg.name])
        sel_bitsize = self.selection_registers[sel_idx].bitsize
        for j in range(sel_bitsize):
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
            sel_i = sel_bitsize - j - 1
            for i in range(0, self.n_target_registers[sel_idx] - 2**j, 2 ** (j + 1)):
                zero_pad = (0,) * (len(self.n_target_registers) - len(idx) - 1)
                left_idx = idx + (i,) + zero_pad
                right_idx = idx + (i + 2**j,) + zero_pad
                sel_soqs[sel_i], targets[left_idx], targets[right_idx] = bb.add(
                    self.cswap_n, ctrl=sel_soqs[sel_i], x=targets[left_idx], y=targets[right_idx]
                )
        sel[sel_reg.name] = bb.join(sel_soqs, dtype=sel_reg.dtype)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', targets: NDArray['Soquet'], **sel: 'Soquet'  # type: ignore[type-var]
    ) -> Dict[str, 'SoquetT']:
        self.build_via_tree(bb, sel, targets, ())
        return sel | {'targets': targets}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        num_swaps = prod(*[x for x in self.n_target_registers]) - 1
        return {(CSwapApprox(self.target_bitsize), num_swaps)}

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


_SWZ_DOC = BloqDocSpec(
    bloq_cls=SwapWithZero,
    import_line='from qualtran.bloqs.swap_network import SwapWithZero',
    examples=(_swz, _swz_small, _swz_multi_dimensional),
)
