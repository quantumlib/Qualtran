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
from typing import Dict, Set, Tuple, TYPE_CHECKING

import cirq
import numpy as np
from attrs import frozen
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
from qualtran.resource_counting.generalizers import ignore_split_join

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class SwapWithZero(GateWithRegisters):
    """Swaps |Psi_0> with |Psi_x> if selection register stores index `x`.

    Implements the unitary U |x> |Psi_0> |Psi_1> ... |Psi_{n-1}> --> |x> |Psi_x> |Rest of Psi>.
    Note that the state of `|Rest of Psi>` is allowed to be anything and should not be depended
    upon.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis](https://arxiv.org/abs/1812.00954).
        Low, Kliuchnikov, Schaeffer. 2018.
    """

    selection_bitsize: int
    target_bitsize: int
    n_target_registers: int

    def __attrs_post_init__(self):
        assert self.n_target_registers <= 2**self.selection_bitsize

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (
            Register(
                'selection',
                BoundedQUInt(
                    bitsize=self.selection_bitsize, iteration_length=self.n_target_registers
                ),
            ),
        )

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return (
            Register('targets', QAny(bitsize=self.target_bitsize), shape=self.n_target_registers),
        )

    @cached_property
    def signature(self) -> Signature:
        return Signature([*self.selection_registers, *self.target_registers])

    def build_composite_bloq(
        self, bb: 'BloqBuilder', selection: Soquet, targets: NDArray[Soquet]
    ) -> Dict[str, 'SoquetT']:
        cswap_n = CSwapApprox(self.target_bitsize)
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
        selection = bb.split(selection)
        for j in range(self.selection_bitsize):
            for i in range(0, self.n_target_registers - 2**j, 2 ** (j + 1)):
                # The inner loop is executed at-most `N - 1` times, where `N:= len(target_regs)`.
                sel_i = self.selection_bitsize - j - 1
                selection[sel_i], targets[i], targets[i + 2**j] = bb.add(
                    cswap_n, ctrl=selection[sel_i], x=targets[i], y=targets[i + 2**j]
                )

        return {'selection': bb.join(selection), 'targets': targets}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        num_swaps = np.floor(
            sum([self.n_target_registers / (2 ** (j + 1)) for j in range(self.selection_bitsize)])
        )
        return {(CSwapApprox(self.target_bitsize), int(num_swaps))}

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@(r⇋0)"] * self.selection_bitsize
        for i in range(self.n_target_registers):
            wire_symbols += [f"swap_{i}"] * self.target_bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)


@bloq_example(generalizer=ignore_split_join)
def _swz() -> SwapWithZero:
    swz = SwapWithZero(selection_bitsize=8, target_bitsize=32, n_target_registers=4)
    return swz


@bloq_example(generalizer=ignore_split_join)
def _swz_small() -> SwapWithZero:
    # A small version on four bits.
    swz_small = SwapWithZero(selection_bitsize=3, target_bitsize=2, n_target_registers=2)
    return swz_small


_SWZ_DOC = BloqDocSpec(
    bloq_cls=SwapWithZero,
    import_line='from qualtran.bloqs.swap_network import SwapWithZero',
    examples=(_swz, _swz_small),
)
