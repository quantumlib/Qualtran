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
from typing import Dict, Tuple, TYPE_CHECKING, Union

import cirq
from attrs import frozen
from cirq_ft import MultiTargetCSwapApprox
from numpy.typing import NDArray

from qualtran import Bloq, BloqBuilder, Register, Signature, Soquet, SoquetT
from qualtran.cirq_interop import decompose_from_cirq_op

if TYPE_CHECKING:
    from qualtran import CompositeBloq
    from qualtran.cirq_interop import CirqQuregT
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class CSwapApprox(Bloq):
    r"""Approximately implements a multi-target controlled swap unitary using only 4 * n T-gates.

    Implements $\mathrm{CSWAP}_n = |0 \rangle\langle 0| I + |1 \rangle\langle 1| \mathrm{SWAP}_n$
    such that the output state is correct up to a global phase factor of +1 / -1.

    This is useful when the incorrect phase can be absorbed in a garbage state of an algorithm
    and thus ignored. See the reference for more details.

    Args:
        bitsize: The bitsize of the two registers being swapped.

    Registers:
        ctrl: the control bit
        x: the first register
        y: the second register

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis](https://arxiv.org/abs/1812.00954).
        Low et. al. 2018. See Appendix B.2.c.
    """

    bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(ctrl=1, x=self.bitsize, y=self.bitsize)

    def decompose_bloq(self) -> 'CompositeBloq':
        return decompose_from_cirq_op(self)

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', **cirq_quregs: 'CirqQuregT'
    ) -> Tuple[Union['cirq.Operation', None], Dict[str, 'CirqQuregT']]:
        (ctrl,) = cirq_quregs['ctrl']
        x = cirq_quregs['x'].tolist()
        y = cirq_quregs['y'].tolist()
        return (
            MultiTargetCSwapApprox(self.bitsize).on_registers(control=ctrl, target_x=x, target_y=y),
            cirq_quregs,
        )

    def on_classical_vals(
        self, ctrl: 'ClassicalValT', x: 'ClassicalValT', y: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        if ctrl == 0:
            return {'ctrl': 0, 'x': x, 'y': y}
        if ctrl == 1:
            return {'ctrl': 1, 'x': y, 'y': x}
        raise ValueError("Bad control value for CSwap classical simulation.")

    def short_name(self) -> str:
        return '~swap'


@frozen
class SwapWithZero(Bloq):
    selection_bitsize: int
    target_bitsize: int
    n_target_registers: int

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('selection', self.selection_bitsize),
                Register('targets', self.target_bitsize, shape=(self.n_target_registers,)),
            ]
        )

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
