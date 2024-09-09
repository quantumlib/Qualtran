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
from typing import Dict, Iterator, Optional, Tuple, TYPE_CHECKING

import cirq
from attrs import frozen
from numpy.typing import NDArray

from qualtran import bloq_example, BloqDocSpec, GateWithRegisters, Register, Signature
from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.bookkeeping import ArbitraryClifford
from qualtran.bloqs.mcmt.multi_target_cnot import MultiTargetCNOT
from qualtran.drawing import Text, WireSymbol
from qualtran.resource_counting.generalizers import (
    cirq_to_bloqs,
    generalize_rotation_angle,
    ignore_cliffords,
    ignore_split_join,
)
from qualtran.symbolics import SymbolicInt

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class CSwapApprox(GateWithRegisters):
    r"""Approximately implements a multi-target controlled swap unitary using only $4n$ T-gates.

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

    bitsize: SymbolicInt

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(ctrl=1, x=self.bitsize, y=self.bitsize)

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]  # type: ignore[type-var]
    ) -> Iterator[cirq.OP_TREE]:
        ctrl, target_x, target_y = quregs['ctrl'], quregs['x'], quregs['y']

        def g(q: cirq.Qid, adjoint=False) -> cirq.ops.op_tree.OpTree:
            yield [cirq.S(q), cirq.H(q)]
            yield cirq.T(q) ** (1 - 2 * adjoint)
            yield [cirq.H(q), cirq.S(q) ** -1]

        cnot_x_to_y = [cirq.CNOT(x, y) for x, y in zip(target_x, target_y)]
        cnot_y_to_x = [cirq.CNOT(y, x) for x, y in zip(target_x, target_y)]
        g_inv_on_y = [list(g(q, True)) for q in target_y]  # Uses len(target_y) T-gates
        g_on_y = [list(g(q)) for q in target_y]  # Uses len(target_y) T-gates

        yield [cnot_y_to_x, g_inv_on_y, cnot_x_to_y, g_inv_on_y]
        yield MultiTargetCNOT(len(target_y)).on(*ctrl, *target_y)
        yield [g_on_y, cnot_x_to_y, g_on_y, cnot_y_to_x]

    def on_classical_vals(
        self, ctrl: 'ClassicalValT', x: 'ClassicalValT', y: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        if ctrl == 0:
            return {'ctrl': 0, 'x': x, 'y': y}
        if ctrl == 1:
            return {'ctrl': 1, 'x': y, 'y': x}
        raise ValueError("Bad control value for CSwap classical simulation.")

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('~swap')
        return super().wire_symbol(reg, idx)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        if not args.use_unicode_characters:
            return cirq.CircuitDiagramInfo(
                ("@(approx)",) + ("swap_x",) * self.bitsize + ("swap_y",) * self.bitsize
            )
        return cirq.CircuitDiagramInfo(
            ("@(approx)",) + ("×(x)",) * self.bitsize + ("×(y)",) * self.bitsize
        )

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        n = self.bitsize
        # 4 * n: G gates, each wth 1 T and 4 single qubit cliffords
        # 4 * n: CNOTs
        # 2 * n - 1: CNOTs from 1 MultiTargetCNOT
        return {TGate(): 4 * n, ArbitraryClifford(n=1): 16 * n, ArbitraryClifford(n=2): 6 * n - 1}


@bloq_example
def _approx_cswap_symb() -> CSwapApprox:
    # A symbolic version. The bitsize is the symbol 'n'.
    from sympy import sympify

    approx_cswap_symb = CSwapApprox(bitsize=sympify('n'))
    return approx_cswap_symb


@bloq_example(
    generalizer=[cirq_to_bloqs, ignore_split_join, ignore_cliffords, generalize_rotation_angle]
)
def _approx_cswap_small() -> CSwapApprox:
    # A small version on four bits.
    approx_cswap_small = CSwapApprox(bitsize=4)
    return approx_cswap_small


@bloq_example(
    generalizer=[cirq_to_bloqs, ignore_split_join, ignore_cliffords, generalize_rotation_angle]
)
def _approx_cswap_large() -> CSwapApprox:
    # A large version that swaps 64-bit registers.
    approx_cswap_large = CSwapApprox(bitsize=64)
    return approx_cswap_large


_APPROX_CSWAP_DOC = BloqDocSpec(
    bloq_cls=CSwapApprox, examples=(_approx_cswap_symb, _approx_cswap_small, _approx_cswap_large)
)
