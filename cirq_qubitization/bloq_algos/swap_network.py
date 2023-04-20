from functools import cached_property
from typing import Dict

import cirq
from attrs import frozen
from numpy.typing import NDArray

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder, SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters
from cirq_qubitization.quantum_graph.quantum_graph import Soquet
from cirq_qubitization.t_complexity_protocol import t_complexity, TComplexity


@frozen
class SwapWithZero(Bloq):
    selection_bitsize: int
    target_bitsize: int
    n_target_registers: int

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters(
            [
                FancyRegister('selection', self.selection_bitsize),
                FancyRegister('targets', self.target_bitsize, wireshape=(self.n_target_registers,)),
            ]
        )

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', selection: Soquet, targets: NDArray[Soquet]
    ) -> Dict[str, 'SoquetT']:
        cswap_n = MultiTargetCSwap(self.target_bitsize)
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


@frozen
class CSwap(Bloq):
    def short_name(self) -> str:
        return 'swap'

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters.build(ctrl=1, x=1, y=1)

    def as_cirq_op(self, cirq_quregs: Dict[str, 'NDArray[cirq.Qid]']) -> 'cirq.Operation':
        (ctrl,) = cirq_quregs['ctrl']
        (x,) = cirq_quregs['x']
        (y,) = cirq_quregs['y']
        return cirq.CSWAP.on(ctrl, x, y)

    def t_complexity(self) -> 'TComplexity':
        # https://arxiv.org/abs/1308.4134
        return TComplexity(t=7, clifford=10)


@frozen
class MultiTargetCSwap(Bloq):
    bitsize: int

    def short_name(self) -> str:
        return 'swap'

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters(
            [
                FancyRegister('ctrl', 1),
                FancyRegister('x', self.bitsize),
                FancyRegister('y', self.bitsize),
            ]
        )

    def t_complexity(self) -> 'TComplexity':
        # TODO: this is for approx
        """TComplexity as explained in Appendix B.2.c of https://arxiv.org/abs/1812.00954"""
        n = self.bitsize
        # 4 * n: G gates, each wth 1 T and 4 cliffords
        # 4 * n: CNOTs
        # 2 * n - 1: CNOTs from 1 MultiTargetCNOT
        return TComplexity(t=4 * n, clifford=22 * n - 1)

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', ctrl: 'SoquetT', x: 'SoquetT', y: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        xs = bb.split(x)
        ys = bb.split(y)

        for i in range(self.bitsize):
            ctrl, xs[i], ys[i] = bb.add(CSwap(), ctrl=ctrl, x=xs[i], y=ys[i])

        return {'ctrl': ctrl, 'x': bb.join(xs), 'y': bb.join(ys)}
