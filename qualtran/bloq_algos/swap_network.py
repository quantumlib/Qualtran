from functools import cached_property
from typing import Dict, Sequence, TYPE_CHECKING

import cirq
import numpy as np
from attrs import frozen
from cirq_ft import MultiTargetCNOT, TComplexity
from numpy.typing import NDArray

from qualtran import Bloq, BloqBuilder, FancyRegister, FancyRegisters, Soquet, SoquetT

if TYPE_CHECKING:
    from qualtran.quantum_graph.classical_sim import ClassicalValT


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
    def registers(self) -> FancyRegisters:
        return FancyRegisters.build(ctrl=1, x=self.bitsize, y=self.bitsize)

    def cirq_decomposition(
        self, ctrl: Sequence[cirq.Qid], x: Sequence[cirq.Qid], y: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        """Write the decomposition as a cirq circuit.

        This method is taken from the cirq.GateWithRegisters implementation and is used
        to partially support `build_composite_bloq` by relying on this cirq implementation.
        """
        (ctrl,) = ctrl

        def g(q: cirq.Qid, adjoint=False) -> cirq.OP_TREE:
            yield [cirq.S(q), cirq.H(q)]
            yield cirq.T(q) ** (1 - 2 * adjoint)
            yield [cirq.H(q), cirq.S(q) ** -1]

        cnot_x_to_y = [cirq.CNOT(x, y) for x, y in zip(x, y)]
        cnot_y_to_x = [cirq.CNOT(y, x) for x, y in zip(x, y)]
        g_inv_on_y = [list(g(q, True)) for q in y]  # Uses len(target_y) T-gates
        g_on_y = [list(g(q)) for q in y]  # Uses len(target_y) T-gates

        yield [cnot_y_to_x, g_inv_on_y, cnot_x_to_y, g_inv_on_y]
        yield MultiTargetCNOT(len(y)).on(ctrl, *y)
        yield [g_on_y, cnot_x_to_y, g_on_y, cnot_y_to_x]

    def build_composite_bloq(
        self, bb: 'BloqBuilder', ctrl: 'SoquetT', x: 'SoquetT', y: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        from qualtran.quantum_graph.cirq_conversion import cirq_circuit_to_cbloq

        cirq_quregs = self.registers.get_cirq_quregs()
        cbloq = cirq_circuit_to_cbloq(cirq.Circuit(self.cirq_decomposition(**cirq_quregs)))

        # Split our registers to "flat" api from cirq circuit; add the circuit; join back up.
        qvars = np.concatenate(([ctrl], bb.split(x), bb.split(y)))
        (qvars,) = bb.add_from(cbloq, qubits=qvars)
        return {
            'ctrl': qvars[0],
            'x': bb.join(qvars[1 : self.bitsize + 1]),
            'y': bb.join(qvars[-self.bitsize :]),
        }

    def on_classical_vals(
        self, ctrl: 'ClassicalValT', x: 'ClassicalValT', y: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        if ctrl == 0:
            return {'ctrl': 0, 'x': x, 'y': y}
        if ctrl == 1:
            return {'ctrl': 1, 'x': y, 'y': x}
        raise ValueError("Bad control value for CSwap classical simulation.")

    def t_complexity(self) -> TComplexity:
        """T complexity as explained in Appendix B.2.c of https://arxiv.org/abs/1812.00954"""
        n = self.bitsize
        # 4 * n: G gates, each wth 1 T and 4 cliffords
        # 4 * n: CNOTs
        # 2 * n - 1: CNOTs from 1 MultiTargetCNOT
        return TComplexity(t=4 * n, clifford=22 * n - 1)

    def short_name(self) -> str:
        return '~swap'


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
                FancyRegister('targets', self.target_bitsize, shape=(self.n_target_registers,)),
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
