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

"""Functionality for moving data between registers (swapping)."""

from functools import cached_property
from typing import Dict, Set, Tuple, TYPE_CHECKING

import cirq
import numpy as np
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import (
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    GateWithRegisters,
    Register,
    SelectionRegister,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran._infra.gate_with_registers import total_bits
from qualtran.bloqs.basic_gates import CSwap, TGate
from qualtran.bloqs.multi_control_multi_target_pauli import MultiTargetCNOT
from qualtran.bloqs.unary_iteration_bloq import UnaryIterationGate
from qualtran.bloqs.util_bloqs import ArbitraryClifford
from qualtran.cirq_interop.t_complexity_protocol import TComplexity

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
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

    bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(ctrl=1, x=self.bitsize, y=self.bitsize)

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
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

    def short_name(self) -> str:
        return '~swap'

    def _t_complexity_(self) -> TComplexity:
        """TComplexity as explained in Appendix B.2.c of https://arxiv.org/abs/1812.00954"""
        n = self.bitsize
        # 4 * n: G gates, each wth 1 T and 4 single qubit cliffords
        # 4 * n: CNOTs
        # 2 * n - 1: CNOTs from 1 MultiTargetCNOT
        return TComplexity(t=4 * n, clifford=22 * n - 1)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        if not args.use_unicode_characters:
            return cirq.CircuitDiagramInfo(
                ("@(approx)",) + ("swap_x",) * self.bitsize + ("swap_y",) * self.bitsize
            )
        return cirq.CircuitDiagramInfo(
            ("@(approx)",) + ("×(x)",) * self.bitsize + ("×(y)",) * self.bitsize
        )

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        n = self.bitsize
        # 4 * n: G gates, each wth 1 T and 4 single qubit cliffords
        # 4 * n: CNOTs
        # 2 * n - 1: CNOTs from 1 MultiTargetCNOT
        return {
            (TGate(), 4 * n),
            (ArbitraryClifford(n=1), 16 * n),
            (ArbitraryClifford(n=2), 6 * n - 1),
        }


@bloq_example
def _approx_cswap_symb() -> CSwapApprox:
    # A symbolic version. The bitsize is the symbol 'n'.
    from sympy import sympify

    approx_cswap_symb = CSwapApprox(bitsize=sympify('n'))
    return approx_cswap_symb


@bloq_example
def _approx_cswap_small() -> CSwapApprox:
    # A small version on four bits.
    approx_cswap_small = CSwapApprox(bitsize=4)
    return approx_cswap_small


@bloq_example
def _approx_cswap_large() -> CSwapApprox:
    # A large version that swaps 64-bit registers.
    approx_cswap_large = CSwapApprox(bitsize=64)
    return approx_cswap_large


_APPROX_CSWAP_DOC = BloqDocSpec(
    bloq_cls=CSwapApprox,
    import_line='from qualtran.bloqs.swap_network import CSwapApprox',
    examples=(_approx_cswap_symb, _approx_cswap_small, _approx_cswap_large),
)


@frozen
class SwapWithZero(GateWithRegisters):
    """Swaps |Psi_0> with |Psi_x> if selection register stores index `x`.

    Implements the unitary U |x> |Psi_0> |Psi_1> ... |Psi_{n-1}> --> |x> |Psi_x> |Rest of Psi>.
    Note that the state of `|Rest of Psi>` is allowed to be anything and should not be depended
    upon.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
        (https://arxiv.org/abs/1812.00954).
            Low, Kliuchnikov, Schaeffer. 2018.
    """

    selection_bitsize: int
    target_bitsize: int
    n_target_registers: int

    def __attrs_post_init__(self):
        assert self.n_target_registers <= 2**self.selection_bitsize

    @cached_property
    def selection_registers(self) -> Tuple[SelectionRegister, ...]:
        return (
            SelectionRegister(
                'selection',
                bitsize=self.selection_bitsize,
                iteration_length=self.n_target_registers,
            ),
        )

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return (Register('targets', bitsize=self.target_bitsize, shape=self.n_target_registers),)

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


@bloq_example
def _swz() -> SwapWithZero:
    swz = SwapWithZero(selection_bitsize=8, target_bitsize=32, n_target_registers=4)
    return swz


@bloq_example
def _swz_small() -> SwapWithZero:
    # A small version on four bits.
    swz_small = SwapWithZero(selection_bitsize=3, target_bitsize=2, n_target_registers=2)
    return swz_small


_SWZ_DOC = BloqDocSpec(
    bloq_cls=SwapWithZero,
    import_line='from qualtran.bloqs.swap_network import SwapWithZero',
    examples=(_swz, _swz_small),
)


@frozen
class MultiplexedCSwap(UnaryIterationGate):
    r"""Swaps the $l$-th register into an ancilla using unary iteration.

    Applies the unitary which performs
    $$
        U |l\rangle|\psi_0\rangle\cdots|\psi_l\rangle\cdots|\psi_n\rangle|\mathrm{junk}\rangle
        \rightarrow
        |l\rangle|\psi_0\rangle\cdots|\mathrm{junk}\rangle\cdots|\psi_n\rangle|\psi_l\rangle
    $$
    through a combination of unary iteration and CSwaps.

    The toffoli cost should be $L n_b + L - 2 + n_c$, where $L$ is the
    iteration length, $n_b$ is the bitsize of
    the registers to swap, and $n_c$ is the number of controls.

    Args:
        selection_regs: Indexing `select` signature of type Tuple[`SelectionRegisters`, ...].
            It also contains information about the iteration length of each selection register.
        target_bitsize: The size of the registers we want to swap.
        control_regs: Control registers for constructing a controlled version of the gate.

    Registers:
        control_registers: Control registers
        selection_regs: Indexing `select` signature of type Tuple[`SelectionRegisters`, ...].
            It also contains information about the iteration length of each selection register.
        target_registers: Target registers to swap. We swap FROM registers
            labelled x`i`, where i is an integer and TO a single register called y

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767) page 20 paragraph 2.
    """
    selection_regs: Tuple[SelectionRegister, ...] = field(
        converter=lambda v: (v,) if isinstance(v, SelectionRegister) else tuple(v)
    )
    target_bitsize: int
    control_regs: Tuple[Register, ...] = field(
        converter=lambda v: (v,) if isinstance(v, Register) else tuple(v), default=()
    )

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return self.control_regs

    @cached_property
    def selection_registers(self) -> Tuple[SelectionRegister, ...]:
        return self.selection_regs

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        target_shape = tuple(sreg.iteration_length for sreg in self.selection_registers)
        return tuple(
            [
                Register('targets', bitsize=self.target_bitsize, shape=target_shape),
                Register('output', bitsize=self.target_bitsize),
            ]
        )

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@"] * total_bits(self.control_registers)
        wire_symbols += ["In"] * total_bits(self.selection_registers)
        for i, target in enumerate(self.target_registers):
            if i == len(self.target_registers) - 1:
                wire_symbols += ["×(y)"] * target.total_bits()
            else:
                wire_symbols += ["×(x)"] * target.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def nth_operation(
        self, context: cirq.DecompositionContext, control: cirq.Qid, **kwargs
    ) -> cirq.OP_TREE:
        selection_idx = tuple(kwargs[reg.name] for reg in self.selection_registers)
        target_regs = kwargs['targets']
        output_reg = kwargs['output']
        return CSwap(self.target_bitsize).make_on(
            ctrl=control, x=target_regs[selection_idx], y=output_reg
        )


@bloq_example
def _multiplexed_cswap() -> MultiplexedCSwap:
    from qualtran import SelectionRegister

    selection_bitsize = 3
    iteration_length = 5
    target_bitsize = 2
    multiplexed_cswap = MultiplexedCSwap(
        SelectionRegister('selection', selection_bitsize, iteration_length),
        target_bitsize=target_bitsize,
    )

    return multiplexed_cswap


_MULTIPLEXED_CSWAP_DOC = BloqDocSpec(
    bloq_cls=MultiplexedCSwap,
    import_line='from qualtran.bloqs.swap_network import MultiplexedCSwap',
    examples=(_multiplexed_cswap,),
)
