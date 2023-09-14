# Copyright 2023 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Iterable, Sequence, Tuple, Union

import attr
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra, linalg
from cirq_ft.algos import arithmetic_gates, swap_network
from cirq_ft.algos.multi_control_multi_target_pauli import MultiControlPauli
from cirq_ft.algos.select_swap_qrom import SelectSwapQROM
from numpy.typing import NDArray

# TODO: This is more similar to THC circuit but there seems to be a bug with multicontrolled-Z
# @attr.frozen
# class UniformPrepareUpperTriangular(infra.GateWithRegisters):
#     r"""Prepares a uniform superposition over first $n * (n + 1) / 2$ basis states.

#     Prepares the state

#     $$
#         |\psi\rangle = \frace{1}{\sqrt{n*(n+1)/2}} \sum_{p \le q} |p q\rangle
#     $$

#     Args:
#         n: The gate prepares a uniform superposition over two $n$ basis state registers.
#         cv: Control values for each control qubit. If specified, a controlled version
#             of the gate is constructed.

#     References:
#         See Fig 12 of https://arxiv.org/abs/1805.03662 for more details.
#     """

#     n: int

#     @cached_property
#     def registers(self) -> infra.Registers:
#         return infra.Registers.build(p=(self.n - 1).bit_length(), q=(self.n - 1).bit_length())

#     def __repr__(self) -> str:
#         return f"cirq_ft.PrepareUniformSuperpositionTwoRegisters({self.n}, {self.n})"

#     def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
#         target_symbols = ['p'] * self.registers['p'].total_bits()
#         target_symbols += ['q'] * self.registers['q'].total_bits()
#         target_symbols[0] = f"UNIFORM({self.n})"
#         return cirq.CircuitDiagramInfo(wire_symbols=target_symbols)

#     def decompose_from_registers(
#         self,
#         *,
#         context: cirq.DecompositionContext,
#         **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
#     ) -> cirq.OP_TREE:
#         p, q = quregs['p'], quregs['q']
#         # Find K and L as per https://arxiv.org/abs/1805.03662 Fig 12.
#         yield cirq.H.on_each(*p)
#         yield cirq.H.on_each(*q)
#         # Alloc
#         ancilla = context.qubit_manager.qalloc(3)

#         l = self.n * (self.n + 1) // 2
#         theta = np.arccos(1 - (2 ** np.floor(np.log2(l))) / l)
#         yield cirq.Ry(rads=theta)(ancilla[0])
#         logn = (self.n - 1).bit_length()
#         yield arithmetic_gates.LessThanEqualGate(logn, logn).on(*p, *q, ancilla[1])
#         yield cirq.X.on(ancilla[2])
#         yield MultiControlPauli(cvs=[1, 1], target_gate=cirq.Z).on_registers(
#             controls=ancilla[:2], target=ancilla[2]
#         )
#         yield arithmetic_gates.LessThanEqualGate(logn, logn).on(*p, *q, ancilla[1])
#         yield cirq.Ry(rads=-theta)(ancilla[0])

#         # Second half of circuit
#         yield cirq.H.on_each(*p)
#         yield cirq.H.on_each(*q)
#         control_pq = self.registers.merge_qubits(**quregs)
#         yield MultiControlPauli(cvs=[1] * 2 * logn, target_gate=cirq.Z).on_registers(
#             controls=control_pq, target=ancilla[0]
#         )
#         yield cirq.H.on_each(*p)
#         yield cirq.H.on_each(*q)
#         yield arithmetic_gates.LessThanEqualGate(logn, logn).on(*p, *q, ancilla[1])
#         yield MultiControlPauli(cvs=[1, 1], target_gate=cirq.Z).on_registers(
#             controls=ancilla[:2], target=ancilla[2]
#         )
#         yield arithmetic_gates.LessThanEqualGate(logn, logn).on(*p, *q, ancilla[1])
#         yield cirq.X.on(ancilla[2])

#         # Free
#         context.qubit_manager.qfree([*ancilla])


def analyze_state_vector(gate_helper, length: int):
    circuit = cirq.Circuit(cirq.decompose_once(gate_helper.operation))
    # result = cirq.Simulator(dtype=np.complex128).simulate(gate_helper.circuit)
    result = cirq.Simulator(dtype=np.complex128).simulate(circuit)
    state_vector = result.final_state_vector
    # State vector is of the form |l>|temp_{l}>. We trace out the |temp_{l}> part to
    # get the coefficients corresponding to |l>.
    # L, logL = length, length.bit_length()
    # state_vector = state_vector.reshape(2**logL, len(state_vector) // 2**logL)
    # num_non_zero = (abs(state_vector) > 1e-6).sum(axis=1)
    # prepared_state = state_vector.sum(axis=1)
    # assert all(num_non_zero[:L] > 0) and all(num_non_zero[L:] == 0)
    # assert all(np.abs(prepared_state[:L]) > 1e-6) and all(np.abs(prepared_state[L:]) <= 1e-6)
    # prepared_state = prepared_state[:L] / np.sqrt(num_non_zero[:L])
    return state_vector


# index_1 = lambda x, y: (x)*(x-1)//2 + y
# ndim = 4
# for i in range(0, ndim):
#     for j in range(0, i + 1):
#         print(i, j, index_1(i, j))

# print()
# index_2 = lambda x, y: (x+1)*(x)//2 + y
# for i in range(0, ndim):
#     for j in range(0, i + 1):
#         print(i, j, index_2(i, j))


@attr.frozen
class ContiguousRegisterGateEqual(cirq.ArithmeticGate):
    """Applies U|x>|y>|0> -> |x>|y>|(y+1)y//2 + x>

    This is useful in the case when $|x>$ and $|y>$ represent two selection registers such that
     $x <= y$. For example, imagine a classical for-loop over two variables $x$ and $y$:

     References:
         [Even More Efficient Quantum Computations of Chemistry Through Tensor Hypercontraction]
         (https://arxiv.org/abs/2011.03494)
            Lee et. al. (2020). Appendix F, Page 67.
    """

    bitsize: int
    target_bitsize: int

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return [2] * self.bitsize, [2] * self.bitsize, [2] * self.target_bitsize

    def with_registers(self, *new_registers) -> 'ContiguousRegisterGateEqual':
        x_bitsize, y_bitsize, target_bitsize = [len(reg) for reg in new_registers]
        assert (
            x_bitsize == y_bitsize
        ), f'x_bitsize={x_bitsize} should be same as y_bitsize={y_bitsize}'
        return ContiguousRegisterGateEqual(x_bitsize, target_bitsize)

    def apply(self, *register_vals: int) -> Union[int, Iterable[int]]:
        x, y, target = register_vals
        return x, y, target ^ ((y * (y + 1)) // 2 + x)

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["In(x)"] * self.bitsize
        wire_symbols += ["In(y)"] * self.bitsize
        wire_symbols += ['+(y(y-1)/2 + x)'] * self.target_bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _t_complexity_(self) -> infra.TComplexity:
        # See the linked reference for explanation of the Toffoli complexity.
        toffoli_complexity = infra.t_complexity(cirq.CCNOT)
        return (self.bitsize**2 + self.bitsize - 1) * toffoli_complexity

    def __repr__(self) -> str:
        return f'cirq_ft.ContiguousRegisterGateEqual({self.bitsize}, {self.target_bitsize})'

    def __pow__(self, power: int):
        if power in [1, -1]:
            return self
        return NotImplemented  # pragma: no cover


@attr.frozen
class UniformPrepareAlt(infra.GateWithRegisters):
    r"""Prepares a uniform superposition over first $n * (n + 1) / 2$ basis states.

    Prepares the state

    $$
        |\psi\rangle = \frace{1}{\sqrt{n*(n+1)/2}} \sum_{p \le q} |p q\rangle
    $$

    Args:
        n: The gate prepares a uniform superposition over two $n$ basis state registers.

        cv: Control values for each control qubit. If specified, a controlled version
            of the gate is constructed.

    References:
        See Fig 12 of https://arxiv.org/abs/1805.03662 for more details.
    """

    n: int

    @cached_property
    def registers(self) -> infra.Registers:
        return infra.Registers.build(p=(self.n - 1).bit_length())

    def __repr__(self) -> str:
        return f"cirq_ft.PrepareUniformAlt({self.n}, {self.n})"

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        target_symbols = ['p'] * self.registers['p'].total_bits()
        target_symbols[0] = f"UNIFORM({self.n})"
        return cirq.CircuitDiagramInfo(wire_symbols=target_symbols)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        p = quregs['p']
        yield cirq.H.on_each(*p)
        # Alloc
        ancilla = context.qubit_manager.qalloc(3)

        l = self.n
        theta = np.arccos(1 - (2 ** np.floor(np.log2(l))) / l)
        logn = (self.n - 1).bit_length()
        yield arithmetic_gates.LessThanGate(logn, self.n).on(*p, ancilla[0])
        yield cirq.Ry(rads=theta)(ancilla[1])
        yield cirq.CCX(ancilla[0], ancilla[1], ancilla[2])
        yield cirq.Z(ancilla[2])
        yield cirq.CCX(ancilla[0], ancilla[1], ancilla[2])
        yield cirq.Ry(rads=-theta)(ancilla[1])
        yield arithmetic_gates.LessThanGate(logn, self.n).on(*p, ancilla[0])

        # Second half of circuit
        yield cirq.H.on_each(*p)
        yield MultiControlPauli(cvs=[0] * (logn + 2), target_gate=cirq.X).on_registers(
            controls=list(p) + list(ancilla[:2]), target=ancilla[2]
        )
        yield cirq.Z(ancilla[2])
        yield MultiControlPauli(cvs=[0] * (logn + 2), target_gate=cirq.X).on_registers(
            controls=list(p) + list(ancilla[:2]), target=ancilla[2]
        )
        yield cirq.H.on_each(*p)
        yield cirq.Ry(rads=theta)(ancilla[1])

        # Free
        context.qubit_manager.qfree([*ancilla])


@attr.frozen
class UniformPrepareUpperTriangular(infra.GateWithRegisters):
    r"""Prepares a uniform superposition over first $n * (n + 1) / 2$ basis states.

    Prepares the state

    $$
        |\psi\rangle = \frace{1}{\sqrt{n*(n+1)/2}} \sum_{p \le q} |p q\rangle
    $$

    Args:
        n: The gate prepares a uniform superposition over two $n$ basis state registers.

        cv: Control values for each control qubit. If specified, a controlled version
            of the gate is constructed.

    References:
        See Fig 12 of https://arxiv.org/abs/1805.03662 for more details.
    """

    n: int

    @cached_property
    def registers(self) -> infra.Registers:
        return infra.Registers.build(p=(self.n - 1).bit_length(), q=(self.n - 1).bit_length())

    def __repr__(self) -> str:
        return f"cirq_ft.PrepareUniformSuperpositionTwoRegisters({self.n}, {self.n})"

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        target_symbols = ['p'] * self.registers['p'].total_bits()
        target_symbols += ['q'] * self.registers['q'].total_bits()
        target_symbols[0] = f"UNIFORM({self.n})"
        return cirq.CircuitDiagramInfo(wire_symbols=target_symbols)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        p, q = quregs['p'], quregs['q']
        # Find K and L as per https://arxiv.org/abs/1805.03662 Fig 12.
        yield cirq.H.on_each(*p)
        yield cirq.H.on_each(*q)
        # Alloc
        ancilla = context.qubit_manager.qalloc(3)

        l = self.n * (self.n + 1) // 2
        theta = np.arccos(1 - (2 ** np.floor(np.log2(l))) / l)
        logn = (self.n - 1).bit_length()
        yield arithmetic_gates.LessThanEqualGate(logn, logn).on(*p, *q, ancilla[0])
        yield cirq.Ry(rads=theta)(ancilla[1])
        yield cirq.CCX(ancilla[0], ancilla[1], ancilla[2])
        yield cirq.Z(ancilla[2])
        yield cirq.CCX(ancilla[0], ancilla[1], ancilla[2])
        yield cirq.Ry(rads=-theta)(ancilla[1])
        yield arithmetic_gates.LessThanEqualGate(logn, logn).on(*p, *q, ancilla[0])

        # Second half of circuit
        yield cirq.H.on_each(*p)
        yield cirq.H.on_each(*q)
        control_pqa = self.registers.merge_qubits(**quregs) + ancilla[:2]
        yield MultiControlPauli(cvs=[0] * (2 * logn + 2), target_gate=cirq.X).on_registers(
            controls=control_pqa, target=ancilla[2]
        )
        yield cirq.Z(ancilla[2])
        yield MultiControlPauli(cvs=[0] * (2 * logn + 2), target_gate=cirq.X).on_registers(
            controls=control_pqa, target=ancilla[2]
        )
        yield cirq.H.on_each(*p)
        yield cirq.H.on_each(*q)
        # Reflect around ancilla
        yield cirq.Ry(rads=theta)(ancilla[1])

        # Free
        context.qubit_manager.qfree([*ancilla])


@attr.frozen
class PrepareUpperTriangular(infra.GateWithRegisters):
    r"""Prepares a uniform superposition over first $n * (n + 1) / 2$ basis states."""

    dim: int
    alt_p: Tuple[int, ...]
    alt_q: Tuple[int, ...]
    keep: Tuple[int, ...]
    mu: int

    @classmethod
    def build(
        cls, dim: int, probs: NDArray[np.int_], epsilon: float = 1e-8
    ) -> "PrepareUpperTriangular":
        alt, keep, mu = linalg.preprocess_lcu_coefficients_for_reversible_sampling(
            lcu_coefficients=probs, epsilon=epsilon
        )
        p_upper, q_upper = np.triu_indices(dim)
        # Get the correct p and q indices from their corresponding upper triangular values.
        alt_p = tuple([int(p) for p in p_upper[alt]])
        alt_q = tuple([int(q) for q in q_upper[alt]])
        keep = tuple([int(k) for k in keep])
        return PrepareUpperTriangular(dim=dim, alt_p=alt_p, alt_q=alt_q, keep=keep, mu=mu)

    @cached_property
    def registers(self) -> infra.Registers:
        return infra.Registers.build(p=(self.dim - 1).bit_length(), q=(self.dim - 1).bit_length())

    def __repr__(self) -> str:
        return f"cirq_ft.PrepareUpperTriangular({self.n}, {self.n})"

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        target_symbols = ['p'] * self.registers['p'].total_bits()
        target_symbols += ['q'] * self.registers['q'].total_bits()
        target_symbols[0] = f"Prepare({self.dim})"
        return cirq.CircuitDiagramInfo(wire_symbols=target_symbols)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        p, q = quregs['p'], quregs['q']
        # Allocate ancillas
        log_dim = (self.dim - 1).bit_length()
        contiguous_reg_anc = context.qubit_manager.qalloc(2 * log_dim)
        alt_p_anc = context.qubit_manager.qalloc(log_dim)
        alt_q_anc = context.qubit_manager.qalloc(log_dim)
        keep_anc = context.qubit_manager.qalloc(self.mu)
        sigma_anc = context.qubit_manager.qalloc(self.mu)
        ineq_anc = context.qubit_manager.qalloc(1)
        # 0. Prepare uniform superposition
        yield UniformPrepareUpperTriangular(n=self.dim).on_registers(p=p, q=q)
        # 1. Contiguous register gate from p and q
        yield ContiguousRegisterGateEqual(log_dim, 2 * log_dim).on(*p, *q, *contiguous_reg_anc)
        # 2. SelectSwapQROM for alt / keep
        yield SelectSwapQROM(
            self.alt_p, self.alt_q, self.keep, target_bitsizes=(log_dim, log_dim, self.mu)
        ).on_registers(
            selection=contiguous_reg_anc, target0=alt_p_anc, target1=alt_q_anc, target2=keep_anc
        )
        yield cirq.H.on_each(*sigma_anc)
        # # 3. Inequality test
        yield arithmetic_gates.LessThanEqualGate(self.mu, self.mu).on(
            *keep_anc, *sigma_anc, *ineq_anc
        )
        # # 4. Swaps
        yield swap_network.MultiTargetCSwap.make_on(
            control=ineq_anc, target_x=alt_p_anc, target_y=p
        )
        yield swap_network.MultiTargetCSwap.make_on(
            control=ineq_anc, target_x=alt_q_anc, target_y=q
        )
        # # Uncompute inequality
        yield arithmetic_gates.LessThanEqualGate(self.mu, self.mu).on(
            *keep_anc, *sigma_anc, *ineq_anc
        )

        # Free
        context.qubit_manager.qfree(
            [*contiguous_reg_anc, *alt_p_anc, *alt_q_anc, *keep_anc, *sigma_anc, *ineq_anc]
        )
