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
from typing import Any, Dict, Iterable, Sequence, Set, TYPE_CHECKING, Union

import attrs
import cirq
import numpy as np
import sympy
from cirq._compat import cached_method
from fxpmath import Fxp
from numpy.typing import NDArray

from qualtran import GateWithRegisters, QBit, QFxp, Register, Side, Signature
from qualtran.bloqs.basic_gates import Hadamard, Toffoli
from qualtran.bloqs.basic_gates.on_each import OnEach
from qualtran.bloqs.basic_gates.rotation import CZPowGate, ZPowGate

if TYPE_CHECKING:
    from qualtran.resource_counting.bloq_counts import BloqCountT


@attrs.frozen
class PhaseGradientUnitary(GateWithRegisters):
    r"""Implementation of (Controlled-/)PhaseGradient unitary gate on an n-bit register.

    The class supports implementing the phase gradient unitary gate and a controlled version
    thereof. The n bit phase gradient unitary is defined as

    $$
        \text{PhaseGrad}_{n, t} = \sum_{k=0}^{2^{n}-1}|k\rangle\ langle k| \omega_{n, t}^{k}
    $$

    where

    $$
        \omega_{n, t} = \exp\left(\frac{2\pi i t}{2^n}\right)
    $$

    The implementation simply decomposes into $n$ (controlled-) rotations, one on each qubit.
    """
    bitsize: int
    exponent: float = 1
    controlled: bool = False
    eps: float = 1e-10

    @cached_property
    def signature(self) -> 'Signature':
        return (
            Signature.build_from_dtypes(ctrl=QBit(), phase_grad=QFxp(self.bitsize, self.bitsize))
            if self.controlled
            else Signature.build_from_dtypes(phase_grad=QFxp(self.bitsize, self.bitsize))
        )

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        ctrl = quregs.get('ctrl', ())
        gate = CZPowGate if self.controlled else ZPowGate
        for i, q in enumerate(quregs['phase_grad']):
            yield gate(exponent=self.exponent / 2**i, eps=self.eps / self.bitsize).on(*ctrl, q)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ['@'] * self.controlled + [
            f'Z^{self.exponent}/{2**(i+1)}' for i in range(self.bitsize)
        ]
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __pow__(self, power):
        if power == 1:
            return self
        return PhaseGradientUnitary(self.bitsize, self.exponent * power, self.controlled, self.eps)


@attrs.frozen
class PhaseGradientState(GateWithRegisters):
    r"""Prepare a phase gradient state $|\phi\rangle$ on a new register of bitsize $b_{grad}$

    $$
        |\phi\rangle = \frac{1}{\sqrt{2^{n}}} \sum_{k=0}^{2^{n} - 1} \omega_{n, t}^{k} |k\rangle
    $$

    where

    $$
        \omega_{n, t} = \exp\left(\frac{2\pi i t}{2^n}\right)
    $$

    Allocates / deallocates registers to store the phase gradient state and delegates
    to the `PhaseGradientUnitary` bloq defined above.

    References:
        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization]
        (https://arxiv.org/abs/2007.07391)
        Appendix A: Addition for controlled rotations
    """

    bitsize: int
    exponent: float = -1
    eps: float = 1e-10

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [Register('phase_grad', QFxp(self.bitsize, self.bitsize), side=Side.RIGHT)]
        )

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        # Assumes `phase_grad` is in big-endian representation.
        phase_grad = quregs['phase_grad']
        yield OnEach(self.bitsize, Hadamard()).on_registers(q=phase_grad)
        yield PhaseGradientUnitary(self.bitsize, exponent=self.exponent, eps=self.eps).on_registers(
            phase_grad=phase_grad
        )


@attrs.frozen
class AddIntoPhaseGrad(GateWithRegisters, cirq.ArithmeticGate):
    r"""Quantum-quantum addition into a phase gradient register using $b_{phase} - 2$ Toffolis

    $$
        U|x\rangle|\text{phase\_grad}\rangle = |x\rangle|\text{phase\_grad} + x\rangle
    $$

    Args:
        x_bitsize: Size of input register.
        phase_bitsize: Size of phase gradient register to which the input value should be added.
        right_shift: An integer specifying the amount by which the input register x should be right
            shifted before adding to the phase gradient register.
        sign: Whether the input register x should be  added or subtracted from the phase gradient
            register.

    Registers:
        - x : Input THRU register storing input value x to be added to the phase gradient register.
        - phase_grad : Phase gradient THRU register.

    References:
        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization]
        (https://arxiv.org/abs/2007.07391), Appendix A: Addition for controlled rotations
    """
    x_bitsize: int
    phase_bitsize: int
    right_shift: int = 0
    sign: int = +1

    def pretty_name(self) -> str:
        sign = '+' if self.sign > 0 else '-'
        return f'pg{sign}=x>>{self.right_shift}' if self.right_shift else f'pg{sign}=x'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(
            x=QFxp(self.x_bitsize, self.x_bitsize, signed=False),
            phase_grad=QFxp(self.phase_bitsize, self.phase_bitsize, signed=False),
        )

    @cached_property
    def phase_dtype(self) -> QFxp:
        return QFxp(self.phase_bitsize, self.phase_bitsize, signed=False)

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return [2] * self.x_bitsize, [2] * self.phase_bitsize

    def with_registers(self, *new_registers: Union[int, Sequence[int]]):
        raise NotImplementedError("not needed.")

    @cached_method
    def scaled_val(self, x: int) -> int:
        """Computes `phase_grad + x` using fixed point arithmetic."""
        x_width = self.x_bitsize + self.right_shift
        x_fxp = _fxp(x / 2**x_width, x_width).like(_fxp(0, self.phase_bitsize)).astype(float)
        return int(x_fxp.astype(float) * 2**self.phase_bitsize)

    def apply(self, x: int, phase_grad: int) -> Union[int, Iterable[int]]:
        out = self.on_classical_vals(x=x, phase_grad=phase_grad)
        return out['x'], out['phase_grad']

    def on_classical_vals(self, x: int, phase_grad: int) -> Dict[str, 'ClassicalValT']:
        phase_grad_out = (phase_grad + self.sign * self.scaled_val(x)) % (2**self.phase_bitsize)
        return {'x': x, 'phase_grad': phase_grad_out}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        num_toffoli = self.phase_bitsize - 2
        return {(Toffoli(), num_toffoli)}

    def _t_complexity_(self) -> 'TComplexity':
        ((toffoli, n),) = self.bloq_counts().items()
        return n * toffoli.t_complexity()

    def add_my_tensors(
        self,
        tn: 'qtn.TensorNetwork',
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        from qualtran.cirq_interop._cirq_to_bloq import _add_my_tensors_from_gate

        _add_my_tensors_from_gate(
            self, self.signature, self.short_name(), tn, tag, incoming=incoming, outgoing=outgoing
        )


def _fxp(x: float, n: int) -> Fxp:
    """When 0 <= x < 1, constructs an n-bit fixed point representation with nice properties.

    Specifically,
    -   op_sizing='same' and const_op_sizing='same' ensure that the returned object is not resized
        to a bigger fixed point number when doing operations with other Fxp objects.
    -   shifting='trunc' ensures that when shifting the Fxp integer to left / right; the digits are
        truncated and no rounding occurs
    -   overflow='wrap' ensures that when performing operations where result overflows, the overflowed
        digits are simply discarded.
    """
    assert 0 <= x < 1
    return Fxp(
        x,
        dtype=f'fxp-u{n}/{n}',
        op_sizing='same',
        const_op_sizing='same',
        shifting='trunc',
        overflow='wrap',
    )


def _mul_via_repeated_add(x_fxp: Fxp, gamma_fxp: Fxp, out: int) -> Fxp:
    """Multiplication via repeated additions algorithm described in Appendix D5"""

    res = _fxp(0, out)
    for i, bit in enumerate(gamma_fxp.bin()):
        if bit == '0':
            continue
        shift = gamma_fxp.n_int - i - 1
        # Left/Right shift by `shift` bits.
        res += x_fxp << shift if shift > 0 else x_fxp >> abs(shift)
    return res


@attrs.frozen
class AddScaledValIntoPhaseReg(GateWithRegisters, cirq.ArithmeticGate):
    r"""Optimized quantum-quantum addition into a phase gradient register scaled by a constant $\gamma$.

    $$
        U(\gamma)|x\rangle|\text{phase\_grad}\rangle = |x\rangle|\text{phase\_grad} + x * \gamma\rangle
    $$

    The operation calls `AddIntoPhaseGrad` gate $(gamma_bitsize + 2) / 2$ times.

    Args:
        x_dtype: Fixed point specification of the input register.
        phase_bitsize: Size of phase gradient register to which the scaled input should be added.
        gamma: Floating point scaling factor.
        gamma_dtype: `QFxp` data type capturing number of bits of precisions to be used for
            integer and fractional part of `gamma`.

    Registers:
        - x : Input THRU register storing input value x to be scaled and added to the phase
            gradient register.
        - phase_grad : Phase gradient THRU register.

    References:
        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization]
        (https://arxiv.org/abs/2007.07391), Appendix A: Addition for controlled rotations
    """

    x_dtype: QFxp
    phase_bitsize: int
    gamma: float
    gamma_dtype: QFxp

    @classmethod
    def from_bitsize(
        cls, x_bitsize: int, phase_bitsize: int, gamma: float, gamma_bitsize: int
    ) -> 'AddScaledValIntoPhaseReg':
        return AddScaledValIntoPhaseReg(
            QFxp(x_bitsize, x_bitsize, signed=False),
            phase_bitsize,
            gamma,
            QFxp(gamma_bitsize + max(0, int(np.log2(abs(gamma)))), gamma_bitsize, signed=False),
        )

    @cached_property
    def signature(self):
        return Signature.build_from_dtypes(x=self.x_dtype, phase_grad=self.phase_dtype)

    def registers(self):
        return [2] * self.x_dtype.num_qubits, [2] * self.phase_bitsize

    def with_registers(self, *new_registers: Union[int, Sequence[int]]):
        raise NotImplementedError("not needed.")

    @cached_property
    def phase_dtype(self) -> QFxp:
        return QFxp(self.phase_bitsize, self.phase_bitsize, signed=False)

    @cached_property
    def gamma_fxp(self) -> Fxp:
        return Fxp(abs(self.gamma), dtype=self.gamma_dtype.fxp_dtype_str)

    @cached_method
    def scaled_val(self, x: int) -> int:
        """Computes `x*self.gamma` using fixed point arithmetic."""
        sign = np.sign(self.gamma)
        # `x` is an integer because we currently represent each bitstring as an integer during simulations.
        # However, `x` should be interpreted as per the fixed point specification given in self.x_dtype.
        # If `self.x_dtype` uses `n_frac` bits to represent the fractional part, `x` should be divided by
        # 2**n_frac (in other words, right shifted by n_frac)
        x_fxp = Fxp(x / 2**self.x_dtype.num_frac, dtype=self.x_dtype.fxp_dtype_str)
        # Similarly, `self.gamma` should be represented as a fixed point number using appropriate number
        # of bits for integer and fractional part. This is done in self.gamma_fxp
        # Compute the result = x_fxp * gamma_fxp
        result = _mul_via_repeated_add(x_fxp, self.gamma_fxp, self.phase_dtype.bitsize)
        # Since the phase gradient register is a fixed point register with `n_word=0`, we discard the integer
        # part of `result`. This is okay because the adding `x.y` to the phase gradient register will impart
        # a phase of `exp(i * 2 * np.pi * x.y)` which is same as `exp(i * 2 * np.pi * y)`
        assert 0 <= result < 1 and result.dtype == self.phase_dtype.fxp_dtype_str
        # Convert the `self.phase_bitsize`-bit fraction into back to an integer and return the result.
        # Sign of `gamma` affects whether we add or subtract into the phase gradient register and thus
        # can be ignored during the fixed point arithmetic analysis.
        return int(np.floor(result.astype(float) * 2**self.phase_dtype.bitsize) * sign)

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        x, phase_grad = quregs['x'], quregs['phase_grad']
        sign = int(np.sign(self.gamma))
        for i, bit in enumerate(self.gamma_fxp.bin()):
            if bit == '0':
                continue
            shift = self.gamma_fxp.n_int - i - 1
            if 0 <= shift < self.x_dtype.num_frac:
                # Left shift by `shift` bits / multiply by 2**shift
                yield AddIntoPhaseGrad(
                    self.x_dtype.num_frac - shift, self.phase_bitsize, sign=sign
                ).on_registers(x=x[shift + self.x_dtype.num_int :], phase_grad=phase_grad)
            elif -len(phase_grad) - self.x_dtype.num_int < shift < 0:
                # Right shift by `shift` bits / divide by 2**shift
                shift = abs(shift)
                x_bitsize = self.x_dtype.num_frac + min(shift, self.x_dtype.num_int)
                x_qubits = x[-x_bitsize:]
                right_shift = max(0, shift - self.x_dtype.num_int)
                x_bitsize = min(x_bitsize, self.phase_dtype.bitsize - right_shift)
                yield AddIntoPhaseGrad(
                    x_bitsize, self.phase_bitsize, right_shift=right_shift, sign=sign
                ).on_registers(x=x_qubits[:x_bitsize], phase_grad=phase_grad)

    def apply(self, x: int, phase_grad: int) -> Union[int, Iterable[int]]:
        out = self.on_classical_vals(x=x, phase_grad=phase_grad)
        return out['x'], out['phase_grad']

    def on_classical_vals(self, x: int, phase_grad: int) -> Dict[str, 'ClassicalValT']:
        phase_grad_out = (phase_grad + self.scaled_val(x)) % 2**self.phase_bitsize
        return {'x': x, 'phase_grad': phase_grad_out}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        num_additions = (self.gamma_dtype.bitsize + 2) // 2
        if not isinstance(self.gamma, sympy.Basic):
            num_additions_naive = 0
            for i, bit in enumerate(self.gamma_fxp.bin()):
                if bit == '0':
                    continue
                shift = self.gamma_fxp.n_int - i - 1
                if -(self.phase_bitsize + self.x_dtype.num_int) < shift < self.x_dtype.num_frac:
                    num_additions_naive += 1
            num_additions = min(num_additions_naive, num_additions)
        return {(AddIntoPhaseGrad(self.x_dtype.bitsize, self.phase_bitsize), num_additions)}

    def _t_complexity_(self):
        ((add_into_phase, n),) = self.bloq_counts().items()
        return n * add_into_phase.t_complexity()

    def add_my_tensors(
        self,
        tn: 'qtn.TensorNetwork',
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        from qualtran.cirq_interop._cirq_to_bloq import _add_my_tensors_from_gate

        _add_my_tensors_from_gate(
            self, self.signature, self.short_name(), tn, tag, incoming=incoming, outgoing=outgoing
        )
