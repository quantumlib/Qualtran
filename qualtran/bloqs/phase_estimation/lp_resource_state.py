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

"""Resource states proposed by A. Luis and J. Peřina (1996) for optimal phase measurements"""
from functools import cached_property
from typing import Set, TYPE_CHECKING

import attrs
import cirq
import numpy as np
import sympy
from numpy._typing import NDArray

from qualtran import bloq_example, BloqDocSpec, GateWithRegisters, QUInt, Register, Side, Signature
from qualtran.bloqs.basic_gates import CZPowGate, Hadamard, OnEach, Ry, Rz, XGate, ZPowGate
from qualtran.bloqs.mcmt import MultiControlPauli
from qualtran.cirq_interop import CirqGateAsBloq
from qualtran.cirq_interop.t_complexity_protocol import TComplexity

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@attrs.frozen
class _LPRSHelper(GateWithRegisters):
    """Helper Bloq to prepare an intermediate resource state which can be used in AA

    Prepares the intermediate resource state described in Eq 19 of
    https://arxiv.org/pdf/1805.03662.pdf, which can then be used in a single round of
    Amplitude Amplification to boost the amplitude of desired resource state to 1.
    """

    bitsize: int
    eps: float = 1e-11

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(m=self.bitsize, anc=1)

    def short_name(self) -> str:
        return 'LPRS'

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        q, anc = quregs['m'].tolist()[::-1], quregs['anc']
        yield [OnEach(self.bitsize, Hadamard()).on(*q), Hadamard().on(*anc)]
        for i in range(self.bitsize):
            rz_angle = -2 * np.pi * (2**i) / (2**self.bitsize + 1)
            yield Rz(angle=rz_angle).controlled().on(q[i], *anc)
        yield Rz(angle=-2 * np.pi / (2**self.bitsize + 1)).on(*anc)
        yield Hadamard().on(*anc)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {
            (
                CZPowGate(exponent=1 / ((2**self.bitsize + 1) * np.pi), global_shift=-0.5),
                self.bitsize,
            ),
            (ZPowGate(exponent=1 / ((2**self.bitsize + 1) * np.pi), global_shift=-0.5), 1),
            (Hadamard(), 2 + self.bitsize),
        }

    def _t_complexity_(self) -> 'TComplexity':
        return TComplexity(rotations=self.bitsize + 1, clifford=2 + self.bitsize)


@attrs.frozen
class LPResourceState(GateWithRegisters):
    r"""Prepares optimal resource state $\chi_{m}$ proposed by A. Luis and J. Peřina (1996)

    Uses a single round of amplitude amplification, as described in Ref[2], to prepare the
    resource state from Ref[1] described as

    $$
    \chi_{m} = \sqrt{\frac{2}{2^m + 1}}\sum_{n=0}^{2^m - 1}\sin{\frac{\pi(n+1)}{2^m+1}}|n\rangle
    $$

    Args:
        bitsize: The size of the phase register to prepare the resource state on.


    References:
        1. [Optimum phase-shift estimation and the quantum description of the phase
        difference](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.54.4564)
        2. [Encoding Electronic Spectra in Quantum Circuits with Linear T
        Complexity](https://arxiv.org/abs/1805.03662) Section II-B
    """

    bitsize: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('m', QUInt(self.bitsize), side=Side.RIGHT)])

    def state(self):
        N = 2**self.bitsize
        return np.sqrt(2 / (1 + N)) * np.sin(np.pi * (1 + np.arange(N)) / (1 + N))

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        """Use the _LPResourceStateHelper and do a single round of amplitude amplification."""
        q = quregs['m'].flatten().tolist()
        anc, flag = context.qubit_manager.qalloc(2)

        flag_angle = np.arccos(1 / (1 + 2**self.bitsize))

        # Prepare initial state
        yield Ry(angle=flag_angle).on(flag)
        yield _LPRSHelper(self.bitsize).on(*q, anc)

        # Reflect around the target state
        yield CZPowGate().on(flag, anc)

        # Reflect around the initial state
        yield _LPRSHelper(self.bitsize).adjoint().on(*q, anc)
        yield Ry(angle=-flag_angle).on(flag)

        yield XGate().on(flag)
        yield MultiControlPauli((0,) * (self.bitsize + 1), target_gate=cirq.Z).on(*q, anc, flag)
        yield XGate().on(flag)

        yield _LPRSHelper(self.bitsize).on(*q, anc)
        yield Ry(angle=flag_angle).on(flag)

        # Reset ancilla to |0> state.
        yield [XGate().on(flag), XGate().on(anc)]
        yield cirq.global_phase_operation(1j)
        context.qubit_manager.qfree([flag, anc])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        flag_angle = sympy.acos(1 / (1 + 2**self.bitsize))

        return {
            (_LPRSHelper(self.bitsize), 2),
            (_LPRSHelper(self.bitsize).adjoint(), 1),
            (Ry(angle=flag_angle), 3),
            (MultiControlPauli(sympy.zeros(self.bitsize + 1), target_gate=cirq.Z), 1),
            (XGate(), 4),
            (CirqGateAsBloq(cirq.GlobalPhaseGate(1j)), 1),
            (CZPowGate(), 1),
        }


@bloq_example
def _lp_resource_state_small() -> LPResourceState:
    lp_resource_state_small = LPResourceState(5)
    return lp_resource_state_small


@bloq_example
def _lp_resource_state_symbolic() -> LPResourceState:
    import sympy

    # Note: Symbolic callgraphs currently don't work due to
    # https://github.com/quantumlib/Qualtran/issues/786

    lp_resource_state_symbolic = LPResourceState(sympy.Symbol('n'))
    return lp_resource_state_symbolic


_CC_LP_RESOURCE_STATE_DOC = BloqDocSpec(
    bloq_cls=LPResourceState,
    import_line='from qualtran.bloqs.phase_estimation.lp_resource_state import LPResourceState',
    examples=(_lp_resource_state_small, _lp_resource_state_symbolic),
)
