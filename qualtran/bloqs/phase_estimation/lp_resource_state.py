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
from typing import Iterator, Set, Tuple, TYPE_CHECKING, Union

import attrs
import cirq
import numpy as np
import sympy
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    GateWithRegisters,
    QUInt,
    Register,
    Side,
    Signature,
)
from qualtran.bloqs.basic_gates import CZPowGate, GlobalPhase, Hadamard, OnEach, Ry, Rz, XGate
from qualtran.bloqs.mcmt import MultiControlPauli
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.symbolics import acos, HasLength, is_symbolic, pi, SymbolicInt

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@attrs.frozen
class LPRSInterimPrep(GateWithRegisters):
    r"""Helper Bloq to prepare an intermediate resource state which can be used in AA

    Specifically, this prepares the state

    $$
        \sqrt{\frac{1}{2^{m}}}\sum_{n=0}^{2^m - 1}\left(\cos{\left(\frac{\pi(n+1)}{2^m+1}\right)}
        |n\rangle|0\rangle + i\sin{\left(\frac{\pi(n+1)}{2^m+1}\right)}|n\rangle|1\rangle\right)
    $$

    This is the state obtained after applying the Hadamard on the flag qubit as described in
    Eq 19 of https://arxiv.org/pdf/1805.03662.pdf, which can then be used in a single round of
    Amplitude Amplification to boost the amplitude of desired resource state to 1.
    """

    bitsize: SymbolicInt
    eps: float = 1e-11

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(m=self.bitsize, anc=1)

    def pretty_name(self) -> str:
        return 'LPRS'

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]  # type: ignore[type-var]
    ) -> Iterator[cirq.OP_TREE]:
        if isinstance(self.bitsize, sympy.Expr):
            raise ValueError(f'Symbolic bitsize {self.bitsize} not supported')
        q, anc = quregs['m'].tolist()[::-1], quregs['anc']
        yield [OnEach(self.bitsize, Hadamard()).on(*q), Hadamard().on(*anc)]
        for i in range(self.bitsize):
            rz_angle = -2 * np.pi * (2**i) / (2**self.bitsize + 1)
            yield Rz(angle=rz_angle).controlled().on(q[i], *anc)
        yield Rz(angle=-2 * np.pi / (2**self.bitsize + 1)).on(*anc)
        yield Hadamard().on(*anc)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        rz_angle = -2 * pi(self.bitsize) / (2**self.bitsize + 1)
        ret: Set[Tuple[Bloq, SymbolicInt]] = {
            (Rz(angle=rz_angle), 1),
            (Hadamard(), 2 + self.bitsize),
        }
        if is_symbolic(self.bitsize):
            ret |= {(Rz(angle=rz_angle).controlled(), self.bitsize)}
        else:
            ret |= {
                (Rz(angle=rz_angle * (2**i)).controlled(), 1) for i in range(int(self.bitsize))
            }
        return ret

    def _t_complexity_(self) -> 'TComplexity':
        return TComplexity(rotations=self.bitsize + 1, clifford=2 + self.bitsize)


@attrs.frozen
class LPResourceState(GateWithRegisters):
    r"""Prepares optimal resource state $\chi_{m}$ proposed by A. Luis and J. Peřina (1996)

    Uses a single round of amplitude amplification, as described in Ref 2, to prepare the
    resource state from Ref 1 described as

    $$
    \chi_{m} = \sqrt{\frac{2}{2^m + 1}}\sum_{n=0}^{2^m - 1}\sin{\frac{\pi(n+1)}{2^m+1}}|n\rangle
    $$

    Args:
        bitsize: The size of the phase register to prepare the resource state on.


    References:
        [Optimum phase-shift estimation and the quantum description of the phase
        difference](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.54.4564)

        [Encoding Electronic Spectra in Quantum Circuits with Linear T
        Complexity](https://arxiv.org/abs/1805.03662) Section II-B
    """

    bitsize: SymbolicInt

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('m', QUInt(self.bitsize), side=Side.THRU)])

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> Iterator[cirq.OP_TREE]:
        """Use the _LPResourceStateHelper and do a single round of amplitude amplification."""
        q = quregs['m'].flatten().tolist()
        anc, flag = context.qubit_manager.qalloc(2)

        flag_angle = np.arccos(1 / (1 + 2**self.bitsize))

        # Prepare initial state
        yield Ry(angle=flag_angle).on(flag)
        yield LPRSInterimPrep(self.bitsize).on(*q, anc)

        # Reflect around the target state
        yield CZPowGate().on(flag, anc)

        # Reflect around the initial state
        yield LPRSInterimPrep(self.bitsize).adjoint().on(*q, anc)
        yield Ry(angle=-flag_angle).on(flag)

        yield XGate().on(flag)
        yield MultiControlPauli((0,) * (self.bitsize + 1), target_gate=cirq.Z).on(*q, anc, flag)
        yield XGate().on(flag)

        yield LPRSInterimPrep(self.bitsize).on(*q, anc)
        yield Ry(angle=flag_angle).on(flag)

        # Reset ancilla to |0> state.
        yield [XGate().on(flag), XGate().on(anc)]
        yield GlobalPhase(1j).on()
        context.qubit_manager.qfree([flag, anc])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        flag_angle = acos(1 / (1 + 2**self.bitsize))
        cvs: Union[HasLength, Tuple[int, ...]] = (
            HasLength(self.bitsize + 1) if is_symbolic(self.bitsize) else (0,) * (self.bitsize + 1)
        )
        return {
            (LPRSInterimPrep(self.bitsize), 2),
            (LPRSInterimPrep(self.bitsize).adjoint(), 1),
            (Ry(angle=flag_angle), 3),
            (MultiControlPauli(cvs, target_gate=cirq.Z), 1),
            (XGate(), 4),
            (GlobalPhase(1j), 1),
            (CZPowGate(), 1),
        }


@bloq_example
def _lprs_interim_prep() -> LPRSInterimPrep:
    lprs_interim_prep = LPRSInterimPrep(5)
    return lprs_interim_prep


_CC_LPRS_INTERIM_PREP_DOC = BloqDocSpec(
    bloq_cls=LPRSInterimPrep,
    import_line='from qualtran.bloqs.phase_estimation.lp_resource_state import LPRSInterimPrep',
    examples=(_lprs_interim_prep,),
)


@bloq_example
def _lp_resource_state_small() -> LPResourceState:
    lp_resource_state_small = LPResourceState(5)
    return lp_resource_state_small


@bloq_example
def _lp_resource_state_symbolic() -> LPResourceState:
    import sympy

    lp_resource_state_symbolic = LPResourceState(sympy.Symbol('n'))
    return lp_resource_state_symbolic


_CC_LP_RESOURCE_STATE_DOC = BloqDocSpec(
    bloq_cls=LPResourceState,
    import_line='from qualtran.bloqs.phase_estimation.lp_resource_state import LPResourceState',
    examples=(_lp_resource_state_small, _lp_resource_state_symbolic),
)
