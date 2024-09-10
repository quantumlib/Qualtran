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
from collections import Counter
from functools import cached_property
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import attrs
import numpy as np
import sympy

from qualtran import Bloq, bloq_example, BloqDocSpec, GateWithRegisters, QBit, Register, Signature
from qualtran.bloqs.basic_gates import CZ, Hadamard, OnEach, Ry, Rz, XGate
from qualtran.bloqs.phase_estimation.qpe_window_state import QPEWindowStateBase
from qualtran.bloqs.reflections.reflection_using_prepare import ReflectionUsingPrepare
from qualtran.drawing import Text, WireSymbol
from qualtran.symbolics import acos, ceil, is_symbolic, log2, pi, SymbolicFloat, SymbolicInt

if TYPE_CHECKING:
    from qualtran import BloqBuilder, Soquet, SoquetT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


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

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('LPRS')
        return super().wire_symbol(reg, idx)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', *, m: 'SoquetT', anc: 'Soquet'
    ) -> Dict[str, 'SoquetT']:
        if isinstance(self.bitsize, sympy.Expr):
            raise ValueError(f'Symbolic bitsize {self.bitsize} not supported')
        m = bb.add(OnEach(self.bitsize, Hadamard()), q=m)
        q = bb.split(m)[::-1]
        anc = bb.add(Hadamard(), q=anc)
        for i in range(self.bitsize):
            rz_angle = -2 * np.pi * (2**i) / (2**self.bitsize + 1)
            q[i], anc = bb.add(Rz(angle=rz_angle).controlled(), ctrl=q[i], q=anc)
        anc = bb.add(Rz(angle=-2 * np.pi / (2**self.bitsize + 1)), q=anc)
        anc = bb.add(Hadamard(), q=anc)
        return {'m': bb.join(q[::-1]), 'anc': anc}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        rz_angle = -2 * pi(self.bitsize) / (2**self.bitsize + 1)
        ret: Counter['Bloq'] = Counter()
        ret[Rz(angle=rz_angle)] += 1
        ret[OnEach(self.bitsize, Hadamard())] += 1
        ret[Hadamard()] += 2
        if is_symbolic(self.bitsize):
            ret[Rz(angle=rz_angle).controlled()] += self.bitsize
        else:
            for i in range(self.bitsize):
                ret[Rz(angle=rz_angle * (2**i)).controlled()] += 1
        return ret


@attrs.frozen
class LPResourceState(QPEWindowStateBase):
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
        return Signature([self.m_register])

    @classmethod
    def from_standard_deviation_eps(cls, eps: SymbolicFloat) -> 'LPResourceState':
        r"""Estimate the phase $\phi$ with uncertainty in standard deviation bounded by $\epsilon$.

        The standard deviation of phase estimation using optimal resource states scales as the
        square of Holevo variance $\tan{\frac{\pi}{2^m}}$.
        This bound can be used to estimate the size of the phase register s.t. the estimated phase
        has a standard deviation of at-most $\epsilon$. See the class docstring for more details.

        $$
            m = \lceil\log_2{\pi/\epsilon}\rceil
        $$

        Args:
            eps: Maximum standard deviation of the estimated phase.
        """
        return LPResourceState(ceil(log2(pi(eps) / eps)))

    @property
    def m_bits(self) -> SymbolicInt:
        return self.bitsize

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        qpe_reg = bb.allocate(dtype=self.m_register.dtype)
        anc, flag = bb.allocate(dtype=QBit()), bb.allocate(dtype=QBit())

        flag_angle = np.arccos(1 / (1 + 2**self.bitsize))

        # Prepare initial state
        flag = bb.add(Ry(angle=flag_angle), q=flag)
        qpe_reg, anc = bb.add(LPRSInterimPrep(self.bitsize), m=qpe_reg, anc=anc)

        # Reflect around the target state
        flag, anc = bb.add(CZ(), q1=flag, q2=anc)

        # Reflect around the initial state
        qpe_reg, anc = bb.add(LPRSInterimPrep(self.bitsize).adjoint(), m=qpe_reg, anc=anc)
        flag = bb.add(Ry(angle=-flag_angle), q=flag)

        flag, anc, qpe_reg = bb.add(
            ReflectionUsingPrepare.reflection_around_zero([1, 1, self.bitsize], global_phase=1j),
            reg0_=flag,
            reg1_=anc,
            reg2_=qpe_reg,
        )

        qpe_reg, anc = bb.add(LPRSInterimPrep(self.bitsize), m=qpe_reg, anc=anc)
        flag = bb.add(Ry(angle=flag_angle), q=flag)

        # Reset ancilla to |0> state.
        flag = bb.add(XGate(), q=flag)
        anc = bb.add(XGate(), q=anc)
        bb.free(flag)
        bb.free(anc)
        return {'qpe_reg': qpe_reg}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        flag_angle = acos(1 / (1 + 2**self.bitsize))
        reflection_bloq: 'Bloq' = ReflectionUsingPrepare.reflection_around_zero(
            [1, 1, self.bitsize], global_phase=1j
        )
        return {
            LPRSInterimPrep(self.bitsize): 2,
            LPRSInterimPrep(self.bitsize).adjoint(): 1,
            Ry(angle=flag_angle): 2,
            Ry(angle=-1 * flag_angle): 1,
            reflection_bloq: 1,
            XGate(): 2,
            CZ(): 1,
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
