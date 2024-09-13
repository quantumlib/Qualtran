#  Copyright 2024 Google LLC
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
from typing import Dict, List, TYPE_CHECKING

import attrs
import numpy as np
import scipy

from qualtran import bloq_example, BloqDocSpec, Signature
from qualtran.bloqs.phase_estimation.qpe_window_state import QPEWindowStateBase
from qualtran.symbolics import (
    ceil,
    HasLength,
    is_symbolic,
    ln,
    log2,
    pi,
    SymbolicFloat,
    SymbolicInt,
)

if TYPE_CHECKING:
    import quimb.tensor as qtn

    from qualtran import ConnectionT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@attrs.frozen
class KaiserWindowState(QPEWindowStateBase):
    r"""Bloq to prepare a Kaiser window state for high confidence Quantum Phase Estimation.

    Kaiser window states are optimal to minimize the probability of error outside a given
    confidence interval.
    Given the bitsize $m$ and parameter $\alpha$, the bloq prepares an $m$-bit state with
    coefficients

    $$
        \sum\limits_{x=-M}^{M}\frac{1}{2M} \frac{I_0\left(\pi\alpha\sqrt{1-(x/M)^2}\right)}{I_0\left(\pi\alpha\right)}\ket{x}
    $$

    where $M = 2^{m-1}$. See Ref[1] for more details.


    Args:
        bitsize: Number of bits in the control register of QPE.
        alpha: Shape parameter, determines trade-off between main-lobe width and side lobe level.

    References:
        [Analyzing Prospects for Quantum Advantage in Topological Data
        Analysis](https://arxiv.org/abs/2209.13581).
        Berry et. al. (2022). Appendix D
    """

    bitsize: SymbolicInt
    alpha: float

    @cached_property
    def kaiser_state_coeff(self) -> np.ndarray:
        if is_symbolic(self.bitsize):
            raise ValueError(f"Cannot compute coefficients for symbilic {self}")
        state = scipy.signal.windows.kaiser(2**self.bitsize, self.alpha * np.pi, sym=False)
        return state / np.linalg.norm(state)

    @property
    def m_bits(self) -> SymbolicInt:
        return self.bitsize

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([self.m_register])

    @classmethod
    def from_precision_and_delta(
        cls, precision: SymbolicInt, delta: SymbolicFloat
    ) -> 'KaiserWindowState':
        r"""Estimate $\varphi$ to $precision$ bits with $1-\delta$ success probability.

        See Eq.D14 and Eq.D15 of Ref[1] for more details.
        $$
            \alpha = \frac{1}{\pi}\ln{(\frac{1}{\delta})} + \mathcal{O}(\ln{\ln{\frac{1}{\delta}}})
        $$
        and
        $$
            m = n + \log_2{(\ln{\frac{1}{\delta}}} + \mathcal{O}(\frac{1}{2^n}\ln\ln{\frac{1}{\delta}}))
        $$

        Args:
            precision: Number of bits of precision
            delta: Probability of success.
        """
        alpha = 1 / pi(delta) * ln(1 / delta) + ln(ln(1 / delta))
        m_bits = precision + ceil(log2(ln(1 / delta + 1 / 2**precision * ln(ln(1 / delta)))))
        return KaiserWindowState(bitsize=m_bits, alpha=alpha)

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        return [
            qtn.Tensor(
                data=self.kaiser_state_coeff.reshape((2,) * self.bitsize),
                inds=[(outgoing['qpe_reg'], i) for i in range(int(self.bitsize))],
                tags=[str(self)],
            )
        ]

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        from qualtran.bloqs.state_preparation.state_preparation_via_rotation import (
            StatePreparationViaRotations,
        )

        if is_symbolic(self.bitsize, self.alpha):
            state_prep_coeff = HasLength(2**self.bitsize)
        else:
            state_prep_coeff = self.kaiser_state_coeff.tolist()
        return {StatePreparationViaRotations(state_prep_coeff, self.bitsize): 1}


@bloq_example
def _kaiser_window_state_small() -> KaiserWindowState:
    kaiser_window_state_small = KaiserWindowState(5, 2)
    return kaiser_window_state_small


@bloq_example
def _kaiser_window_state_symbolic() -> KaiserWindowState:
    import sympy

    kaiser_window_state_symbolic = KaiserWindowState(*sympy.symbols('n, alpha'))
    return kaiser_window_state_symbolic


_CC_KAISER_WINDOW_STATE_DOC = BloqDocSpec(
    bloq_cls=KaiserWindowState,
    import_line='from qualtran.bloqs.phase_estimation.kaiser_window_state import KaiserWindowState',
    examples=(_kaiser_window_state_small, _kaiser_window_state_symbolic),
)
