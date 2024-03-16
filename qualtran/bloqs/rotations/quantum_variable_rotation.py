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
r"""
Quantum variable rotation (QVR) represents a family of Bloqs that can act as a Phase Oracle[1, 2],
i.e. it implements a unitary which phases each computational basis state $|x\rangle$, on which
the wave-function has support, by an amount $e^{i 2\pi \gamma x}$. The general unitary can be
defined as

$$
\text{QVR}_{n, \epsilon}(\gamma)\sum_{j=0}^{2^n-1} c_j|x_j\rangle\rightarrow\sum_{j=0}^{2^n-1}
e^{2\pi i\widetilde{\gamma x_j}}c_j|x_j\rangle
$$

where $\epsilon$ parameterizes the accuracy to which we wish to synthesize the phase
coefficients s.t.

$$
|e^{2\pi i\widetilde{\gamma x_j}} - e^{2\pi i \gamma x_j}| \leq \epsilon
$$

which, using rules of propagation of error [3], implies

$$
|\gamma x_j - \widetilde{\gamma x_j}| \leq \frac{\epsilon}{2\pi}
$$

The linked references typically assume that $0 \leq x_{j} \le 1$ and $-1 \leq \gamma \leq 1$,
for ease of exposition and analysis, but we do not have any such constraint. In the
implementations presented below, both the cost register $|x\rangle$ and $\gamma$ can be
arbitrary fixed point integer types.
Each section below presents more details about the constraints on cost register
$|x\rangle$ and scaling constant $\gamma$.


References:
  1. [Faster quantum chemistry simulation on fault-tolerant quantum
        computers](https://iopscience.iop.org/article/10.1088/1367-2630/14/11/115023/meta)
        Fig 14.
  2. [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial
        Optimization](https://arxiv.org/abs/2007.07391) Appendix C: Oracles for
        phasing by cost function
  3. [Formulae for propagating
        uncertainty](https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae)
"""

import abc
from functools import cached_property
from typing import Dict, Sequence, Set, TYPE_CHECKING, Union

import attrs
import numpy as np
import sympy

from qualtran import (
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    GateWithRegisters,
    QFxp,
    Register,
    Signature,
)
from qualtran.bloqs.basic_gates.rotation import ZPowGate
from qualtran.bloqs.rotations.phase_gradient import AddScaledValIntoPhaseReg
from qualtran.resource_counting.symbolic_counting_utils import ceil, log2, smax

if TYPE_CHECKING:
    from qualtran import SoquetT


class QvrInterface(GateWithRegisters, metaclass=abc.ABCMeta):
    """Interface for phase oracles that implement a quantum variable rotation (QVR)."""

    @property
    @abc.abstractmethod
    def cost_registers(self) -> Sequence[Register]:
        ...

    @property
    @abc.abstractmethod
    def extra_registers(self) -> Sequence[Register]:
        ...

    @cached_property
    def signature(self) -> Signature:
        return Signature([*self.cost_registers, *self.extra_registers])


@attrs.frozen
class QvrZPow(QvrInterface):
    r"""QVR oracle that applies a ZPow rotation to every qubit in the n-bit cost register.

    This phase oracle simply applies a $Z^{2^{k}}$ rotation to every qubit in the cost register.
    To obtain a desired accuracy of $\epsilon$, each individual rotation is synthesized with accuracy
    $\frac{\epsilon}{n}$, where $n$ is the size of cost register.

    The toffoli cost of this method scales as

    $$
        \text{Toffoli Cost} \approxeq \mathcal{O}\left(n \log{\frac{n}{\epsilon}} \right)
    $$

    Args:
        cost_reg: Cost register of dtype `QFxp`. Supports arbitrary `QFxp` types, including signed
            and unsigned.
        gamma: Scaling factor to multiply the cost register by, before applying the phase. Can be arbitrary
            floating point number.
        eps: Precision for synthesizing the phases.
    """
    cost_reg: Register
    gamma: Union[float, sympy.Expr] = 1.0
    eps: Union[float, sympy.Expr] = 1e-9

    @classmethod
    def from_bitsize(
        cls,
        bitsize: int,
        gamma: Union[float, sympy.Expr] = 1.0,
        eps: Union[float, sympy.Expr] = 1e-9,
    ) -> 'QvrZPow':
        cost_reg = Register("x", QFxp(bitsize, bitsize, signed=False))
        return QvrZPow(cost_reg, gamma=gamma, eps=eps)

    @cached_property
    def cost_dtype(self) -> QFxp:
        dtype = self.cost_reg.dtype
        assert isinstance(dtype, QFxp)
        return dtype

    @cached_property
    def cost_registers(self) -> Sequence[Register]:
        return [self.cost_reg]

    @cached_property
    def extra_registers(self) -> Sequence[Register]:
        return ()

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        out = soqs[self.cost_reg.name]
        out = bb.split(out)
        eps = self.eps / len(out)
        if self.cost_dtype.signed:
            out[0] = bb.add(ZPowGate(exponent=1, eps=eps), q=out[0])
        for i in range(self.cost_dtype.bitsize):
            power_of_two = i - self.cost_dtype.num_frac
            out[-(i + 1)] = bb.add(
                ZPowGate(exponent=(2**power_of_two) * self.gamma * 2, eps=self.eps / len(out)),
                q=out[-(i + 1)],
            )
        return {self.cost_reg.name: bb.join(out)}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        zpow = ZPowGate(exponent=self.gamma, eps=self.eps / self.cost_dtype.bitsize)
        return {(zpow, self.cost_dtype.bitsize)}


@bloq_example
def _qvr_zpow() -> QvrZPow:
    qvr_zpow = QvrZPow.from_bitsize(12)
    return qvr_zpow


_QVR_ZPOW = BloqDocSpec(
    bloq_cls=QvrZPow,
    import_line='from qualtran.bloqs.rotations.quantum_variable_rotation import QvrZPow',
    examples=(_qvr_zpow,),
)


@attrs.frozen
class QvrPhaseGradient(QvrInterface):
    r"""QVR oracle that applies a rotation via addition into the phase gradient register.

    A $b_\text{grad}$-bit phase gradient state $|\phi\rangle_{b_\text{grad}}$ can be written as

    $$
        |\phi\rangle_{b_\text{grad}} = \frac{1}{\sqrt{2^{b_\text{grad}}}}
        \sum_{k=0}^{2^{b_\text{grad}} - 1} e^{\frac{-2\pi i k}{2^{b_\text{grad}}}}
        \ket{\frac{k}{2^{b_\text{grad}}}}
    $$

    In the above equation $\frac{k}{2^{b_\text{grad}}}$ represents a fixed point fraction. In
    Qualtran, we can represent such a quantum register using quantum data type
    `QFxp(bitsize=b_grad, num_frac=b_grad, signed=False)`. Let
    $\tilde{k}=\frac{k}{2^{b_\text{grad}}}$ be a $b_\text{grad}$-bit fixed point fraction,
    we can rewrite the phase gradient state as


    $$
        |\phi\rangle_{b_\text{grad}} = \frac{1}{\sqrt{2^{b_\text{grad}}}}
        \sum_{\tilde{k}=0}^{\frac{2^{b_\text{grad}-1}}{2^{b_\text{grad}}}}
        e^{-2\pi i \tilde{k}} \ket{\tilde{k}}
    $$


    A useful property of the phase gradient state is that adding a fixed-point number
    $\tilde{l}$ to the state applies a phase-kickback of $e^{2\pi i \tilde{l}}$

    $$
    |\phi + \tilde{l}\rangle_{b_\text{grad}} = e^{2\pi i \tilde{l}}|\phi\rangle_{b_\text{grad}}
    $$

    We exploit this property of the phase gradient states to implement a quantum variable
    rotation via addition into the phase gradient state s.t.

    $$\begin{aligned}
        \text{QVR}_{n,\epsilon}(\gamma)|x\rangle|\phi\rangle &=|x\rangle|\phi+\gamma x\rangle \\
                                          &= e^{2\pi i \gamma x}|x\rangle |\phi\rangle
    \end{aligned}$$

    A number of subtleties arise as part of this procedure and we describe them below one by one.

    - **Adding a scaled value into phase gradient register** Instead of computing $\gamma x$ an
        intermediate register, we perform the multiplication via repeated additions into the phase
        gradient register, as described in [2]. This requires us to represent $\gamma$ as a fixed
        point fraction with bitsize $\gamma_\text{bitsize}$. This procedure introduces two sources
        of errors:
        - **Errors due to fixed point representation of $\gamma$** - Note that adding any fixed
            point number of the form $a.b$ to the phase gradient register is equivalent to adding
            $0.b$ since $e^{2\pi i a} = 1$ for every integer $a$. Let $\tilde{\gamma} = a.b$ and
            $x = p.q$ be fixed point decimal representations of $\gamma$ and $x$. We can write
            the product $\gamma x$ as
        $$
              \tilde{\gamma} x = (\sum_{i=0}^{\gamma_\text{n\_int}} a_{i} * 2^{i} +
              \sum_{i=1}^{\gamma_\text{n\_frac}} \frac{b_i}{2^i}) (\sum_{j=0}^{x_\text{n\_int}}
              p_{j} * 2^{j} + \sum_{j=1}^{x_\text{n\_frac}} \frac{q_{j}}{2^{j}})
        $$
        In order to compute $\tilde{\gamma} x$ to precision $\frac{\epsilon}{2\pi}$, we can
        ignore all terms in the above summation that are < $\frac{\epsilon}{2\pi}$.
        Let $b_\text{phase} = \log_2{\frac{2\pi}{\epsilon}}$, then we get
        $\gamma_\text{n\_frac} = x_\text{n\_int} + b_\text{phase}$. Thus,

        $$\begin{aligned}
              \gamma_\text{bitsize} &= \gamma_\text{n\_int} + x_\text{n\_int} + b_\text{phase} \\
                                    &\approxeq \log_2{\frac{1}{\epsilon}} + x_\text{n\_int} + O(1)
        \end{aligned}$$

        - **Errors due to truncation of digits of $|x\rangle$ during multiplication via repeated
            addition** - Let $b_\text{grad}$ be the size of the phase gradient register. When
            adding left/right shifted copies of state $x$ to the phase gradient register, we incur
            an error every time the fractional part of the shifted state to be added needs to be
            truncated to $b_\text{grad}$ digits. For each such addition the error is upper bounded
            by $\frac{2\pi}{2^{b_\text{grad}}}$, because we omit adding bits that would correspond
            to phase shifts of $\frac{2\pi}{2^{b_\text{grad}+1}}$, $\frac{2\pi}{2^{b_\text{grad}+2}}$,
            and so forth. The number of such additions can be upper bounded by
            $\frac{(\gamma_\text{bitsize} + 2)}{2}$ using techniques from [2].

          - **When $b_\text{grad} \geq x_\text{bitsize}$**:  the first $x_\text{n\_int}$ additions
            do not contribute to any phase error and thus the number of error causing additions can
            be upper bounded by $\frac{(b_\text{phase} + 2)}{2}$. In order to keep the error less
            than $\epsilon$, we get
            $$\begin{aligned}
            b_\text{grad}&=\left\lceil\log_2{\frac{\text{num\_additions}\times2\pi}{\epsilon}}
                        \right\rceil \\
                        &=\left\lceil\log_2{\frac{(b_\text{phase}+2)\pi}{\epsilon}}\right\rceil
                        \text{; if }
                        b_\text{grad} \geq x_\text{bitsize}  \\
            \end{aligned}$$
          - **When $b_\text{grad} \lt x_\text{bitsize}$**: We believe that the above precision for
            $b_\text{grad}$ holds even for this case we have some numerics in tests to verify that.
            Currently, `QvrPhaseGradient` always sets the bitsize of phase gradient register as per
            the above equation.

    - **Constraints on $\gamma$ and $|x\rangle$** - In the current implementation, $\gamma$ can be
        any arbitrary floating point number (signed or unsigned) and $|x\rangle$ must be an unsigned
        fixed point register.

    - **Cost of the phase gradient procedure** - Each addition into the phase gradient register
        costs $b_\text{grad} - 2$ Toffoli's and there are $\frac{\gamma_\text{bitsize} + 2}{2}$
        such additions, therefore the total Toffoli cost is

        $$\begin{aligned}
            \text{Toffoli Cost} &= \frac{(b_\text{grad} - 2)(\gamma_\text{bitsize} + 2)}{2} \\
        \end{aligned}$$


    Thus, for cases where $-1\lt \gamma \lt 1$ and $0 \leq x \lt 1$, the toffoli cost scales
    as $\mathcal{O}\left(\log^2{\frac{1}{\epsilon}} \log{\log{\frac{1}{\epsilon}}}\right)$

    References:

      1. [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization](https://arxiv.org/abs/2007.07391) Section II-C: Oracles for phasing by cost function
      2. [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization](https://arxiv.org/abs/2007.07391) Appendix A: Addition for controlled rotations
    """

    cost_reg: Register
    gamma: Union[float, sympy.Expr] = 1.0
    eps: Union[float, sympy.Expr] = 1e-9

    def __attrs_post_init__(self):
        dtype = self.cost_reg.dtype
        assert isinstance(dtype, QFxp)
        assert dtype.signed is False, "We don't yet support signed integers in QvrPhaseGradient"

    @classmethod
    def from_bitsize(
        cls,
        bitsize: int,
        gamma: Union[float, sympy.Expr] = 1.0,
        eps: Union[float, sympy.Expr] = 1e-9,
    ) -> 'QvrPhaseGradient':
        cost_reg = Register("x", QFxp(bitsize, bitsize, signed=False))
        return QvrPhaseGradient(cost_reg, gamma=gamma, eps=eps)

    @cached_property
    def cost_registers(self) -> Sequence[Register]:
        return [self.cost_reg]

    @cached_property
    def extra_registers(self) -> Sequence[Register]:
        return [Register('phase_grad', QFxp(self.b_grad, self.b_grad))]

    @cached_property
    def cost_dtype(self) -> QFxp:
        dtype = self.cost_reg.dtype
        assert isinstance(dtype, QFxp)
        return dtype

    @cached_property
    def b_phase(self) -> int:
        pi = sympy.pi if isinstance(self.eps, sympy.Expr) else np.pi
        return ceil(log2(pi * 2 / self.eps))

    @cached_property
    def b_grad(self) -> int:
        # Using Equation A7 from https://arxiv.org/abs/2007.07391
        pi = sympy.pi if isinstance(self.eps, sympy.Expr) else np.pi
        return ceil(log2(self.num_additions * 2 * pi / self.eps))

    @cached_property
    def num_additions(self) -> int:
        # Number of additions contributing to the multiplicative error in the reference as
        # assumed to be (gamma_bitsize+2)//2. That limit holds when 0 < gamma < 1 and
        # 0 < cost_reg < 1 and thus `gamma_bitsize` is equal to `b_phase`.
        # However, we support additional cases where both `gamma` and `cost_reg` can be arbitrary
        # integers. We claim that even in this case, the number of additions (affecting the
        # accuracy of multiplication and thus bitsize of gradient register) are still
        # (self.b_phase + 2) // 2.
        # Let `gamma` be represented as `x.y` and `cost_reg` as `a.b`. We perform
        # multiplication of `gamma` and `cost_reg` using repeated additions and can write the
        # product as `x.0 * a.b + 0.y * a.b`. The contribution of each of these terms to the
        # total multiplicative error can be given as:
        #   E(x.0 * a.b) = 0 # Since we only left shift `a.b` and thus do not discard any digit.
        #   E(0.y * a.b) = (y_bitsize-a_bitsize + 2)//2 # Assuming b_grad >= cost_reg.bitsize;
        #                           no digit is discarded for the first `a_bitsize` right shifts.
        return (self.b_phase + 2) // 2

    @cached_property
    def gamma_dtype(self) -> QFxp:
        # Using `gamma_bitsize = log(gamma) + b_{phase} + O(1)` defined b/w equation 34 & 35
        # of https://arxiv.org/abs/2007.07391. Note that for 0 < gamma < 1, `log(gamma)` here
        # means ignoring leading digits of `gamma` which are 0. This optimization is currently
        # not implemented.
        # The reference assumes that cost register always stores a fraction between [0, 1). We
        # do not have this assumption and therefore, we also need to add self.cost_dtype.num_int
        # to the gamma bitsize.
        n_int = smax(0, ceil(log2(abs(self.gamma))))
        n_frac = self.cost_dtype.num_int + self.b_phase
        return QFxp(bitsize=n_int + n_frac, num_frac=n_frac, signed=False)

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        add_scaled_val = AddScaledValIntoPhaseReg(
            self.cost_dtype, self.b_grad, self.gamma, self.gamma_dtype
        )
        out, phase_grad = bb.add(
            add_scaled_val, x=soqs[self.cost_reg.name], phase_grad=soqs['phase_grad']
        )
        return {self.cost_reg.name: out, 'phase_grad': phase_grad}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {
            (
                AddScaledValIntoPhaseReg(
                    self.cost_dtype, self.b_grad, self.gamma, self.gamma_dtype
                ),
                1,
            )
        }


@bloq_example
def _qvr_phase_gradient() -> QvrPhaseGradient:
    qvr_phase_gradient = QvrPhaseGradient.from_bitsize(12)
    return qvr_phase_gradient


_QVR_PHASE_GRADIENT = BloqDocSpec(
    bloq_cls=QvrPhaseGradient,
    import_line='from qualtran.bloqs.rotations.quantum_variable_rotation import QvrPhaseGradient',
    examples=(_qvr_phase_gradient,),
)
