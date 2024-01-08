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
from typing import Dict, Set, TYPE_CHECKING

import attrs
import numpy as np

from qualtran import GateWithRegisters, Signature
from qualtran.bloqs.arithmetic import HammingWeightCompute
from qualtran.bloqs.basic_gates import ZPowGate
from qualtran.bloqs.rotations.phase_gradient import AddScaledValIntoPhaseReg

if TYPE_CHECKING:
    from qualtran import BloqBuilder, SoquetT
    from qualtran.resource_counting.bloq_counts import BloqCountT, SympySymbolAllocator


@attrs.frozen
class HammingWeightPhasing(GateWithRegisters):
    r"""Applies $Z^{\text{exponent}}$ to every qubit of an input register of size `bitsize`.

    Hamming weight phasing reduces the number of rotations to be synthesized from $n$ (where
    $n=\text{bitsize}$ is the size of the input register) to $\log_2(n)$ via the following steps:
        1. Compute the hamming weight (HW) of the input register in using (at-most) $n-1$ ancilla
            and Toffolis in a newly allocated output register of size $\log_2(n)$.
            $HW|x\rangle -> |x\rangle |\text{HW}(x)\rangle$
        2. Apply $Z^{2^{k}\text{exponent}}$ to the k'th qubit of newly allocated hamming weight
             register.
        3. Uncompute the hamming weight register and ancillas allocated in Step-1 with 0 Toffoli
            cost.

    Overall, for an input register of size $n$, the procedure uses $n - \alpha$ Toffoli's and
    $n - \alpha + \log_2(n)$ ancilla to reduce $n$ rotation syntheses into $\log_2(n)$  rotation
    synthesis. Here $\alpha = \text{hamming\_weight}(n)$.

    Args:
        bitsize: Size of input register to apply `Z ** exponent` to.
        exponent: The exponent of `Z ** exponent` to be applied to each qubit in the input register.
        eps: Accuracy of synthesizing the Z rotations.

    Registers:
        A single THRU register of size `bitsize`.

    References:
        [Halving the cost of quantum addition](https://arxiv.org/abs/1709.06648), Page-4
    """

    bitsize: int
    exponent: float
    eps: int = 1e-10

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(x=self.bitsize)

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        soqs['x'], junk, out = bb.add(HammingWeightCompute(self.bitsize), x=soqs['x'])
        out = bb.split(out)
        for i in range(len(out)):
            out[-(i + 1)] = bb.add(
                ZPowGate(exponent=(2**i) * self.exponent, eps=self.eps / len(out)),
                q=out[-(i + 1)],
            )
        out = bb.join(out)
        soqs['x'] = bb.add(
            HammingWeightCompute(self.bitsize).adjoint(), x=soqs['x'], junk=junk, out=out
        )
        return soqs

    def short_name(self) -> str:
        return f'HWP_{self.bitsize}(Z^{self.exponent})'

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {
            (HammingWeightCompute(self.bitsize), 1),
            (HammingWeightCompute(self.bitsize).adjoint(), 1),
            (ZPowGate(exponent=self.exponent), self.bitsize.bit_length()),
        }


@attrs.frozen
class HammingWeightPhasingViaPhaseGradient(GateWithRegisters):
    r"""Applies $Z^{\text{exponent}}$ to every qubit of an input register of size `bitsize`.

    See docstring of `HammingWeightPhasing` for more details about how hamming weight phasing works.

    In this variant of Hamming Weight Phasing, instead of directly synthesizing $O(n.bit_length())$
    rotations on the hamming weight register, we synthesize the rotations via an addition into the
    phase gradient register. See reference [1] for more details on this technique.

    Note: For most reasonable values of `bitsize` and `eps`, the naive `HammingWeightPhasing` would
    have better constant factors than `HammingWeightPhasingViaPhaseGradient`. This is because, in
    general, the primary advantage of using phase gradient is to reduce the complexity from
    $O(n * log(1/eps))$ to O(log(1/eps) ** 2) (the phase gradient register is of size
    $O(log(1/eps))$ and a scaled addition into the target takes $(b_{grad} - 2)(log(1/eps) + 2)$).
    Therefore, to apply $n$ individual rotations on a target register of size $n$, the complexity is
    independent of $n$ and is essentially a constant (scales only with log(1/eps)).
    However, for the actual constant values to be better, the value of $n$ needs to be
    $> log(1/eps)$. In the case of hamming weight phasing, $n$ corresponds to the hamming weight
    register which itself is $log(\text{bitsize})$. Thus, as `eps` becomes smaller, the required
    value of $\text{bitsize}$, for the phase gradient version to become more performant, becomes
    larger.

    Args:
        bitsize: Size of input register to apply `Z ** exponent` to.
        exponent: The exponent of `Z ** exponent` to be applied to each qubit in the input register.
        eps: Accuracy of synthesizing the Z rotations.

    Registers:
        - x : Input THRU register of size `bitsize`, to apply `Z**exponent` to.
        - phase_grad : Phase gradient THRU register of size `O(log2(1/eps))`, to be used to
            apply the phasing via addition.

    References:
        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization]
        (https://arxiv.org/abs/2007.07391), Appendix A: Addition for controlled rotations
    """

    bitsize: int
    exponent: float = 1
    eps: int = 1e-10

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(x=self.bitsize, phase_grad=self.b_grad)

    @cached_property
    def b_phase(self) -> int:
        return int(np.ceil(np.log2(1 / self.eps)))

    @cached_property
    def b_grad(self) -> int:
        return max(self.bitsize.bit_length(), self.b_phase + int(np.ceil(np.log2(self.b_phase))))

    def build_composite_bloq(
        self, bb: 'BloqBuilder', *, x: 'SoquetT', phase_grad: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        x, junk, out = bb.add(HammingWeightCompute(self.bitsize), x=x)
        out, phase_grad = bb.add(
            AddScaledValIntoPhaseReg(
                self.bitsize.bit_length(), self.b_grad, self.exponent / 2, self.b_phase
            ),
            x=out,
            phase_grad=phase_grad,
        )
        x = bb.add(HammingWeightCompute(self.bitsize).adjoint(), x=x, junk=junk, out=out)
        return {'x': x, 'phase_grad': phase_grad}

    def short_name(self) -> str:
        return f'HWPG_{self.bitsize}(Z^{self.exponent})'
