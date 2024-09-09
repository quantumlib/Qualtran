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
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import attrs
import numpy as np

from qualtran import bloq_example, BloqDocSpec, GateWithRegisters, QFxp, QUInt, Register, Signature
from qualtran.bloqs.arithmetic import HammingWeightCompute
from qualtran.bloqs.basic_gates import ZPowGate
from qualtran.bloqs.rotations.quantum_variable_rotation import QvrPhaseGradient
from qualtran.drawing import Text, WireSymbol
from qualtran.symbolics import SymbolicFloat, SymbolicInt

if TYPE_CHECKING:
    from qualtran import BloqBuilder, SoquetT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@attrs.frozen
class HammingWeightPhasing(GateWithRegisters):
    r"""Applies $Z^{\text{exponent}}$ to every qubit of an input register of size `bitsize`.

    The goal of Hamming Weight Phasing is to reduce the number of rotations needed to
    apply a single qubit rotation $Z^{\texttt{exponent}}$
    to every qubit of an input register `x` of size `bitsize` from `bitsize` to $O(\log (\texttt{bitsize}))$.
    Naively this would take exactly `bitsize` rotations to be synthesized. The number of rotations synthesized is
    reduced by taking advantage of the insight that the resulting phase that is applied to
    an input state only depends on the Hamming weight of the state. Since each `1` that is present in the input register
    accumulates a phase of $(-1)^{\texttt{exponenet}}$, the total accumulated
    phase of an input basis state is $(-1)^{\text{exponent} * HW(x)}$, where
    $HW(x)$ is the Hamming weight of $x$. The overall procedure is done in 3 steps:

    1. Compute the input register Hamming weight coherently using (at-most) $\texttt{bitsize}-1$ ancilla
        and Toffolis, storing the result in a newly allocated output
        register of size $\log_2(\texttt{bitsize})$. $HW|x\rangle \mapsto |x\rangle |HW(x)\rangle$.
        See `HammingWeightCompute` for implementation details of this step.
    2. Apply $Z^{2^{k}\text{exponent}}$ to the k'th qubit of newly allocated Hamming weight
         register.
    3. Uncompute the Hamming weight register and ancillas allocated in Step-1 with 0 Toffoli
        cost.

    Since the size of the Hamming weight register is $\log_2(\texttt{bitsize})$, as the maximum
    Hamming weight is $\texttt{bitsize}$ and we only need $\log_2$ bits to store that as an integer, we
    have reduced the number of costly rotations to be synthesized from $\texttt{bitsize}$
    to $\log_2(\texttt{bitsize})$. This procedure uses $\texttt{bitsize} - HW(\texttt{bitsize})$
    Toffoli's and $\texttt{bitsize} - HW(\texttt{bitsize}) + \log_2(\texttt{bitsize})$
    ancilla qubits to achieve this reduction in rotations.

    Args:
        bitsize: Size of input register to apply `Z ** exponent` to.
        exponent: The exponent of `Z ** exponent` to be applied to each qubit in the input register.
        eps: Accuracy of synthesizing the Z rotations.

    Registers:
        x: A `THRU` register of `bitsize` qubits.

    References:
        [Halving the cost of quantum addition](https://arxiv.org/abs/1709.06648), Page-4
    """

    bitsize: int
    exponent: float = 1
    eps: SymbolicFloat = 1e-10

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(x=QUInt(self.bitsize))

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        soqs['x'], junk, out = bb.add(HammingWeightCompute(self.bitsize), x=soqs['x'])
        out = bb.split(out)
        for i in range(len(out)):
            out[-(i + 1)] = bb.add(
                ZPowGate(exponent=(2**i) * self.exponent, eps=self.eps / len(out)),
                q=out[-(i + 1)],
            )
        out = bb.join(out, dtype=QUInt(self.bitsize.bit_length()))
        soqs['x'] = bb.add(
            HammingWeightCompute(self.bitsize).adjoint(), x=soqs['x'], junk=junk, out=out
        )
        return soqs

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text(f'HWP_{self.bitsize}(Z^{self.exponent})')
        return super().wire_symbol(reg, idx)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {
            HammingWeightCompute(self.bitsize): 1,
            HammingWeightCompute(self.bitsize).adjoint(): 1,
            ZPowGate(
                exponent=self.exponent, eps=self.eps / self.bitsize.bit_length()
            ): self.bitsize.bit_length(),
        }


@bloq_example
def _hamming_weight_phasing() -> HammingWeightPhasing:
    hamming_weight_phasing = HammingWeightPhasing(4, np.pi / 2.0)
    # Applying this unitary to |1111> should be the identity, and |0101> will flip the sign.
    return hamming_weight_phasing


_HAMMING_WEIGHT_PHASING_DOC = BloqDocSpec(
    bloq_cls=HammingWeightPhasing, examples=(_hamming_weight_phasing,)
)


@attrs.frozen
class HammingWeightPhasingViaPhaseGradient(GateWithRegisters):
    r"""Applies $Z^{\text{exponent}}$ to every qubit of an input register of size `bitsize`.

    See docstring of `HammingWeightPhasing` for more details about how hamming weight phasing works.

    In this variant of Hamming Weight Phasing, instead of directly synthesizing $O(\log_2 (\texttt{bitsize}))$
    rotations on the Hamming weight register we synthesize the rotations via an addition into the
    phase gradient register. See reference [1] for more details on this technique.

    Note: For most reasonable values of `bitsize` and `eps`, the naive `HammingWeightPhasing` would
    have better constant factors than `HammingWeightPhasingViaPhaseGradient`. This is because, in
    general, the primary advantage of using phase gradient is to reduce the complexity from
    $O(n * \log(1/ \texttt{eps} ))$ to $O(\log^2(1/ \texttt{eps} ))$ (the phase gradient register is of size
    $O(\log(1/\texttt{eps}))$ and a scaled addition into the target takes $(b_{grad} - 2)(\log(1/\texttt{eps}) + 2)$).
    Therefore, to apply $n$ individual rotations on a target register of size $n$, the complexity is
    independent of $n$ and is essentially a constant (scales only with $log(1/\texttt{eps})$).
    However, for the actual constant values to be better, the value of $n$ needs to be
    $> \log(1/\texttt{eps})$. In the case of hamming weight phasing, $n$ corresponds to the hamming weight
    register which itself is $\log(\texttt{bitsize})$. Thus, as `eps` becomes smaller, the required
    value of $\texttt{bitsize}$, for the phase gradient version to become more performant, becomes
    larger.

    Args:
        bitsize: Size of input register to apply `Z ** exponent` to.
        exponent: The exponent of `Z ** exponent` to be applied to each qubit in the input register.
        eps: Accuracy of synthesizing the Z rotations.

    Registers:
        x : Input THRU register of size `bitsize`, to apply `Z**exponent` to.
        phase_grad : Phase gradient THRU register of size `O(log2(1/eps))`, to be used to
            apply the phasing via addition.

    References:
        1. [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization]
        (https://arxiv.org/abs/2007.07391), Appendix A: Addition for controlled rotations
    """

    bitsize: int
    exponent: float = 1
    eps: float = 1e-10

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(
            x=QUInt(self.bitsize), phase_grad=QFxp(self.b_grad, self.b_grad)
        )

    @cached_property
    def phase_oracle(self) -> QvrPhaseGradient:
        return QvrPhaseGradient(
            Register('out', QFxp(bitsize=self.bitsize.bit_length(), num_frac=0, signed=False)),
            self.exponent / 2,
            self.eps,
        )

    @cached_property
    def b_grad(self) -> 'SymbolicInt':
        return self.phase_oracle.b_grad

    @cached_property
    def gamma_dtype(self) -> QFxp:
        return self.phase_oracle.gamma_dtype

    def build_composite_bloq(
        self, bb: 'BloqBuilder', *, x: 'SoquetT', phase_grad: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        x, junk, out = bb.add(HammingWeightCompute(self.bitsize), x=x)
        out, phase_grad = bb.add(self.phase_oracle, out=out, phase_grad=phase_grad)
        x = bb.add(HammingWeightCompute(self.bitsize).adjoint(), x=x, junk=junk, out=out)
        return {'x': x, 'phase_grad': phase_grad}

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text(f'HWPG_{self.bitsize}(Z^{self.exponent})')
        return super().wire_symbol(reg, idx)


@bloq_example
def _hamming_weight_phasing_via_phase_gradient() -> HammingWeightPhasingViaPhaseGradient:
    hamming_weight_phasing_via_phase_gradient = HammingWeightPhasingViaPhaseGradient(4, np.pi / 2.0)
    # Applying this unitary to |1111> should be the identity, and |0101> will flip the sign.
    return hamming_weight_phasing_via_phase_gradient


_HAMMING_WEIGHT_PHASING_VIA_PHASE_GRADIENT_DOC = BloqDocSpec(
    bloq_cls=HammingWeightPhasingViaPhaseGradient,
    examples=(_hamming_weight_phasing_via_phase_gradient,),
)
