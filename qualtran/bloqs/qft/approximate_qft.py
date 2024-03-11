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
import math
from functools import cached_property
from typing import Callable, Dict

import attrs
import cirq
from numpy.typing import NDArray

from qualtran import GateWithRegisters, QFxp, QUInt, Signature, Bloq, SoquetT, QInt
from qualtran.bloqs.arithmetic import Add
from qualtran.bloqs.arithmetic.multiplication import PlusEqualProduct
from qualtran.bloqs.basic_gates import Hadamard


@attrs.frozen
class ApproximateQFT(Bloq):
    r"""
    An approximate QFT using the phase gradient trick where controlled z-power gates past a user-defined threshold
    are cut off.

    Given an n-bit phase gradient state $|\phi\rangle$ prepared as

    $$
        |\phi\rangle = \frac{1}{\sqrt{2^{n}}} \sum_{k=0}^{2^{n} - 1} \omega_{n}^{-k} |k\rangle
    $$

    The QFT uses exponentially small z-power gates. In practice, it is often sufficient to perform
    an approximate qft, where z-power gates smaller than a certain threshold are dropped.

    Phase gradient rotations can be synthesized via additions into the phase gradient register.
    This leads to significant reductions in T/Toffoli complexity and requires 0 arbitrary
    rotations (given a one-time cost to prepare the gradient register). See the linked reference
    for more details.

    Args:
        bitsize: Size of input register to apply QFT on.
        b: The threshold for the z-power gates. Gates with exponents smaller than b are dropped.
        with_reverse: Whether or not to include the swaps at the end
            of the circuit decomposition that reverse the order of the
            qubits. If True, the swaps are inserted. Defaults to True.
            These are technically necessary in order to perform the
            correct effect, but can almost always be optimized away by just
            performing later operations on different qubits.

    """

    bitsize: int
    b_func: Callable = lambda n: math.ceil(math.log2(n))
    with_reverse: bool = True

    @cached_property
    def signature(self) -> 'Signature':
        b = self.b_func(self.bitsize)
        assert(b > 0, "b_func must return a positive value for b")
        return Signature.build_from_dtypes(
            q=QUInt(self.bitsize), phase_grad=QFxp(b, b)
        )

    def build_composite_bloq(
        self, bb: 'BloqBuilder', q: SoquetT, phase_grad: SoquetT
    ) -> Dict[str, 'SoquetT']:
        qs = bb.split(q)
        phase_grad_qubits = bb.split(phase_grad)
        for i in range(len(qs)):
            qs[i] = bb.add(Hadamard(), q=qs[i])
            addition_bitsize = min(i + 1, len(phase_grad_qubits))
            a = qs[:addition_bitsize]
            b = phase_grad_qubits[:addition_bitsize]
            a, b = bb.add(Add(QUInt(addition_bitsize)), a=a, b=b)
            qs[:addition_bitsize] = a
            phase_grad_qubits[:addition_bitsize] = b
        q = bb.join(qs)
        phase_grad = bb.join(phase_grad_qubits)

        return dict(q=q, phase_grad=phase_grad)
