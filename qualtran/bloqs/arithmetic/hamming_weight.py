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
from typing import List, Set, TYPE_CHECKING

import attrs
import cirq
from attrs import frozen
from numpy.typing import NDArray

from qualtran import Bloq, GateWithRegisters, QAny, QUInt, Register, Side, Signature
from qualtran.bloqs.mcmt.and_bloq import And
from qualtran.bloqs.util_bloqs import ArbitraryClifford
from qualtran.cirq_interop.t_complexity_protocol import TComplexity

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class HammingWeightCompute(GateWithRegisters):
    r"""A gate to compute the hamming weight of an n-bit register in a new log_{n} bit register.

    Implements $U|x\rangle |0\rangle \rightarrow |x\rangle|\text{hamming\_weight}(x)\rangle$
    using $\alpha$ Toffoli gates and $\alpha$ ancilla qubits, where
    $\alpha = n - \text{hamming\_weight}(n)$ for an n-bit input register.

    Args:
        bitsize: Number of bits in the input register. The allocated output register
            is of size $\log_2(\text{bitsize})$ so it has enough space to hold the hamming weight
            of x.
        uncompute: Whether to uncompute the hamming weight instead.

    Registers:
     - x: A $\text{bitsize}$-sized input register (register x above).
     - junk: A RIGHT ancilla register of size $\text{bitsize} - \text{hamming\_weight(bitsize)}$.
     - out: A RIGHT output register of size $\log_2(\text{bitize})$.

    References:
        [Halving the cost of quantum addition](https://arxiv.org/abs/1709.06648), Page-4
    """

    bitsize: int
    uncompute: bool = False

    @cached_property
    def signature(self):
        side = Side.LEFT if self.uncompute else Side.RIGHT
        jnk_size = self.bitsize - self.bitsize.bit_count()
        out_size = self.bitsize.bit_length()
        return Signature(
            [
                Register('x', QUInt(self.bitsize)),
                Register('junk', QAny(jnk_size), side=side),
                Register('out', QUInt(out_size), side=side),
            ]
        )

    def short_name(self) -> str:
        return "out = x.bit_count()"

    def _three_to_two_adder(self, a, b, c, out) -> cirq.OP_TREE:
        return [
            [cirq.CX(a, b), cirq.CX(a, c)],
            And().on(b, c, out),
            [cirq.CX(a, b), cirq.CX(a, out), cirq.CX(b, c)],
        ]

    def _decompose_using_three_to_two_adders(
        self, x: List[cirq.Qid], junk: List[cirq.Qid], out: List[cirq.Qid]
    ) -> cirq.OP_TREE:
        for out_idx in range(len(out)):
            y = []
            for in_idx in range(0, len(x) - 2, 2):
                a, b, c = x[in_idx], x[in_idx + 1], x[in_idx + 2]
                anc = junk.pop()
                y.append(anc)
                yield self._three_to_two_adder(a, b, c, anc)
            if len(x) % 2 == 1:
                yield cirq.CNOT(x[-1], out[out_idx])
            else:
                anc = junk.pop()
                yield self._three_to_two_adder(x[-2], x[-1], out[out_idx], anc)
                y.append(anc)
            x = [*y]

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        # Qubit order needs to be reversed because the registers store Big Endian representation
        # of integers.
        x: List[cirq.Qid] = [*quregs['x'][::-1]]
        junk: List[cirq.Qid] = [*quregs['junk'][::-1]]
        out: List[cirq.Qid] = [*quregs['out'][::-1]]
        optree = self._decompose_using_three_to_two_adders(x, junk, out)
        return cirq.inverse(optree) if self.uncompute else optree

    def _t_complexity_(self) -> TComplexity:
        and_t = And(uncompute=self.uncompute)._t_complexity_()
        junk_bitsize = self.bitsize - self.bitsize.bit_count()
        num_clifford = junk_bitsize * (5 + and_t.clifford) + self.bitsize.bit_count()
        num_t = junk_bitsize * and_t.t
        return TComplexity(t=num_t, clifford=num_clifford)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {
            (And(uncompute=self.uncompute), self.bitsize),
            (ArbitraryClifford(n=2), 5 * self.bitsize),
        }

    def adjoint(self) -> 'Bloq':
        # TODO: There is nothing special about the `uncompute` construction of this bloq.
        # However: using the default `Adjoint` bloq somehow breaks Cirq simulation of
        # `HammingWeightPhasing`.
        return attrs.evolve(self, uncompute=not self.uncompute)

    def __pow__(self, power: int):
        if power == 1:
            return self
        if power == -1:
            return self.adjoint()
        raise NotImplementedError("HammingWeightCompute.__pow__ defined only for +1/-1.")
