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
from typing import Dict, Iterator

import cirq
import sympy
from attrs import frozen
from numpy.typing import NDArray

from qualtran import bloq_example, BloqDocSpec, GateWithRegisters, Signature
from qualtran.simulation.classical_sim import ClassicalValT
from qualtran.symbolics import SymbolicInt


@frozen
class MultiTargetCNOT(GateWithRegisters):
    r"""Implements single control, multi-target $C[X^{\otimes n}]$ gate.

    Implements $|0><0| I + |1><1| X^{\otimes n}$ using a circuit of depth $2\log(n) + 1$
    containing only CNOT gates.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis](https://arxiv.org/abs/1812.00954).
        Appendix B.1.
    """

    bitsize: 'SymbolicInt'

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(control=1, targets=self.bitsize)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        control: NDArray[cirq.Qid],  # type: ignore[type-var]
        targets: NDArray[cirq.Qid],  # type: ignore[type-var]
    ):
        def cnots_for_depth_i(i: int, q: NDArray[cirq.Qid]) -> Iterator[cirq.OP_TREE]:  # type: ignore[type-var]
            for c, t in zip(q[: 2**i], q[2**i : min(len(q), 2 ** (i + 1))]):
                yield cirq.CNOT(c, t)

        depth = len(targets).bit_length()
        for i in range(depth):
            yield cirq.Moment(cnots_for_depth_i(depth - i - 1, targets))
        yield cirq.CNOT(*control, targets[0])
        for i in range(depth):
            yield cirq.Moment(cnots_for_depth_i(i, targets))

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        if isinstance(self.bitsize, sympy.Expr):
            raise ValueError(f'Symbolic bitsize {self.bitsize} not supported')
        return cirq.CircuitDiagramInfo(wire_symbols=["@"] + ["X"] * self.bitsize)

    def on_classical_vals(
        self, control: 'ClassicalValT', targets: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        if control:
            targets = (2**self.bitsize - 1) ^ targets
        return {'control': control, 'targets': targets}


@bloq_example
def _c_multi_not_symb() -> MultiTargetCNOT:
    n = sympy.Symbol('n')
    c_multi_not_symb = MultiTargetCNOT(bitsize=n)
    return c_multi_not_symb


@bloq_example
def _c_multi_not() -> MultiTargetCNOT:
    c_multi_not = MultiTargetCNOT(bitsize=5)
    return c_multi_not


_C_MULTI_NOT_DOC = BloqDocSpec(bloq_cls=MultiTargetCNOT, examples=(_c_multi_not_symb, _c_multi_not))
