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
from collections import Counter
from functools import cached_property

import numpy as np
import sympy
from attrs import field, frozen

from qualtran import Bloq, bloq_example, BloqBuilder, QBit, Register, Side, Signature, SoquetT
from qualtran.bloqs.basic_gates import CNOT, Hadamard, Rz, XGate, ZPowGate
from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
from qualtran.symbolics import ceil, is_symbolic, log2, SymbolicFloat, SymbolicInt


@frozen
class ZPowProgrammedAncilla(Bloq):
    r"""Resource qubit with state $\frac1{\sqrt2} (|0\rangle + e^{i \pi t} |1\rangle)$.

    Args:
        exponent: value of $t$.
        eps: precision of the synthesized state.
    """
    exponent: SymbolicFloat
    eps: SymbolicFloat = 1e-11

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register("q", QBit(), side=Side.RIGHT)])

    def build_composite_bloq(self, bb: 'BloqBuilder') -> dict[str, 'SoquetT']:
        q = bb.allocate(dtype=QBit())
        q = bb.add(Hadamard(), q=q)
        q = bb.add(ZPowGate(self.exponent, self.eps), q=q)
        return {'q': q}


@bloq_example
def _zpow_programmed_ancilla() -> ZPowProgrammedAncilla:
    zpow_programmed_ancilla = ZPowProgrammedAncilla(np.pi / 4)
    return zpow_programmed_ancilla


@bloq_example
def _zpow_programmed_ancilla_symb() -> ZPowProgrammedAncilla:
    t = sympy.Symbol(r"t")
    zpow_programmed_ancilla_symb = ZPowProgrammedAncilla(t)
    return zpow_programmed_ancilla_symb


@frozen
class ZPowUsingProgrammedAncilla(Bloq):
    r"""Single qubit ZPow rotation using resource states.

    This bloq applies a single qubit Z**t rotation only using ZPow resource states and
    clifford gates. It is designed to exit early as soon as a measurement succeeds.

    Notes:
        - This bloq uses measurements.
        - To use this Bloq in costing, use the precise number of rounds that are actually
          expected during execution. As Qualtran does not support analyzing measurement-based
          post-selection circuits, the complexity of this Bloq is the worst-case for the
          chosen number of rounds.
        - `apply_final_correction` defaults to False, therefore the resource estimates assume
          that one of the measurements succeeded and the process quit early. To get an exact
          channel, instead set this parameter to True.

    Args:
         exponent: The value $t$ to apply $Z**t$ on the input qubit.
         eps: The precision of the synthesized rotation.
         n_rounds: The max number of rounds to attempt the rotation.
         apply_final_correction: Whether to apply an expensive ZPow rotation at
                 the end to correct the qubit in case all measurements failed.

    References:
        [Simulating chemistry efficiently on fault-tolerant quantum computers](https://arxiv.org/abs/1204.0567)
        Jones et. al. 2012. Fig 4.
    """

    exponent: SymbolicFloat
    eps: SymbolicFloat = 1e-11
    n_rounds: SymbolicInt = field(default=2, kw_only=True)
    apply_final_correction: bool = field(default=False, kw_only=True)

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(q=QBit())

    @classmethod
    def from_failure_probability(
        cls,
        exponent: SymbolicFloat,
        *,
        max_fail_probability: SymbolicFloat,
        eps: SymbolicFloat = 1e-11,
    ) -> 'ZPowUsingProgrammedAncilla':
        """Applies the rotation `Rz(angle)` except with some specified failure probability.

        Args:
            exponent: Rotation exponent.
            max_fail_probability: Upper bound on fail probability of the rotation gate.
            eps: The precision of the synthesized rotation.
        """
        n_rounds = ceil(log2(1 / max_fail_probability))
        return cls(exponent=exponent, eps=eps, n_rounds=n_rounds)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> set['BloqCountT']:
        import cirq

        from qualtran.cirq_interop import CirqGateAsBloq

        resources: Counter[Bloq] = Counter({CNOT(): self.n_rounds, XGate(): self.n_rounds})

        n_rz = self.n_rounds + (1 if self.apply_final_correction else 0)

        if is_symbolic(self.n_rounds):
            phi = ssa.new_symbol(r"\phi")
            eps = ssa.new_symbol(r"\epsilon")
            resources[ZPowProgrammedAncilla(phi, eps)] += self.n_rounds
        else:
            for i in range(int(self.n_rounds)):
                resources[ZPowProgrammedAncilla(2**i * self.exponent, eps=self.eps / n_rz)] += 1

        if self.apply_final_correction:
            resources[Rz(2**self.n_rounds * self.exponent, eps=self.eps / n_rz)] += 1

        resources[CirqGateAsBloq(cirq.MeasurementGate(num_qubits=1))] += self.n_rounds

        return set(resources.items())


@bloq_example
def _zpow_using_programmed_ancilla() -> ZPowUsingProgrammedAncilla:
    zpow_using_programmed_ancilla = ZPowUsingProgrammedAncilla(np.pi / 4)
    return zpow_using_programmed_ancilla


@bloq_example
def _zpow_using_programmed_ancilla_symb() -> ZPowUsingProgrammedAncilla:
    """Paper example.

    References:
        [Simulating chemistry efficiently on fault-tolerant quantum computers](https://arxiv.org/abs/1204.0567)
        Jones et. al. 2012. Fig 4.
    """
    phi, eps = sympy.symbols(r"\phi \epsilon")
    zpow_using_programmed_ancilla_symb = ZPowUsingProgrammedAncilla(
        phi / sympy.pi, eps=eps, n_rounds=3
    )
    return zpow_using_programmed_ancilla_symb


@bloq_example
def _zpow_using_programmed_ancilla_symb_rounds() -> ZPowUsingProgrammedAncilla:
    """Paper example.

    References:
        [Simulating chemistry efficiently on fault-tolerant quantum computers](https://arxiv.org/abs/1204.0567)
        Jones et. al. 2012. Fig 4.
    """
    phi, n = sympy.symbols(r"\phi n")
    zpow_using_programmed_ancilla_symb_rounds = ZPowUsingProgrammedAncilla(
        phi / sympy.pi, n_rounds=n
    )
    return zpow_using_programmed_ancilla_symb_rounds
