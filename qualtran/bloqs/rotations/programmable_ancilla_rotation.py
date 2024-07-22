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
from typing import Set

import numpy as np
import sympy
from attrs import frozen

from qualtran import Bloq, bloq_example, BloqBuilder, QBit, Register, Side, Signature, SoquetT
from qualtran.bloqs.basic_gates import CNOT, Hadamard, Rz, XGate
from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
from qualtran.symbolics import ceil, is_symbolic, log2, SymbolicFloat, SymbolicInt


@frozen
class RzResourceState(Bloq):
    r"""Resource qubit with state $Rz(\phi) |+\rangle$"""
    angle: SymbolicFloat
    eps: SymbolicFloat = 1e-11

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register("q", QBit(), side=Side.RIGHT)])

    def build_composite_bloq(self, bb: 'BloqBuilder') -> dict[str, 'SoquetT']:
        q = bb.allocate(dtype=QBit())
        q = bb.add(Hadamard(), q=q)
        q = bb.add(Rz(self.angle, self.eps), q=q)
        return {'q': q}


@bloq_example
def _rz_resource_state() -> RzResourceState:
    rz_resource_state = RzResourceState(np.pi / 4)
    return rz_resource_state


@bloq_example
def _rz_resource_state_symb() -> RzResourceState:
    phi = sympy.Symbol(r"\phi")
    rz_resource_state_symb = RzResourceState(phi)
    return rz_resource_state_symb


@frozen
class RzViaProgrammableAncillaRotation(Bloq):
    r"""Single qubit rotation using Rz resource states.

    This bloq applies a single qubit Rz rotation only using Rz resource states and
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
         angle: The angle $\phi$ to apply $Rz(\phi)$ on the input qubit.
         n_rounds: The max number of rounds to attempt the rotation.
         apply_final_correction: Whether to apply an expensive Rz rotation at
                 the end to correct the qubit in case all measurements failed.

    References:
        [Simulating chemistry efficiently on fault-tolerant quantum computers](https://arxiv.org/abs/1204.0567)
        Jones et. al. 2012. Fig 4.
    """

    angle: SymbolicFloat
    eps: SymbolicFloat = 1e-11
    n_rounds: SymbolicInt = 2
    apply_final_correction: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(q=QBit())

    @classmethod
    def from_failure_probability(
        cls, angle: SymbolicFloat, *, max_fail_probability: SymbolicFloat
    ) -> 'RzViaProgrammableAncillaRotation':
        """Applies the rotation `Rz(angle)` except with some specified failure probability.

        Args:
            angle: Rotation angle.
            max_fail_probability: Upper bound on fail probability of the rotation gate.
        """
        n_rounds = ceil(log2(1 / max_fail_probability))
        return cls(angle, n_rounds)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        import cirq

        from qualtran.cirq_interop import CirqGateAsBloq

        resources: Counter[Bloq] = Counter({CNOT(): self.n_rounds, XGate(): self.n_rounds})

        if is_symbolic(self.n_rounds):
            phi = ssa.new_symbol(r"\phi")
            eps = ssa.new_symbol(r"\epsilon")
            resources[RzResourceState(phi, eps)] += self.n_rounds
        else:
            for i in range(int(self.n_rounds)):
                resources[(RzResourceState(2**i * self.angle, eps=self.eps / 2 ** (i + 1)))] += 1

        if self.apply_final_correction:
            resources[Rz(2**self.n_rounds * self.angle, eps=self.eps / 2**self.n_rounds)] += 1

        resources[CirqGateAsBloq(cirq.MeasurementGate(num_qubits=1))] += self.n_rounds

        return set(resources.items())


@bloq_example
def _rz_via_par() -> RzViaProgrammableAncillaRotation:
    rz_via_par = RzViaProgrammableAncillaRotation(np.pi / 4)
    return rz_via_par


@bloq_example
def _rz_via_par_symb() -> RzViaProgrammableAncillaRotation:
    """Paper example.

    References:
        [Simulating chemistry efficiently on fault-tolerant quantum computers](https://arxiv.org/abs/1204.0567)
        Jones et. al. 2012. Fig 4.
    """
    phi, eps = sympy.symbols(r"\phi \epsilon")
    rz_via_par_symb = RzViaProgrammableAncillaRotation(phi, eps=eps, n_rounds=3)
    return rz_via_par_symb


@bloq_example
def _rz_via_par_symb_rounds() -> RzViaProgrammableAncillaRotation:
    """Paper example.

    References:
        [Simulating chemistry efficiently on fault-tolerant quantum computers](https://arxiv.org/abs/1204.0567)
        Jones et. al. 2012. Fig 4.
    """
    phi, n = sympy.symbols(r"\phi n")
    rz_via_par_symb_rounds = RzViaProgrammableAncillaRotation(phi, n_rounds=n)
    return rz_via_par_symb_rounds
