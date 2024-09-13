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
from qualtran.bloqs.basic_gates import CNOT, Hadamard, XGate, ZPowGate
from qualtran.bloqs.basic_gates._shims import Measure
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.symbolics import ceil, is_symbolic, log2, SymbolicFloat, SymbolicInt


@frozen
class ZPowProgrammedAncilla(Bloq):
    r"""Resource qubit with state $\frac1{\sqrt2} (|0\rangle + e^{i \pi t} |1\rangle)$.

    Args:
        exponent: value of $t$.
        eps: precision of the synthesized state.

    Signature:
        q: the ancilla qubit prepared in the above state.
    """
    exponent: SymbolicFloat
    eps: SymbolicFloat = 1e-11

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register("q", QBit(), side=Side.RIGHT)])

    def build_composite_bloq(self, bb: 'BloqBuilder') -> dict[str, 'SoquetT']:
        q = bb.allocate(dtype=QBit())
        q = bb.add(Hadamard(), q=q)
        q = bb.add(ZPowGate(self.exponent, eps=self.eps), q=q)
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

    This bloq applies a single qubit `Z**t` rotation only using ZPow resource states and
    clifford gates. It is designed to exit early as soon as a measurement succeeds.

    The circuit is described in Fig. 4 of Ref [1].
    The `k`-th round uses an ancilla in state `Z**(2^k t)|+>` and cliffords+measurement to
    probabilistically apply either a `Z**(2^k t)` or `Z**(-2^k t)` with equal probability.
    In the first case we stop, and in the second, we continue with `k+1` to correct the
    wrong sign.

    The T-cost of this bloq is the sum of T-cost of preparing the `n_rounds` ancilla.

    Notes:
        - This bloq uses measurements.
        - To use this Bloq in costing, use the precise number of rounds that are actually
          expected during execution. As Qualtran does not support analyzing measurement-based
          post-selection circuits, the complexity of this Bloq is the worst-case for the
          chosen number of rounds.
          See issue https://github.com/quantumlib/Qualtran/issues/445.
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
        r"""Applies the rotation `Z**t` except with some specified failure probability.

        As each round has success probability 1/2, to achieve a max failure probability $p$,
        the number of rounds is picked as $\ceil{\log_2(1/p)}$.

        Args:
            exponent: Rotation exponent `t`.
            max_fail_probability: Upper bound $p$ on fail probability of the rotation gate.
            eps: The precision of the synthesized rotation.
        """
        n_rounds = ceil(log2(1 / max_fail_probability))
        return cls(exponent=exponent, eps=eps, n_rounds=n_rounds)

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        resources: Counter[Bloq] = Counter({CNOT(): self.n_rounds, XGate(): self.n_rounds})

        n_rz = self.n_rounds + (1 if self.apply_final_correction else 0)

        if is_symbolic(self.n_rounds):
            phi = ssa.new_symbol(r"\phi")
            eps = self.eps / n_rz
            resources[ZPowProgrammedAncilla(phi, eps)] += self.n_rounds
        else:
            for i in range(int(self.n_rounds)):
                resources[ZPowProgrammedAncilla(2**i * self.exponent, eps=self.eps / n_rz)] += 1

        if self.apply_final_correction:
            resources[ZPowGate(2**self.n_rounds * self.exponent, eps=self.eps / n_rz)] += 1

        resources[Measure()] += self.n_rounds

        return resources


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
