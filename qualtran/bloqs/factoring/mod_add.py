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
from typing import Dict, Optional, Set, Union

import sympy
from attrs import frozen
from cirq_ft import TComplexity

from qualtran import Bloq, Register, Signature
from qualtran.bloqs.basic_gates.t_gate import TGate
from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class CtrlScaleModAdd(Bloq):
    """Perform y += x*k mod m for constant k, m and quantum x, y.

    Args:
        k: The constant integer to scale `x` before adding into `y`.
        mod: The modulus of the addition
        bitsize: The size of the two registers.

    Registers:
     - ctrl: The control bit
     - x: The 'source' quantum register containing the integer to be scaled and added to `y`.
     - y: The 'destination' quantum register to which the addition will apply.
    """

    k: Union[int, sympy.Expr]
    mod: Union[int, sympy.Expr]
    bitsize: Union[int, sympy.Expr]

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('ctrl', bitsize=1),
                Register('x', bitsize=self.bitsize),
                Register('y', bitsize=self.bitsize),
            ]
        )

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set['BloqCountT']:
        if ssa is None:
            raise ValueError(f"{self} requires a SympySymbolAllocator")
        k = ssa.new_symbol('k')
        return {(self.bitsize, CtrlModAddK(k=k, bitsize=self.bitsize, mod=self.mod))}

    def t_complexity(self) -> 'TComplexity':
        ((n, bloq),) = self.bloq_counts(SympySymbolAllocator())
        return n * bloq.t_complexity()

    def on_classical_vals(
        self, ctrl: 'ClassicalValT', x: 'ClassicalValT', y: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        if ctrl == 0:
            return {'ctrl': 0, 'x': x, 'y': y}

        assert ctrl == 1, 'Bad ctrl value.'
        y_out = (y + x * self.k) % self.mod
        return {'ctrl': ctrl, 'x': x, 'y': y_out}

    def short_name(self) -> str:
        return f'y += x*{self.k} % {self.mod}'


@frozen
class CtrlModAddK(Bloq):
    """Perform x += k mod m for constant k, m and quantum x.

    Args:
        k: The integer to add to `x`.
        mod: The modulus for the addition.
        bitsize: The bitsize of the `x` register.

    Registers:
        ctrl: The control bit
        x: The register to perform the in-place modular addition.
    """

    k: Union[int, sympy.Expr]
    mod: Union[int, sympy.Expr]
    bitsize: Union[int, sympy.Expr]

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('ctrl', bitsize=1), Register('x', bitsize=self.bitsize)])

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set['BloqCountT']:
        if ssa is None:
            raise ValueError(f"{self} requires a SympySymbolAllocator")
        k = ssa.new_symbol('k')
        return {(5, CtrlAddK(k=k, bitsize=self.bitsize))}

    def t_complexity(self) -> 'TComplexity':
        ((n, bloq),) = self.bloq_counts(SympySymbolAllocator())
        return n * bloq.t_complexity()

    def short_name(self) -> str:
        return f'x += {self.k} % {self.mod}'


@frozen
class CtrlAddK(Bloq):
    """Perform x += k for constant k and quantum x.

    Args:
        k: The integer to add to `x`.
        bitsize: The bitsize of the `x` register.

    Registers:
        ctrl: The control bit
        x: The register to perform the addition.
    """

    k: Union[int, sympy.Expr]
    bitsize: Union[int, sympy.Expr]

    def short_name(self) -> str:
        return f'x += {self.k}'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('ctrl', bitsize=1), Register('x', bitsize=self.bitsize)])

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set['BloqCountT']:
        return {(2 * self.bitsize, TGate())}

    def t_complexity(self) -> 'TComplexity':
        return TComplexity(t=2 * self.bitsize)
