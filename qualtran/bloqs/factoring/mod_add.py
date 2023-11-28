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
from typing import Dict, Set, Union

import numpy as np
import sympy
from attrs import frozen

from qualtran import Bloq, Register, Signature, SoquetT
from qualtran.bloqs.basic_gates import Toffoli, TGate, XGate
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
from qualtran.simulation.classical_sim import ClassicalValT
from qualtran.bloqs.arithmetic.addition import Add, SimpleAddConstant
from qualtran.bloqs.arithmetic.comparison import GreaterThan


@frozen
class CtrlScaleModAdd(Bloq):
    """Perform y += x*k mod m for constant k, m and quantum x, y.

    Args:
        k: The constant integer to scale `x` before adding into `y`.
        mod: The modulus of the addition
        bitsize: The size of the two registers.

    Registers:
        ctrl: The control bit
        x: The 'source' quantum register containing the integer to be scaled and added to `y`.
        y: The 'destination' quantum register to which the addition will apply.
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

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        k = ssa.new_symbol('k')
        return {(CtrlModAddK(k=k, bitsize=self.bitsize, mod=self.mod), self.bitsize)}

    def t_complexity(self) -> 'TComplexity':
        ((bloq, n),) = self.bloq_counts().items()
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

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        k = ssa.new_symbol('k')
        return {(CtrlAddK(k=k, bitsize=self.bitsize), 5)}

    def t_complexity(self) -> 'TComplexity':
        ((bloq, n),) = self.bloq_counts().items()
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

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(TGate(), 2 * self.bitsize)}

    def t_complexity(self) -> 'TComplexity':
        return TComplexity(t=2 * self.bitsize)


@frozen
class ModAdd(Bloq):
    r"""An n-bit modular addition gate.

    This gate is designed to operate on integers in the Montogomery form.

    Implements $U|x\rangle|y\rangle \rightarrow |x\rangle|y + x \mod p\rangle$ using $4n$ Toffoli
    gates.

    Args:
        bitsize: Number of bits used to represent each integer.
        p: The modulus for the addition.

    Registers:
        x: A bitsize-sized input register (register x above).
        y: A bitsize-sized input/output register (register y above).

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585) Fig 6a and 8
    """

    bitsize: int
    p: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', bitsize=self.bitsize), Register('y', bitsize=self.bitsize)])

    def build_composite_bloq(self, bb: 'BloqBuilder', **regs: SoquetT) -> Dict[str, 'SoquetT']:

        # Assign registers to variables and allocate ancilla bits for use in addition.
        x = regs['x']
        y = regs['y']
        junk_bit = bb.allocate(n=1)
        sign = bb.allocate(n=1)

        # Join ancilla bits to x and y registers in order to be able to compute addition of
        # bitsize+1 registers. This allows us to keep track of the sign of the y register after a
        # constant subtraction circuit.
        x_split = bb.split(x)
        y_split = bb.split(y)
        x = bb.join(np.concatenate([x_split, [junk_bit]]))
        y = bb.join(np.concatenate([y_split, [sign]]))

        # Perform in-place addition on quantum register y.
        x, y = bb.add(Add(bitsize=self.bitsize + 1), a=x, b=y)

        # Temporary solution to equalize the bitlength of the x and y registers for Add().
        x_split = bb.split(x)
        junk_bit = x_split[-1]
        x = bb.join(x_split[0:-1])

        # Add constant -p to the y register.
        y = bb.add(
            SimpleAddConstant(bitsize=self.bitsize + 1, k=-1 * self.p, signed=True, cvs=()), x=y
        )

        # Controlled addition of classical constant p if the sign of y after the last addition is
        # negative.
        y_split = bb.split(y)
        sign = y_split[-1]
        y = bb.join(y_split[0:-1])
        sign, y = bb.add(
            SimpleAddConstant(bitsize=self.bitsize, k=self.p, signed=True, cvs=(1)), ctrl=sign, x=y
        )

        # Check if x < y; if yes flip the bit of the signed ancilla bit. Then bitflip the sign bit
        # again before freeing.
        x, y, sign = bb.add(GreaterThan(bitsize=self.bitsize, signed=False), a=x, b=y, target=sign)
        sign = bb.add(XGate(), q=sign)

        # Free the ancilla qubits.
        junk_bit = bb.free(junk_bit)
        sign = bb.free(sign)

        # Return the output registers.
        return {'x': x, 'y': y}

    def short_name(self) -> str:
        return f'y = y + x mod {self.p}'
