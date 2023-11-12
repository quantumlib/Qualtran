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
from typing import Optional, Set, Union

import sympy
from attrs import frozen

from qualtran import Bloq, Register, Signature
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class ModSub(Bloq):
    r"""An n-bit modular subtraction gate.

    Implements $U|x\rangle|y\rangle \rightarrow |x\rangle|y - x \mod p\rangle$ using $6n$ Toffoli
    gates.

    Args:
        bitsize: Number of bits used to represent each integer.
        p: The modulus for the subtraction.

    Registers:
        x: A bitsize-sized input register (register x above).
        y: A bitsize-sized input/output register (register y above).

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585)
        Fig 6 and 8
    """

    bitsize: Union[int, sympy.Expr]
    p: Union[int, sympy.Expr]

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', bitsize=self.bitsize), Register('y', bitsize=self.bitsize)])

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set['BloqCountT']:
        return {(6 * self.bitsize, Toffoli())}

    def short_name(self) -> str:
        return f'y = y - x mod {self.p}'


@frozen
class ModNeg(Bloq):
    r"""An n-bit modular negation gate.

    Implements $U|x\rangle \rightarrow |-x \mod p\rangle$ using $2n$ Toffoli gates.

    Args:
        bitsize: Number of bits used to represent each integer.
        p: The modulus for the negation.

    Registers:
        x: A bitsize-sized input register (register x above).

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585)
        Fig 6 and 8
    """

    bitsize: Union[int, sympy.Expr]
    p: Union[int, sympy.Expr]

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', bitsize=self.bitsize), Register('p', bitsize=self.bitsize)])

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set['BloqCountT']:
        return {(2 * self.bitsize, Toffoli())}

    def short_name(self) -> str:
        return f'x = -x mod {self.p}'
