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
from typing import Dict

from attrs import frozen

from qualtran import (
    Bloq,
    BloqBuilder,
    QBit,
    QInt,
    QIntOnesComp,
    QUInt,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.basic_gates import XGate
from qualtran.simulation.classical_sim import ClassicalValT
from qualtran.cirq_interop.bit_tools import iter_bits, iter_bits_twos_complement


@frozen
class ClassicalToQBit(Bloq):
    r"""A 1-bit bloq which converts a classical bit of information into a QBit of information.
        Assumes the input register is initialized to 0.

    Args:
        k: A classical bit value to be represented as a QBit.

    Registers:
        x: A 1-bit input register.
    """

    k: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', QBit())])

    def on_classical_vals(self, x: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        assert self.k == 0 or self.k == 1, "Argument `k` should be a classical bit containing value 0 or 1"
        return {'x': self.k}

    def build_composite_bloq(self, bb: 'BloqBuilder', x: SoquetT) -> Dict[str, 'SoquetT']:

        # Flip bit value for x if the classical bit is 1.
        if self.k == 1:
            x = bb.add(XGate(), q=x)

        # Return the output registers.
        return {'x': x}

    def short_name(self) -> str:
        return f'x = {self.k}'


@frozen
class ClassicalToQInt(Bloq):
    r"""Converts a classical integer into a QInt. Assumes the input register is initialized to 0.

    Args:
        bitsize: Number of bits used to represent the integer.
        k: A classical int value to be represented as a QInt.

    Registers:
        x: A bitsize-sized input register.
    """

    bitsize: int
    k: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', QInt(self.bitsize))])

    def on_classical_vals(self, x: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        return {'x': self.k}

    def build_composite_bloq(self, bb: 'BloqBuilder', x: SoquetT) -> Dict[str, 'SoquetT']:

        # Flip the bit values of x corresponding to the 2's complement representation of integer k.
        x_split = bb.split(x)
        binary_rep = list(iter_bits_twos_complement(self.k, self.bitsize))
        for i in range(self.bitsize):
            if binary_rep[i] == 1:
                x_split[i] = bb.add(XGate(), q=x_split[i])
        x = bb.join(x_split)

        # Return the output registers.
        return {'x': x}

    def short_name(self) -> str:
        return f'x = {self.k}'
    

@frozen
class ClassicalToQUInt(Bloq):
    r"""Converts a classical integer into a QUInt. Assumes the input register is initialized to 0.

    Args:
        bitsize: Number of bits used to represent the unsigned integer.
        k: A classical int value to be represented as a QUInt.

    Registers:
        x: A bitsize-sized input register.
    """

    bitsize: int
    k: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', QUInt(self.bitsize))])

    def on_classical_vals(self, x: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        return {'x': self.k}

    def build_composite_bloq(self, bb: 'BloqBuilder', x: SoquetT) -> Dict[str, 'SoquetT']:

        # Flip the bit values of x corresponding to the representation of the unsigned integer k.
        x_split = bb.split(x)
        binary_rep = list(iter_bits(self.k, self.bitsize, signed=False))
        for i in range(self.bitsize):
            if binary_rep[i] == 1:
                x_split[i] = bb.add(XGate(), q=x_split[i])
        x = bb.join(x_split)

        # Return the output registers.
        return {'x': x}

    def short_name(self) -> str:
        return f'x = {self.k}'
    

@frozen
class ClassicalToQIntOnesComp(Bloq):
    r"""Converts a classical integer into a QIntOnesComp. Assumes the input register is initialized
            to 0.

    Args:
        bitsize: Number of bits used to represent the ones complement integer.
        k: A classical int value to be represented as a QIntOnesComp.

    Registers:
        x: A bitsize-sized input register.
    """

    bitsize: int
    k: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', QIntOnesComp(self.bitsize))])

    def on_classical_vals(self, x: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        return {'x': self.k}

    def build_composite_bloq(self, bb: 'BloqBuilder', x: SoquetT) -> Dict[str, 'SoquetT']:

        # Flip the bit values of x corresponding to the 1's complement representation of integer k.
        x_split = bb.split(x)
        binary_rep = list(iter_bits(self.k, self.bitsize, signed=True))
        for i in range(self.bitsize):
            if binary_rep[i] == 1:
                x_split[i] = bb.add(XGate(), q=x_split[i])
        x = bb.join(x_split)

        # Return the output registers.
        return {'x': x}

    def short_name(self) -> str:
        return f'x = {self.k}'