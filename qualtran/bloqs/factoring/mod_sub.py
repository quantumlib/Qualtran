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
from typing import Dict, Set, TYPE_CHECKING, Union

from attrs import frozen

from qualtran import Bloq, QMontgomeryUInt, Register, Signature, Soquet, SoquetT
from qualtran.bloqs.arithmetic.addition import AddK
from qualtran.bloqs.basic_gates import CNOT, XGate
from qualtran.bloqs.mcmt import MultiControlX
from qualtran.bloqs.mod_arithmetic import ModAdd
from qualtran.symbolics import HasLength

if TYPE_CHECKING:
    from qualtran import BloqBuilder
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT
    from qualtran.symbolics import SymbolicInt


@frozen
class MontgomeryModSub(Bloq):
    r"""An n-bit modular subtraction gate.

    This gate is designed to operate on integers in the Montgomery form.
    Implements |x>|y> => |x>|y - x % p> using $6n$ Toffoli
    gates.

    Args:
        bitsize: Number of bits used to represent each integer.
        p: The modulus for the subtraction.

    Registers:
        x: A bitsize-sized input register (register x above).
        y: A bitsize-sized input/output register (register y above).

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585)
        Fig 6c and 8
    """

    bitsize: 'SymbolicInt'
    p: 'SymbolicInt'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('x', QMontgomeryUInt(self.bitsize)),
                Register('y', QMontgomeryUInt(self.bitsize)),
            ]
        )

    def on_classical_vals(
        self, x: 'ClassicalValT', y: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        if x < self.p and y < self.p:
            return {'x': x, 'y': (y - x) % self.p}
        return {'x': x, 'y': y}

    def build_composite_bloq(self, bb: 'BloqBuilder', x: Soquet, y: Soquet) -> Dict[str, 'SoquetT']:
        if not isinstance(self.bitsize, int):
            raise NotImplementedError(f'symbolic decomposition is not supported for {self}')
        # Bit flip all qubits in register x.
        x_split = bb.split(x)
        for i in range(self.bitsize):
            x_split[i] = bb.add(XGate(), q=x_split[i])
        x = bb.join(x_split, dtype=QMontgomeryUInt(self.bitsize))

        # Add constant p+1 to the x register.
        x = bb.add(AddK(bitsize=self.bitsize, k=self.p + 1, signed=False, cvs=()), x=x)

        # Perform in-place addition on quantum register y.
        x, y = bb.add(ModAdd(bitsize=self.bitsize, mod=self.p), x=x, y=y)

        # Add constant -(p+1) to the x register to uncompute the first addition.
        x = bb.add(AddK(bitsize=self.bitsize, k=self.p + 1, signed=False, cvs=()).adjoint(), x=x)

        # Bit flip all qubits in register x.
        x_split = bb.split(x)
        for i in range(self.bitsize):
            x_split[i] = bb.add(XGate(), q=x_split[i])
        x = bb.join(x_split, dtype=QMontgomeryUInt(self.bitsize))

        # Return the output registers.
        return {'x': x, 'y': y}

    def pretty_name(self) -> str:
        return f'y = y - x mod {self.p}'

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {
            (XGate(), 2 * self.bitsize),
            (AddK(self.bitsize, self.p + 1, signed=False), 1),
            (ModAdd(self.bitsize, self.p), 1),
            (AddK(self.bitsize, self.p + 1, signed=False).adjoint(), 1),
        }


@frozen
class MontgomeryModNeg(Bloq):
    r"""An n-bit modular negation gate.

    This gate is designed to operate on integers in the Montgomery form.
    Implements |x> => |-x % p> using $2n$ Toffoli gates.

    Args:
        bitsize: Number of bits used to represent each integer.
        p: The modulus for the negation.

    Registers:
        x: A bitsize-sized input register (register x above).

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585)
        Fig 6b and 8
    """

    bitsize: 'SymbolicInt'
    p: 'SymbolicInt'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', QMontgomeryUInt(self.bitsize))])

    def on_classical_vals(self, x: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        return {'x': (-1 * x) % self.p}

    def build_composite_bloq(self, bb: 'BloqBuilder', x: Soquet) -> Dict[str, 'SoquetT']:
        if not isinstance(self.bitsize, int):
            raise NotImplementedError(f'symbolic decomposition is not supported for {self}')
        # Initialize an ancilla qubit to |1>.
        ctrl = bb.allocate(n=1)
        ctrl = bb.add(XGate(), q=ctrl)

        # Perform a multi-controlled bitflip on the ancilla bit if the state of x is the bitstring
        # representing 0.
        cvs = tuple([0] * self.bitsize)
        x_split = bb.split(x)
        x_split, ctrl = bb.add(MultiControlX(cvs=cvs), controls=x_split, target=ctrl)
        x = bb.join(x_split)

        # Bitflips all qubits if the ctrl bit is set to 1 (the input x register is not in the 0
        # state).
        x_split = bb.split(x)
        for i in range(self.bitsize):
            ctrl, x_split[i] = bb.add(CNOT(), ctrl=ctrl, target=x_split[i])
        x = bb.join(x_split)

        # Add constant p+1 to the x register.
        ctrl_split = bb.split(ctrl)
        ctrl_split, x = bb.add(
            AddK(bitsize=self.bitsize, k=self.p + 1, cvs=(1,), signed=False), ctrls=ctrl_split, x=x
        )
        ctrl = bb.join(ctrl_split)

        # Perform a multi-controlled bitflip on the ancilla bit if the state of x is the bitstring
        # representing 0.
        x_split = bb.split(x)
        x_split, ctrl = bb.add(MultiControlX(cvs=cvs), controls=x_split, target=ctrl)
        x = bb.join(x_split)

        # Return the ancilla qubit to the 0 state and free it.
        ctrl = bb.add(XGate(), q=ctrl)
        bb.free(ctrl)

        # Return the output registers.
        return {'x': x}

    def pretty_name(self) -> str:
        return f'x = -x mod {self.p}'

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        cvs: Union[list[int], HasLength]
        if isinstance(self.bitsize, int):
            cvs = [0] * self.bitsize
        else:
            cvs = HasLength(self.bitsize)
        return {
            (XGate(), 2),
            (MultiControlX(cvs=cvs), 2),
            (CNOT(), self.bitsize),
            (AddK(bitsize=self.bitsize, k=self.p + 1, cvs=(1,), signed=False), 1),
        }
