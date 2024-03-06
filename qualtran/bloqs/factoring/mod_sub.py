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

from qualtran import Bloq, QMontgomeryUInt, Register, Signature, SoquetT
from qualtran.bloqs.arithmetic.addition import SimpleAddConstant
from qualtran.bloqs.basic_gates import CNOT, XGate
from qualtran.bloqs.factoring.mod_add import MontgomeryModAdd
from qualtran.bloqs.mcmt.multi_control_multi_target_pauli import MultiControlX


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

    bitsize: int
    p: int

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
        return {'x': x, 'y': (y - x) % self.p}

    def build_composite_bloq(
        self, bb: 'BloqBuilder', x: SoquetT, y: SoquetT
    ) -> Dict[str, 'SoquetT']:

        # Bit flip all qubits in register x.
        x_split = bb.split(x)
        for i in range(self.bitsize):
            x_split[i] = bb.add(XGate(), q=x_split[i])
        x = bb.join(x_split)

        # Add constant p+1 to the x register.
        x = bb.add(SimpleAddConstant(bitsize=self.bitsize, k=self.p + 1, signed=False, cvs=()), x=x)

        # Perform in-place addition on quantum register y.
        x, y = bb.add(MontgomeryModAdd(bitsize=self.bitsize, p=self.p), x=x, y=y)

        # Add constant -(p+1) to the x register to uncompute the first addition.
        x = bb.add(
            SimpleAddConstant(bitsize=self.bitsize, k=self.p + 1, signed=False, cvs=()).adjoint(),
            x=x,
        )

        # Bit flip all qubits in register x.
        x_split = bb.split(x)
        for i in range(self.bitsize):
            x_split[i] = bb.add(XGate(), q=x_split[i])
        x = bb.join(x_split)

        # Return the output registers.
        return {'x': x, 'y': y}

    def short_name(self) -> str:
        return f'y = y - x mod {self.p}'


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

    bitsize: int
    p: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', QMontgomeryUInt(self.bitsize))])

    def on_classical_vals(self, x: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        return {'x': (-1 * x) % self.p}

    def build_composite_bloq(self, bb: 'BloqBuilder', x: SoquetT) -> Dict[str, 'SoquetT']:

        # Initialize an ancilla qubit to |1>.
        ctrl = bb.allocate(n=1)
        ctrl = bb.add(XGate(), q=ctrl)

        # Perform a multi-controlled bitflip on the ancilla bit if the state of x is the bitstring
        # representing 0.
        cvs = ()
        for i in range(self.bitsize):
            cvs = cvs + (0,)
        x_split = bb.split(x)
        x_split, ctrl = bb.add(MultiControlX(cvs=cvs), ctrls=x_split, x=ctrl)
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
            SimpleAddConstant(bitsize=self.bitsize, k=self.p + 1, cvs=(1,), signed=False),
            ctrls=ctrl_split,
            x=x,
        )
        ctrl = bb.join(ctrl_split)

        # Perform a multi-controlled bitflip on the ancilla bit if the state of x is the bitstring
        # representing 0.
        x_split = bb.split(x)
        x_split, ctrl = bb.add(MultiControlX(cvs=cvs), ctrls=x_split, x=ctrl)
        x = bb.join(x_split)

        # Return the ancilla qubit to the 0 state and free it.
        ctrl = bb.add(XGate(), q=ctrl)
        bb.free(ctrl)

        # Return the output registers.
        return {'x': x}

    def short_name(self) -> str:
        return f'x = -x mod {self.p}'
