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
from typing import Dict, Optional, Tuple, Union

import sympy
from attrs import frozen

from qualtran import Bloq, bloq_example, BloqDocSpec, QBit, QUInt, Register, Signature
from qualtran.drawing import Circle, Text, TextBox, WireSymbol
from qualtran.simulation.classical_sim import ClassicalValT

from .ec_point import ECPoint


@frozen
class ECAddR(Bloq):
    r"""Perform elliptic curve addition of constant `R`.

    Given the constant elliptic curve point $R$ and an input point $A$
    factored into the `x` and `y` registers such that $|A\rangle = |(a_x,a_y)\rangle$,
    this bloq takes

    $$
    |A\rangle \rightarrow |A+R\rangle
    $$

    Args:
        n: The bitsize of the two registers storing the elliptic curve point.
        R: The elliptic curve point to add.

    Registers:
        ctrl: A control bit.
        x: The x component of the input elliptic curve point of bitsize `n`.
        y: The y component of the input elliptic curve point of bitsize `n`.

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585).
        Litinski. 2023. Section 1, eq. (3) and (4).

        [Quantum resource estimates for computing elliptic curve discrete logarithms](https://arxiv.org/abs/1706.06752).
        Roetteler et. al. 2017. Algorithm 1 and Figure 10.

        [https://github.com/microsoft/QuantumEllipticCurves/blob/dbf4836afaf7a9fab813cbc0970e65af85a6f93a/MicrosoftQuantumCrypto/EllipticCurves.qs#L456](QuantumQuantumCrypto).
        `DistinctEllipticCurveClassicalPointAddition`.

    """

    n: int
    R: ECPoint

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [Register('ctrl', QBit()), Register('x', QUInt(self.n)), Register('y', QUInt(self.n))]
        )

    def on_classical_vals(self, ctrl, x, y) -> Dict[str, Union['ClassicalValT', sympy.Expr]]:
        if ctrl == 0:
            return {'ctrl': ctrl, 'x': x, 'y': y}

        A = ECPoint(x, y, mod=self.R.mod, curve_a=self.R.curve_a)
        result: ECPoint = A + self.R
        return {'ctrl': 1, 'x': result.x, 'y': result.y}

    def wire_symbol(self, reg: 'Register', idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        if reg.name == 'ctrl':
            return Circle()
        if reg.name == 'x':
            return TextBox(f'$+{self.R.x}$')
        if reg.name == 'y':
            return TextBox(f'$+{self.R.y}$')
        raise ValueError(f'Unrecognized register name {reg.name}')


@bloq_example
def _ec_add_r() -> ECAddR:
    n, p, Rx, Ry = sympy.symbols('n p R_x R_y')
    ec_add_r = ECAddR(n=n, R=ECPoint(Rx, Ry, mod=p))
    return ec_add_r


@bloq_example
def _ec_add_r_small() -> ECAddR:
    n = 5  # fits our mod = 17
    P = ECPoint(15, 13, mod=17, curve_a=0)
    ec_add_r_small = ECAddR(n=n, R=P)
    return ec_add_r_small


_ECC_ADD_R_BLOQ_DOC = BloqDocSpec(bloq_cls=ECAddR, examples=[_ec_add_r, _ec_add_r_small])


@frozen
class ECWindowAddR(Bloq):
    r"""Perform elliptic curve addition of many multiples of constant `R`.

    This adds R, 2R, ... 2^window_size into the register.

    Args:
        n: The bitsize of the two registers storing the elliptic curve point
        window_size: The number of bits in the window.
        R: The elliptic curve point to add.

    Registers:
        ctrl: `window_size` control bits.
        x: The x component of the input elliptic curve point of bitsize `n`.
        y: The y component of the input elliptic curve point of bitsize `n`.

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585).
        Litinski. 2013. Section 1, eq. (3) and (4).
    """

    n: int
    window_size: int
    R: ECPoint

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('ctrl', QBit(), shape=(self.window_size,)),
                Register('x', QUInt(self.n)),
                Register('y', QUInt(self.n)),
            ]
        )

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text(f'ECWindowAddR({self.n=})')
        if reg.name == 'ctrl':
            return Circle()
        if reg.name == 'x':
            return TextBox(f'$+{self.R.x}$')
        if reg.name == 'y':
            return TextBox(f'$+{self.R.y}$')
        raise ValueError(f'Unrecognized register name {reg.name}')

    def __str__(self):
        return f'ECWindowAddR({self.n=})'


@bloq_example
def _ec_window_add() -> ECWindowAddR:
    n, p = sympy.symbols('n p')
    Rx, Ry = sympy.symbols('Rx Ry')
    ec_window_add = ECWindowAddR(n=n, window_size=3, R=ECPoint(Rx, Ry, mod=p))
    return ec_window_add


_EC_WINDOW_ADD_BLOQ_DOC = BloqDocSpec(bloq_cls=ECWindowAddR, examples=[_ec_window_add])
