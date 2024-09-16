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

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    DecomposeTypeError,
    QBit,
    QUInt,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.arithmetic import XorK
from qualtran.bloqs.basic_gates import IntEffect, IntState
from qualtran.bloqs.bookkeeping import Free
from qualtran.bloqs.data_loading import QROAMClean
from qualtran.drawing import Circle, Text, TextBox, WireSymbol
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.simulation.classical_sim import ClassicalValT

from .ec_add import ECAdd
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

    def build_composite_bloq(
        self, bb: 'BloqBuilder', ctrl: Soquet, x: Soquet, y: Soquet
    ) -> Dict[str, 'SoquetT']:
        if isinstance(self.n, sympy.Expr):
            raise DecomposeTypeError("Cannot decompose symbolic `n`.")

        a = bb.add(IntState(bitsize=self.n, val=0))
        b = bb.add(IntState(bitsize=self.n, val=0))

        ctrl, a = bb.add(XorK(QUInt(bitsize=self.n), self.R.x).controlled(), ctrl=ctrl, x=a)
        ctrl, b = bb.add(XorK(QUInt(bitsize=self.n), self.R.y).controlled(), ctrl=ctrl, x=b)

        lam_num = (3 * self.R.x**2 + self.R.curve_a) % self.R.mod
        lam_denom = (2 * self.R.y) % self.R.mod

        lam = (lam_num * pow(lam_denom, -1, mod=self.R.mod)) % self.R.mod
        lam_r = bb.add(IntState(bitsize=self.n, val=lam))

        a, b, x, y, lam_r = bb.add(ECAdd(self.n, self.R.mod), a=a, b=b, x=x, y=y, lam_r=lam_r)

        ctrl, a = bb.add(XorK(QUInt(bitsize=self.n), self.R.x).controlled(), ctrl=ctrl, x=a)
        ctrl, b = bb.add(XorK(QUInt(bitsize=self.n), self.R.y).controlled(), ctrl=ctrl, x=b)

        bb.add(Free(QUInt(self.n)), reg=a)
        bb.add(Free(QUInt(self.n)), reg=b)
        bb.add(IntEffect(bitsize=self.n, val=lam), val=lam_r)

        return {'ctrl': ctrl, 'x': x, 'y': y}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {
            (XorK(QUInt(self.n), self.R.x).controlled(), 2),
            (XorK(QUInt(self.n), self.R.y).controlled(), 2),
            (ECAdd(self.n, self.R.mod), 1),
        }

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

    def build_composite_bloq(
        self, bb: 'BloqBuilder', ctrl: 'SoquetT', x: Soquet, y: Soquet
    ) -> Dict[str, 'SoquetT']:
        ctrl = bb.join(ctrl)

        cR = self.R
        lam_num = (3 * self.R.x**2 + self.R.curve_a) % self.R.mod
        lam_denom = (2 * self.R.y) % self.R.mod
        lam = (lam_num * pow(lam_denom, -1, mod=self.R.mod)) % self.R.mod
        data_a, data_b, data_lam = [0, self.R.x], [0, self.R.y], [0, lam]
        for _ in range(2, 2**self.window_size):
            cR = cR + self.R
            data_a.append(cR.x)
            data_b.append(cR.y)
            lam_num = (3 * cR.x**2 + cR.curve_a) % cR.mod
            lam_denom = (2 * cR.y) % cR.mod
            lam = (lam_num * pow(lam_denom, -1, mod=cR.mod)) % cR.mod
            data_lam.append(lam)

        ctrl, a, b, lam_r = bb.add(
            QROAMClean(
                [data_a, data_b, data_lam],
                selection_bitsizes=(self.window_size,),
                target_bitsizes=(self.n, self.n, self.n),
            ),
            selection=ctrl,
        )

        a, b, x, y, lam_r = bb.add(
            ECAdd(n=self.n, mod=self.R.mod, window_size=self.window_size),
            a=a,
            b=b,
            x=x,
            y=y,
            lam_r=lam_r,
        )

        ctrl = bb.add(
            QROAMClean(
                [data_a, data_b, data_lam],
                selection_bitsizes=(self.window_size,),
                target_bitsizes=(self.n, self.n, self.n),
            ).adjoint(),
            selection=ctrl,
            target0_=a,
            target1_=b,
            target2_=lam_r,
        )

        return {'ctrl': bb.split(ctrl), 'x': x, 'y': y}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        cR = self.R
        lam_num = (3 * self.R.x**2 + self.R.curve_a) % self.R.mod
        lam_denom = (2 * self.R.y) % self.R.mod
        lam = (lam_num * pow(lam_denom, -1, mod=self.R.mod)) % self.R.mod
        data_a, data_b, data_lam = [0, self.R.x], [0, self.R.y], [0, lam]
        for _ in range(2, 2**self.window_size):
            cR = cR + self.R
            data_a.append(cR.x)
            data_b.append(cR.y)
            lam_num = (3 * cR.x**2 + cR.curve_a) % cR.mod
            lam_denom = (2 * cR.y) % cR.mod
            lam = (lam_num * pow(lam_denom, -1, mod=cR.mod)) % cR.mod
            data_lam.append(lam)
        return {
            (
                QROAMClean(
                    [data_a, data_b, data_lam],
                    selection_bitsizes=(self.window_size,),
                    target_bitsizes=(self.n, self.n, self.n),
                ),
                1,
            ),
            (ECAdd(self.n, self.R.mod, self.window_size), 1),
            (
                QROAMClean(
                    [data_a, data_b, data_lam],
                    selection_bitsizes=(self.window_size,),
                    target_bitsizes=(self.n, self.n, self.n),
                ).adjoint(),
                1,
            ),
        }

    def on_classical_vals(self, ctrl, x, y) -> Dict[str, Union['ClassicalValT', sympy.Expr]]:
        A = ECPoint(x, y, mod=self.R.mod, curve_a=self.R.curve_a)
        result: ECPoint = A + (ctrl * self.R)
        return {'ctrl': ctrl, 'x': result.x, 'y': result.y}

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


@bloq_example
def _ec_window_add() -> ECWindowAddR:
    n, p = sympy.symbols('n p')
    Rx, Ry = sympy.symbols('Rx Ry')
    ec_window_add = ECWindowAddR(n=n, window_size=3, R=ECPoint(Rx, Ry, mod=p))
    return ec_window_add


_EC_WINDOW_ADD_BLOQ_DOC = BloqDocSpec(bloq_cls=ECWindowAddR, examples=[_ec_window_add])
