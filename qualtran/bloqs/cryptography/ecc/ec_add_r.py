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

import numpy as np
import sympy
from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    QBit,
    QMontgomeryUInt,
    QUInt,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.data_loading import QROAMClean
from qualtran.drawing import Circle, Text, TextBox, WireSymbol
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.simulation.classical_sim import ClassicalValT
from qualtran.symbolics import is_symbolic, Shaped, SymbolicInt

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
        Roetteler et al. 2017. Algorithm 1 and Figure 10.

        [Microsoft Quantum Crypto (GitHub)](https://github.com/microsoft/QuantumEllipticCurves/blob/dbf4836afaf7a9fab813cbc0970e65af85a6f93a/MicrosoftQuantumCrypto/EllipticCurves.qs#L456).
        Jaques et al. 2020. `DistinctEllipticCurveClassicalPointAddition`.
    """

    n: 'SymbolicInt'
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
        R: The elliptic curve point to add (NOT in montgomery form).
        add_window_size: The number of bits in the ECAdd window.
        mul_window_size: The number of bits in the modular multiplication window.

    Registers:
        ctrl: `window_size` control bits.
        x: The x component of the input elliptic curve point of bitsize `n` in montgomery form.
        y: The y component of the input elliptic curve point of bitsize `n` in montgomery form.

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585).
        Litinski. 2013. Section 1, eq. (3) and (4).
    """

    n: 'SymbolicInt'
    R: ECPoint
    add_window_size: 'SymbolicInt'
    mul_window_size: 'SymbolicInt' = 1

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('ctrl', QBit(), shape=(self.add_window_size,)),
                Register('x', QUInt(self.n)),
                Register('y', QUInt(self.n)),
            ]
        )

    @cached_property
    def qrom(self) -> QROAMClean:
        if is_symbolic(self.n) or is_symbolic(self.add_window_size):
            log_block_sizes = None
            if is_symbolic(self.n) and not is_symbolic(self.add_window_size):
                # We assume that bitsize is much larger than window_size
                log_block_sizes = (0,)
            return QROAMClean(
                [
                    Shaped((2**self.add_window_size,)),
                    Shaped((2**self.add_window_size,)),
                    Shaped((2**self.add_window_size,)),
                ],
                selection_bitsizes=(self.add_window_size,),
                target_bitsizes=(self.n, self.n, self.n),
                log_block_sizes=log_block_sizes,
            )

        cR = self.R
        data_a, data_b, data_lam = [0], [0], [0]
        mon_int = QMontgomeryUInt(self.n, self.R.mod)
        for _ in range(1, 2**self.add_window_size):
            data_a.append(mon_int.uint_to_montgomery(int(cR.x)))
            data_b.append(mon_int.uint_to_montgomery(int(cR.y)))
            lam_num = (3 * cR.x**2 + cR.curve_a) % cR.mod
            lam_denom = (2 * cR.y) % cR.mod
            if lam_denom != 0:
                lam = (lam_num * pow(lam_denom, -1, mod=cR.mod)) % cR.mod
            else:
                lam = 0
            data_lam.append(mon_int.uint_to_montgomery(int(lam)))
            cR = cR + self.R

        return QROAMClean(
            [data_a, data_b, data_lam],
            selection_bitsizes=(self.add_window_size,),
            target_bitsizes=(self.n, self.n, self.n),
        )

    def build_composite_bloq(
        self, bb: 'BloqBuilder', ctrl: 'SoquetT', x: 'Soquet', y: 'Soquet'
    ) -> Dict[str, 'SoquetT']:
        ctrl = bb.join(np.array(ctrl))

        ctrl, a, b, lam_r, *junk = bb.add(self.qrom, selection=ctrl)

        a, b, x, y, lam_r = bb.add(
            # TODO(https://github.com/quantumlib/Qualtran/issues/1476): make ECAdd accept SymbolicInt.
            ECAdd(n=self.n, mod=int(self.R.mod), window_size=self.mul_window_size),
            a=a,
            b=b,
            x=x,
            y=y,
            lam_r=lam_r,
        )

        if junk:
            assert len(junk) == 3
            ctrl = bb.add(
                self.qrom.adjoint(),
                selection=ctrl,
                target0_=a,
                target1_=b,
                target2_=lam_r,
                junk_target0_=junk[0],
                junk_target1_=junk[1],
                junk_target2_=junk[2],
            )
        else:
            ctrl = bb.add(
                self.qrom.adjoint(), selection=ctrl, target0_=a, target1_=b, target2_=lam_r
            )

        return {'ctrl': bb.split(ctrl), 'x': x, 'y': y}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {
            self.qrom: 1,
            # TODO(https://github.com/quantumlib/Qualtran/issues/1476): make ECAdd accept SymbolicInt.
            ECAdd(self.n, int(self.R.mod), self.mul_window_size): 1,
            self.qrom.adjoint(): 1,
        }

    def on_classical_vals(self, ctrl, x, y) -> Dict[str, Union['ClassicalValT', sympy.Expr]]:
        # TODO(https://github.com/quantumlib/Qualtran/issues/1476): make ECAdd accept SymbolicInt.
        dtype = QMontgomeryUInt(self.n, self.R.mod)
        A = ECPoint(
            dtype.montgomery_to_uint(int(x)),
            dtype.montgomery_to_uint(int(y)),
            mod=self.R.mod,
            curve_a=self.R.curve_a,
        )
        ctrls = QUInt(self.n).from_bits(ctrl)
        result: ECPoint = A + (ctrls * self.R)
        return {
            'ctrl': ctrl,
            'x': dtype.uint_to_montgomery(int(result.x)),
            'y': dtype.uint_to_montgomery(int(result.y)),
        }

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
def _ec_window_add_r_small() -> ECWindowAddR:
    n = 16
    P = ECPoint(2, 2, mod=7, curve_a=3)
    ec_window_add_r_small = ECWindowAddR(n=n, R=P, add_window_size=4)
    return ec_window_add_r_small


_EC_WINDOW_ADD_BLOQ_DOC = BloqDocSpec(bloq_cls=ECWindowAddR, examples=[_ec_window_add_r_small])
