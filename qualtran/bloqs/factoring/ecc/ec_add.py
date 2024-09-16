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
from typing import Dict, Union

import numpy as np
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
    QMontgomeryUInt,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.arithmetic._shims import MultiCToffoli
from qualtran.bloqs.arithmetic.comparison import Equals
from qualtran.bloqs.basic_gates import IntState, ZeroState, CNOT, Toffoli
from qualtran.bloqs.mod_arithmetic import (
    CModAdd,
    CModNeg,
    CModSub,
    DirtyOutOfPlaceMontgomeryModMul,
    ModAdd,
    ModDbl,
    ModNeg,
    ModSub,
)
from qualtran.bloqs.bookkeeping import Free
from qualtran.bloqs.mcmt import MultiAnd, MultiControlX, MultiTargetCNOT
from qualtran.bloqs.mod_arithmetic._shims import ModDbl, ModInv, ModMul
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator

from .ec_point import ECPoint


@frozen
class ECAdd(Bloq):
    r"""Add two elliptic curve points.

    This takes elliptic curve points given by (a, b) and (x, y)
    and outputs the sum (x_r, y_r) in the second pair of registers.

    Args:
        n: The bitsize of the two registers storing the elliptic curve point
        mod: The modulus of the field in which we do the addition.

    Registers:
        a: The x component of the first input elliptic curve point of bitsize `n`.
        b: The y component of the first input elliptic curve point of bitsize `n`.
        x: The x component of the second input elliptic curve point of bitsize `n`, which
           will contain the x component of the resultant curve point.
        y: The y component of the second input elliptic curve point of bitsize `n`, which
           will contain the y component of the resultant curve point.
        lam: The precomputed lambda slope used in the addition operation.

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585).
        Litinski. 2023. Fig 5.
    """

    n: int
    mod: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('a', QUInt(self.n)),
                Register('b', QUInt(self.n)),
                Register('x', QUInt(self.n)),
                Register('y', QUInt(self.n)),
                Register('lam_r', QUInt(self.n)),
            ]
        )

    def build_composite_bloq(
        self, bb: 'BloqBuilder', a: Soquet, b: Soquet, x: Soquet, y: Soquet, lam_r: Soquet
    ) -> Dict[str, 'SoquetT']:
        if isinstance(self.n, sympy.Expr):
            raise DecomposeTypeError("Cannot decompose symbolic `n`.")

        # Step 0: Initialize ancilla qubits to the zero state.
        f_1 = bb.add(ZeroState())
        f_2 = bb.add(ZeroState())
        f_3 = bb.add(ZeroState())
        f_4 = bb.add(ZeroState())
        ctrl = bb.add(ZeroState())
        z_1 = bb.add(IntState(bitsize=self.n, val=0))
        z_2 = bb.add(IntState(bitsize=self.n, val=0))
        z_3 = bb.add(IntState(bitsize=self.n, val=0))
        z_4 = bb.add(IntState(bitsize=self.n, val=0))
        lam = bb.add(IntState(bitsize=self.n, val=0))

        # Step 1:
        a, x, f_1 = bb.add(Equals(self.n), x=a, y=x, target=f_1)
        y = bb.add(ModNeg(QMontgomeryUInt(self.n), mod=self.mod), x=y)
        b, y, f_2 = bb.add(Equals(self.n), x=b, y=y, target=f_2)
        y = bb.add(ModNeg(QMontgomeryUInt(self.n), mod=self.mod), x=y)

        a_arr = bb.split(a)
        b_arr = bb.split(b)
        ab_arr = np.concatenate((a_arr, b_arr), axis=None)
        ab_arr, f_3 = bb.add(MultiControlX(cvs=[0] * 2 * self.n), controls=ab_arr, target=f_3)
        ab_arr = np.split(ab_arr, 2)
        a = bb.join(ab_arr[0], dtype=QMontgomeryUInt(bitsize=self.n))
        b = bb.join(ab_arr[1], dtype=QMontgomeryUInt(bitsize=self.n))

        x_arr = bb.split(x)
        y_arr = bb.split(y)
        xy_arr = np.concatenate((x_arr, y_arr), axis=None)
        xy_arr, f_4 = bb.add(MultiControlX(cvs=[0] * 2 * self.n), controls=xy_arr, target=f_4)
        xy_arr = np.split(xy_arr, 2)
        x = bb.join(xy_arr[0], dtype=QMontgomeryUInt(bitsize=self.n))
        y = bb.join(xy_arr[1], dtype=QMontgomeryUInt(bitsize=self.n))

        f_ctrls = [f_2, f_3, f_4]
        f_ctrls, ctrl = bb.add(MultiControlX(cvs=[0] * 3), controls=f_ctrls, target=ctrl)
        f_2 = f_ctrls[0]
        f_3 = f_ctrls[1]
        f_4 = f_ctrls[2]

        # Step 2:
        a, x = bb.add(ModSub(QMontgomeryUInt(self.n), mod=self.mod), x=a, y=x)
        ctrl, b, y = bb.add(CModSub(QMontgomeryUInt(self.n), mod=self.mod), ctrl=ctrl, x=b, y=y)
        # ModInv (needs 2n for out/garbage register)
        # ModMult (needs more garbage register)

        z_4_split = bb.split(z_4)
        lam_split = bb.split(lam)
        for i in range(self.n):
            ctrls = [f_1, ctrl, z_4_split[i]]
            ctrls, lam_split[i] = bb.add(
                MultiControlX(cvs=[0, 1, 1]), controls=ctrls, target=lam_split[i]
            )
            f_1 = ctrls[0]
            ctrl = ctrls[1]
            z_4_split[i] = ctrls[2]
        z_4 = bb.join(z_4_split, dtype=QUInt(self.n))

        lam_r_split = bb.split(lam_r)
        for i in range(self.n):
            ctrls = [f_1, ctrl, lam_r_split[i]]
            ctrls, lam_split[i] = bb.add(
                MultiControlX(cvs=[1, 1, 1]), controls=ctrls, target=lam_split[i]
            )
            f_1 = ctrls[0]
            ctrl = ctrls[1]
            lam_r_split[i] = ctrls[2]
        lam_r = bb.join(lam_r_split, dtype=QUInt(self.n))

        lam = bb.join(lam_split, dtype=QUInt(self.n))

        lam, lam_r, f_1 = bb.add(Equals(self.n), x=lam, y=lam_r, target=f_1)
        # uncompute modinv modmult

        # Step 3:
        # ModMult (needs more garbage register)
        ctrl, z_1, y = bb.add(CModSub(QMontgomeryUInt(self.n), mod=self.mod), ctrl=ctrl, x=z_1, y=y)
        # uncompute modmult
        a_split = bb.split(a)
        z_1_split = bb.split(z_1)
        for i in range(self.n):
            a_split[i], z_1_split[i] = bb.add(CNOT(), ctrl=a_split[i], target=z_1_split[i])
        a = bb.join(a_split, QMontgomeryUInt(bitsize=self.n))
        z_1 = bb.join(z_1_split, QUInt(bitsize=self.n))

        z_1 = bb.add(ModDbl(QMontgomeryUInt(self.n), mod=self.mod), x=z_1)
        a, z_1 = bb.add(ModAdd(self.n, mod=self.mod), x=a, y=z_1)
        ctrl, z_1, x = bb.add(CModAdd(QMontgomeryUInt(self.n), mod=self.mod), ctrl=ctrl, x=z_1, y=x)
        # uncompute cnot moddbl modadd

        # Step 4:
        lam_split = bb.split(lam)
        z_4_split = bb.split(z_4)
        for i in range(self.n):
            lam_split[i], z_4_split[i] = bb.add(CNOT(), ctrl=lam_split[i], target=z_4_split[i])
        lam = bb.join(lam_split, QUInt(bitsize=self.n))
        z_4 = bb.join(z_4_split, QUInt(bitsize=self.n))

        # ModMult (needs more garbage register)

        z_3, x = bb.add(ModSub(QMontgomeryUInt(self.n), mod=self.mod), x=z_3, y=x)

        # uncompute cnot modmult
        # ModMult (needs more garbage register)
        z_3_split = bb.split(z_3)
        y_split = bb.split(y)
        for i in range(self.n):
            z_3_split[i], y_split[i] = bb.add(CNOT(), ctrl=z_3_split[i], target=y_split[i])
        z_3 = bb.join(z_3_split, QUInt(bitsize=self.n))
        y = bb.join(y_split, QMontgomeryUInt(bitsize=self.n))

        # uncompute modmult

        # Step 5:

        # uncompute modinv modmult multicontrolx uncompute modinv modmult
        ctrl, x = bb.add(CModNeg(QMontgomeryUInt(self.n), mod=self.mod), ctrl=ctrl, x=x)
        a, x = bb.add(ModAdd(self.n, mod=self.mod), x=a, y=x)
        ctrl, b, y = bb.add(CModSub(QMontgomeryUInt(self.n), mod=self.mod), ctrl=ctrl, x=b, y=y)

        # Step 6:
        f_ctrls = [f_2, f_3, f_4]
        f_ctrls, ctrl = bb.add(MultiControlX(cvs=[0] * 3), controls=f_ctrls, target=ctrl)
        f_2 = f_ctrls[0]
        f_3 = f_ctrls[1]
        f_4 = f_ctrls[2]

        a_split = bb.split(a)
        x_split = bb.split(x)
        for i in range(self.n):
            toff_ctrl = [f_4, a_split[i]]
            toff_ctrl, x_split[i] = bb.add(Toffoli(), ctrl=toff_ctrl, target=x_split[i])
            f_4 = toff_ctrl[0]
            a_split[i] = toff_ctrl[1]
        a = bb.join(a_split, QMontgomeryUInt(self.n))
        x = bb.join(x_split, QMontgomeryUInt(self.n))

        b_split = bb.split(b)
        y_split = bb.split(y)
        for i in range(self.n):
            toff_ctrl = [f_4, b_split[i]]
            toff_ctrl, y_split[i] = bb.add(Toffoli(), ctrl=toff_ctrl, target=y_split[i])
            f_4 = toff_ctrl[0]
            b_split[i] = toff_ctrl[1]
        b = bb.join(b_split, QMontgomeryUInt(self.n))
        y = bb.join(y_split, QMontgomeryUInt(self.n))

        a_arr = bb.split(a)
        b_arr = bb.split(b)
        ab = bb.join(
            np.concatenate((a_arr, b_arr), axis=None), dtype=QMontgomeryUInt(bitsize=2 * self.n)
        )
        x_arr = bb.split(x)
        y_arr = bb.split(y)
        xy = bb.join(
            np.concatenate((x_arr, y_arr), axis=None), dtype=QMontgomeryUInt(bitsize=2 * self.n)
        )
        ab, xy, f_4 = bb.add(Equals(2 * self.n), x=ab, y=xy, target=f_4)
        ab_split = bb.split(ab)
        a = bb.join(ab_split[: self.n], dtype=QMontgomeryUInt(bitsize=self.n))
        b = bb.join(ab_split[self.n :], dtype=QMontgomeryUInt(bitsize=self.n))
        xy_split = bb.split(xy)
        x = bb.join(xy_split[: self.n], dtype=QMontgomeryUInt(bitsize=self.n))
        y = bb.join(xy_split[self.n :], dtype=QMontgomeryUInt(bitsize=self.n))

        a_arr = bb.split(a)
        b_arr = bb.split(b)
        ab_arr = np.concatenate((a_arr, b_arr), axis=None)
        ab_arr, f_3 = bb.add(MultiControlX(cvs=[0] * 2 * self.n), controls=ab_arr, target=f_3)
        ab_arr = np.split(ab_arr, 2)
        a = bb.join(ab_arr[0], dtype=QMontgomeryUInt(bitsize=self.n))
        b = bb.join(ab_arr[1], dtype=QMontgomeryUInt(bitsize=self.n))

        ancilla = bb.add(ZeroState())
        toff_ctrl = [f_1, f_2]
        toff_ctrl, ancilla = bb.add(Toffoli(), ctrl=toff_ctrl, target=ancilla)
        ancilla, a, x = bb.add(
            CModSub(QMontgomeryUInt(self.n), mod=self.mod), ctrl=ancilla, x=a, y=x
        )
        toff_ctrl, ancilla = bb.add(Toffoli(), ctrl=toff_ctrl, target=ancilla)
        f_1 = toff_ctrl[0]
        f_2 = toff_ctrl[1]
        bb.add(Free(QBit()), reg=ancilla)

        ancilla = bb.add(ZeroState())
        toff_ctrl = [f_1, f_2]
        toff_ctrl, ancilla = bb.add(Toffoli(), ctrl=toff_ctrl, target=ancilla)
        ancilla, b, y = bb.add(
            CModAdd(QMontgomeryUInt(self.n), mod=self.mod), ctrl=ancilla, x=b, y=y
        )
        toff_ctrl, ancilla = bb.add(Toffoli(), ctrl=toff_ctrl, target=ancilla)
        f_1 = toff_ctrl[0]
        f_2 = toff_ctrl[1]
        bb.add(Free(QBit()), reg=ancilla)

        x_arr = bb.split(x)
        y_arr = bb.split(y)
        xy_arr = np.concatenate((x_arr, y_arr), axis=None)
        xy_arr, junk, out = bb.add(MultiAnd(cvs=[0] * 2 * self.n), ctrl=xy_arr)
        targets = bb.join(np.array([f_1, f_2]))
        out, targets = bb.add(MultiTargetCNOT(2), control=out, targets=targets)
        targets = bb.split(targets)
        f_1 = targets[0]
        f_2 = targets[1]
        xy_arr = bb.add(
            MultiAnd(cvs=[0] * 2 * self.n).adjoint(), ctrl=xy_arr, junk=junk, target=out
        )
        xy_arr = np.split(xy_arr, 2)
        x = bb.join(xy_arr[0], dtype=QMontgomeryUInt(bitsize=self.n))
        y = bb.join(xy_arr[1], dtype=QMontgomeryUInt(bitsize=self.n))

        # Step 7: Free all ancilla qubits in the zero state.
        bb.add(Free(QBit()), reg=f_1)
        bb.add(Free(QBit()), reg=f_2)
        bb.add(Free(QBit()), reg=f_3)
        bb.add(Free(QBit()), reg=f_4)
        bb.add(Free(QBit()), reg=ctrl)
        bb.add(Free(QUInt(self.n)), reg=z_1)
        bb.add(Free(QUInt(self.n)), reg=z_2)
        bb.add(Free(QUInt(self.n)), reg=z_3)
        bb.add(Free(QUInt(self.n)), reg=z_4)
        bb.add(Free(QUInt(self.n)), reg=lam)

        return {'a': a, 'b': b, 'x': x, 'y': y, 'lam_r': lam_r}

    def on_classical_vals(self, a, b, x, y, lam_r) -> Dict[str, Union['ClassicalValT', sympy.Expr]]:
        p1 = ECPoint(a, b, mod=self.mod, curve_a=0)
        p2 = ECPoint(x, y, mod=self.mod, curve_a=0)
        result: ECPoint = p1 + p2
        return {'a': a, 'b': b, 'x': result.x, 'y': result.y, 'lam_r': lam_r}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # litinksi
        return {
            (MultiCToffoli(n=self.n), 18),
            (ModAdd(bitsize=self.n, mod=self.mod), 3),
            (CModAdd(QUInt(self.n), mod=self.mod), 2),
            (ModSub(QUInt(self.n), mod=self.mod), 2),
            (CModSub(QUInt(self.n), mod=self.mod), 4),
            (ModNeg(QUInt(self.n), mod=self.mod), 2),
            (CModNeg(QUInt(self.n), mod=self.mod), 1),
            (ModDbl(QUInt(self.n), mod=self.mod), 2),
            (ModMul(n=self.n, mod=self.mod), 10),
            (ModInv(n=self.n, mod=self.mod), 4),
        }


@bloq_example
def _ec_add() -> ECAdd:
    n, p = sympy.symbols('n p')
    ec_add = ECAdd(n, mod=p)
    return ec_add


@bloq_example
def _ec_add_small() -> ECAdd:
    ec_add = ECAdd(5, mod=17)
    return ec_add


@bloq_example
def _ec_add_256() -> ECAdd:
    ec_add_256 = ECAdd(
        n=256, mod=0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
    )
    return ec_add_256


_EC_ADD_DOC = BloqDocSpec(bloq_cls=ECAdd, examples=[_ec_add, _ec_add_small, _ec_add_256])
