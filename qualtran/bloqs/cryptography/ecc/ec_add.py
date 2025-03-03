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
    CtrlSpec,
    DecomposeTypeError,
    QBit,
    QMontgomeryUInt,
    Register,
    Side,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.arithmetic import Equals, Xor
from qualtran.bloqs.basic_gates import CNOT, IntState, Toffoli, XGate, ZeroState
from qualtran.bloqs.bookkeeping import Free
from qualtran.bloqs.mcmt import MultiAnd, MultiControlX, MultiTargetCNOT
from qualtran.bloqs.mod_arithmetic import (
    CModAdd,
    CModNeg,
    CModSub,
    DirtyOutOfPlaceMontgomeryModMul,
    KaliskiModInverse,
    ModAdd,
    ModDbl,
    ModNeg,
    ModSub,
)
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.simulation.classical_sim import ClassicalValT
from qualtran.symbolics.types import HasLength, is_symbolic, SymbolicInt

from .ec_point import ECPoint


@frozen
class _ECAddStepOne(Bloq):
    r"""Performs step one of the ECAdd bloq.

    Args:
        n: The bitsize of the two registers storing the elliptic curve point
        mod: The modulus of the field in which we do the addition.

    Registers:
        f1: Flag to set if a = x.
        f2: Flag to set if b = -y.
        f3: Flag to set if (a, b) = (0, 0).
        f4: Flag to set if (x, y) = (0, 0).
        ctrl: Flag to set if neither the input points nor the output point are (0, 0).
        a: The x component of the first input elliptic curve point of bitsize `n` in montgomery form.
        b: The y component of the first input elliptic curve point of bitsize `n` in montgomery form.
        x: The x component of the second input elliptic curve point of bitsize `n` in montgomery form, which
           will contain the x component of the resultant curve point.
        y: The y component of the second input elliptic curve point of bitsize `n` in montgomery form, which
           will contain the y component of the resultant curve point.

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585)
        Fig 10.
    """

    n: 'SymbolicInt'
    mod: 'SymbolicInt'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('f1', QBit(), side=Side.RIGHT),
                Register('f2', QBit(), side=Side.RIGHT),
                Register('f3', QBit(), side=Side.RIGHT),
                Register('f4', QBit(), side=Side.RIGHT),
                Register('ctrl', QBit(), side=Side.RIGHT),
                Register('a', QMontgomeryUInt(self.n)),
                Register('b', QMontgomeryUInt(self.n)),
                Register('x', QMontgomeryUInt(self.n)),
                Register('y', QMontgomeryUInt(self.n)),
            ]
        )

    def on_classical_vals(
        self, a: 'ClassicalValT', b: 'ClassicalValT', x: 'ClassicalValT', y: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        f1 = int(a == x)
        f2 = int(b == (-y % self.mod))
        f3 = int(a == b == 0)
        f4 = int(x == y == 0)
        ctrl = int(f2 == f3 == f4 == 0)
        return {
            'f1': f1,
            'f2': f2,
            'f3': f3,
            'f4': f4,
            'ctrl': ctrl,
            'a': a,
            'b': b,
            'x': x,
            'y': y,
        }

    def build_composite_bloq(
        self, bb: 'BloqBuilder', a: Soquet, b: Soquet, x: Soquet, y: Soquet
    ) -> Dict[str, 'SoquetT']:
        if is_symbolic(self.n):
            raise DecomposeTypeError(f"Cannot decompose {self} with symbolic `n`.")

        # Initialize control flags to 0.
        f1 = bb.add(ZeroState())
        f2 = bb.add(ZeroState())
        f3 = bb.add(ZeroState())
        f4 = bb.add(ZeroState())
        ctrl = bb.add(ZeroState())

        # Set flag 1 if a = x.
        a, x, f1 = bb.add(Equals(QMontgomeryUInt(self.n)), x=a, y=x, target=f1)

        # Set flag 2 if b = -y.
        y = bb.add(ModNeg(QMontgomeryUInt(self.n), mod=self.mod), x=y)
        b, y, f2 = bb.add(Equals(QMontgomeryUInt(self.n)), x=b, y=y, target=f2)
        y = bb.add(ModNeg(QMontgomeryUInt(self.n), mod=self.mod), x=y)

        # Set flag 3 if (a, b) == (0, 0).
        ab_arr = np.concatenate([bb.split(a), bb.split(b)])
        ab_arr, f3 = bb.add(MultiControlX(cvs=[0] * 2 * self.n), controls=ab_arr, target=f3)
        ab_arr = np.split(ab_arr, 2)
        a = bb.join(ab_arr[0], dtype=QMontgomeryUInt(self.n))
        b = bb.join(ab_arr[1], dtype=QMontgomeryUInt(self.n))

        # Set flag 4 if (x, y) == (0, 0).
        xy_arr = np.concatenate([bb.split(x), bb.split(y)])
        xy_arr, f4 = bb.add(MultiControlX(cvs=[0] * 2 * self.n), controls=xy_arr, target=f4)
        xy_arr = np.split(xy_arr, 2)
        x = bb.join(xy_arr[0], dtype=QMontgomeryUInt(self.n))
        y = bb.join(xy_arr[1], dtype=QMontgomeryUInt(self.n))

        # Set ctrl flag if f2, f3, f4 are set.
        f_ctrls = [f2, f3, f4]
        f_ctrls, ctrl = bb.add(MultiControlX(cvs=[0] * 3), controls=f_ctrls, target=ctrl)
        f2 = f_ctrls[0]
        f3 = f_ctrls[1]
        f4 = f_ctrls[2]

        # Return the output registers.
        return {
            'f1': f1,
            'f2': f2,
            'f3': f3,
            'f4': f4,
            'ctrl': ctrl,
            'a': a,
            'b': b,
            'x': x,
            'y': y,
        }

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        cvs: Union[list[int], HasLength]
        if isinstance(self.n, int):
            cvs = [0] * 2 * self.n
        else:
            cvs = HasLength(2 * self.n)
        return {
            Equals(QMontgomeryUInt(self.n)): 2,
            ModNeg(QMontgomeryUInt(self.n), mod=self.mod): 2,
            MultiControlX(cvs=cvs): 2,
            MultiControlX(cvs=[0] * 3): 1,
        }


@frozen
class _ECAddStepTwo(Bloq):
    r"""Performs step two of the ECAdd bloq.

    Includes a bugfix for the scenario where the calculated λ ( = (y - b) / (x - a)) is equivalent
    to λ_r ( = 3 * a ^ 2 + c_1 / (2 * b)) and f_1 is wrongfully cleared. We accomplish this by
    introducing a new ancilla qubit set by an equals operation on the computed λ and the classical,
    pre-computed λ_r. We then control the equals bloq on this ancilla qubit which will only clear
    the f_1 flag in the correct situation. Finally, we clear and free the ancilla afterwards.

    Args:
        n: The bitsize of the two registers storing the elliptic curve point
        mod: The modulus of the field in which we do the addition.
        window_size: The number of bits in the ModMult window.

    Registers:
        f1: Flag set if a = x.
        ctrl: Flag set if neither the input points nor the output point are (0, 0).
        a: The x component of the first input elliptic curve point of bitsize `n` in montgomery form.
        b: The y component of the first input elliptic curve point of bitsize `n` in montgomery form.
        x: The x component of the second input elliptic curve point of bitsize `n` in montgomery form, which
           will contain the x component of the resultant curve point.
        y: The y component of the second input elliptic curve point of bitsize `n` in montgomery form, which
           will contain the y component of the resultant curve point.
        lam: The lambda slope used in the addition operation.
        lam_r: The precomputed lambda slope used in the addition operation if (a, b) = (x, y) in montgomery form.

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585)
        Fig 10.
    """

    n: 'SymbolicInt'
    mod: 'SymbolicInt'
    window_size: 'SymbolicInt' = 1

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('f1', QBit()),
                Register('ctrl', QBit()),
                Register('a', QMontgomeryUInt(self.n)),
                Register('b', QMontgomeryUInt(self.n)),
                Register('x', QMontgomeryUInt(self.n)),
                Register('y', QMontgomeryUInt(self.n)),
                Register('lam', QMontgomeryUInt(self.n), side=Side.RIGHT),
                Register('lam_r', QMontgomeryUInt(self.n)),
            ]
        )

    def on_classical_vals(
        self,
        f1: 'ClassicalValT',
        ctrl: 'ClassicalValT',
        a: 'ClassicalValT',
        b: 'ClassicalValT',
        x: 'ClassicalValT',
        y: 'ClassicalValT',
        lam_r: 'ClassicalValT',
    ) -> Dict[str, 'ClassicalValT']:
        x = (x - a) % self.mod
        if ctrl == 1:
            y = (y - b) % self.mod
            if f1 == 1:
                lam = lam_r
                f1 = 0
            else:
                lam = QMontgomeryUInt(self.n, self.mod).montgomery_product(
                    int(y), QMontgomeryUInt(self.n, self.mod).montgomery_inverse(int(x))
                )
        else:
            lam = 0
        return {'f1': f1, 'ctrl': ctrl, 'a': a, 'b': b, 'x': x, 'y': y, 'lam': lam, 'lam_r': lam_r}

    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        f1: Soquet,
        ctrl: Soquet,
        a: Soquet,
        b: Soquet,
        x: Soquet,
        y: Soquet,
        lam_r: Soquet,
    ) -> Dict[str, 'SoquetT']:
        if is_symbolic(self.n):
            raise DecomposeTypeError(f"Cannot decompose {self} with symbolic `n`.")

        # Initalize lambda to 0.
        lam = bb.add(IntState(bitsize=self.n, val=0))

        # Perform modular subtraction so that x = (x - a) % p.
        a, x = bb.add(ModSub(QMontgomeryUInt(self.n), mod=self.mod), x=a, y=x)

        # Perform controlled modular subtraction so that y = (y - b) % p iff ctrl = 1.
        ctrl, b, y = bb.add(CModSub(QMontgomeryUInt(self.n), mod=self.mod), ctrl=ctrl, x=b, y=y)

        # Perform modular inversion s.t. x = (x - a)^-1 % p.
        x, junk = bb.add(KaliskiModInverse(bitsize=self.n, mod=self.mod), x=x)

        # Perform modular multiplication z4 = (y / x) % p.
        x, y, z4, z3, reduced = bb.add(
            DirtyOutOfPlaceMontgomeryModMul(
                bitsize=self.n, window_size=self.window_size, mod=self.mod
            ),
            x=x,
            y=y,
        )

        # Allocate an ancilla qubit that acts as a flag for the rare condition that the
        # pre-computed lambda_r is equal to the calculated lambda. This ancilla is used to properly
        # clear the f1 qubit when lambda is set to lambda_r.
        ancilla = bb.allocate()
        z4, lam_r, ancilla = bb.add(Equals(QMontgomeryUInt(self.n)), x=z4, y=lam_r, target=ancilla)

        # If ctrl = 1 and x != a: lam = (y - b) / (x - a) % p.
        z4_split = bb.split(z4)
        lam_split = bb.split(lam)
        for i in range(int(self.n)):
            ctrls = [f1, ctrl, z4_split[i]]
            ctrls, lam_split[i] = bb.add(
                MultiControlX(cvs=[0, 1, 1]), controls=ctrls, target=lam_split[i]
            )
            f1 = ctrls[0]
            ctrl = ctrls[1]
            z4_split[i] = ctrls[2]
        z4 = bb.join(z4_split, dtype=QMontgomeryUInt(self.n))

        # If ctrl = 1 and x = a: lam = lam_r.
        lam_r_split = bb.split(lam_r)
        for i in range(int(self.n)):
            ctrls = [f1, ctrl, lam_r_split[i]]
            ctrls, lam_split[i] = bb.add(
                MultiControlX(cvs=[1, 1, 1]), controls=ctrls, target=lam_split[i]
            )
            f1 = ctrls[0]
            ctrl = ctrls[1]
            lam_r_split[i] = ctrls[2]
        lam_r = bb.join(lam_r_split, dtype=QMontgomeryUInt(self.n))
        lam = bb.join(lam_split, dtype=QMontgomeryUInt(self.n))

        # If lam = lam_r: return f1 = 0. (If not we will flip f1 to 0 at the end iff x_r = y_r = 0).
        # Only flip when lam is set to lam_r.
        ancilla, lam, lam_r, f1 = bb.add(
            Equals(QMontgomeryUInt(self.n)).controlled(ctrl_spec=CtrlSpec(cvs=0)),
            ctrl=ancilla,
            x=lam,
            y=lam_r,
            target=f1,
        )

        # Clear the ancilla bit and free it.
        z4, lam_r, ancilla = bb.add(Equals(QMontgomeryUInt(self.n)), x=z4, y=lam_r, target=ancilla)
        bb.free(ancilla)

        # Uncompute the modular multiplication then the modular inversion.
        x, y = bb.add(
            DirtyOutOfPlaceMontgomeryModMul(
                bitsize=self.n, window_size=self.window_size, mod=self.mod
            ).adjoint(),
            x=x,
            y=y,
            target=z4,
            qrom_indices=z3,
            reduced=reduced,
        )
        x = bb.add(KaliskiModInverse(bitsize=self.n, mod=self.mod).adjoint(), x=x, junk=junk)

        # Return the output registers.
        return {'f1': f1, 'ctrl': ctrl, 'a': a, 'b': b, 'x': x, 'y': y, 'lam': lam, 'lam_r': lam_r}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        return {
            Equals(QMontgomeryUInt(self.n)): 2,
            Equals(QMontgomeryUInt(self.n)).controlled(ctrl_spec=CtrlSpec(cvs=0)): 1,
            ModSub(QMontgomeryUInt(self.n), mod=self.mod): 1,
            CModSub(QMontgomeryUInt(self.n), mod=self.mod): 1,
            KaliskiModInverse(bitsize=self.n, mod=self.mod): 1,
            DirtyOutOfPlaceMontgomeryModMul(
                bitsize=self.n, window_size=self.window_size, mod=self.mod
            ): 1,
            MultiControlX(cvs=[0, 1, 1]): self.n,
            MultiControlX(cvs=[1, 1, 1]): self.n,
            DirtyOutOfPlaceMontgomeryModMul(
                bitsize=self.n, window_size=self.window_size, mod=self.mod
            ).adjoint(): 1,
            KaliskiModInverse(bitsize=self.n, mod=self.mod).adjoint(): 1,
        }


@frozen
class _ECAddStepThree(Bloq):
    r"""Performs step three of the ECAdd bloq.

    Args:
        n: The bitsize of the two registers storing the elliptic curve point
        mod: The modulus of the field in which we do the addition.
        window_size: The number of bits in the ModMult window.

    Registers:
        ctrl: Flag set if neither the input points nor the output point are (0, 0).
        a: The x component of the first input elliptic curve point of bitsize `n` in montgomery form.
        b: The y component of the first input elliptic curve point of bitsize `n` in montgomery form.
        x: The x component of the second input elliptic curve point of bitsize `n` in montgomery form, which
           will contain the x component of the resultant curve point.
        y: The y component of the second input elliptic curve point of bitsize `n` in montgomery form, which
           will contain the y component of the resultant curve point.
        lam: The lambda slope used in the addition operation.

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585)
        Fig 10.
    """

    n: 'SymbolicInt'
    mod: 'SymbolicInt'
    window_size: 'SymbolicInt' = 1

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('ctrl', QBit()),
                Register('a', QMontgomeryUInt(self.n)),
                Register('b', QMontgomeryUInt(self.n)),
                Register('x', QMontgomeryUInt(self.n)),
                Register('y', QMontgomeryUInt(self.n)),
                Register('lam', QMontgomeryUInt(self.n)),
            ]
        )

    def on_classical_vals(
        self,
        ctrl: 'ClassicalValT',
        a: 'ClassicalValT',
        b: 'ClassicalValT',
        x: 'ClassicalValT',
        y: 'ClassicalValT',
        lam: 'ClassicalValT',
    ) -> Dict[str, 'ClassicalValT']:
        if ctrl == 1:
            x = (x + 3 * a) % self.mod
            y = 0
        return {'ctrl': ctrl, 'a': a, 'b': b, 'x': x, 'y': y, 'lam': lam}

    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        ctrl: Soquet,
        a: Soquet,
        b: Soquet,
        x: Soquet,
        y: Soquet,
        lam: Soquet,
    ) -> Dict[str, 'SoquetT']:
        if is_symbolic(self.n):
            raise DecomposeTypeError(f"Cannot decompose {self} with symbolic `n`.")

        # Store (x - a) * lam % p in z1 (= (y - b) % p).
        x, lam, z1, z2, reduced = bb.add(
            DirtyOutOfPlaceMontgomeryModMul(
                bitsize=self.n, window_size=self.window_size, mod=self.mod
            ),
            x=x,
            y=lam,
        )

        # If ctrl: subtract z1 from y (= 0).
        ctrl, z1, y = bb.add(CModSub(QMontgomeryUInt(self.n), mod=self.mod), ctrl=ctrl, x=z1, y=y)

        # Uncompute original multiplication.
        x, lam = bb.add(
            DirtyOutOfPlaceMontgomeryModMul(
                bitsize=self.n, window_size=self.window_size, mod=self.mod
            ).adjoint(),
            x=x,
            y=lam,
            target=z1,
            qrom_indices=z2,
            reduced=reduced,
        )

        # z1 = a.
        z1 = bb.add(IntState(bitsize=self.n, val=0))
        a_split = bb.split(a)
        z1_split = bb.split(z1)
        for i in range(int(self.n)):
            a_split[i], z1_split[i] = bb.add(CNOT(), ctrl=a_split[i], target=z1_split[i])
        a = bb.join(a_split, QMontgomeryUInt(self.n))
        z1 = bb.join(z1_split, QMontgomeryUInt(self.n))

        # z1 = (3 * a) % p.
        z1 = bb.add(ModDbl(QMontgomeryUInt(self.n), mod=self.mod), x=z1)
        a, z1 = bb.add(ModAdd(self.n, mod=self.mod), x=a, y=z1)

        # If ctrl: x = (x + 2 * a) % p.
        ctrl, z1, x = bb.add(CModAdd(QMontgomeryUInt(self.n), mod=self.mod), ctrl=ctrl, x=z1, y=x)

        # Uncompute z1.
        a, z1 = bb.add(ModAdd(self.n, mod=self.mod).adjoint(), x=a, y=z1)
        z1 = bb.add(ModDbl(QMontgomeryUInt(self.n), mod=self.mod).adjoint(), x=z1)
        a_split = bb.split(a)
        z1_split = bb.split(z1)
        for i in range(int(self.n)):
            a_split[i], z1_split[i] = bb.add(CNOT(), ctrl=a_split[i], target=z1_split[i])
        a = bb.join(a_split, QMontgomeryUInt(self.n))
        z1 = bb.join(z1_split, QMontgomeryUInt(self.n))
        bb.add(Free(QMontgomeryUInt(self.n)), reg=z1)

        # Return the output registers.
        return {'ctrl': ctrl, 'a': a, 'b': b, 'x': x, 'y': y, 'lam': lam}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        return {
            CModSub(QMontgomeryUInt(self.n), mod=self.mod): 1,
            DirtyOutOfPlaceMontgomeryModMul(
                bitsize=self.n, window_size=self.window_size, mod=self.mod
            ): 1,
            DirtyOutOfPlaceMontgomeryModMul(
                bitsize=self.n, window_size=self.window_size, mod=self.mod
            ).adjoint(): 1,
            CNOT(): 2 * self.n,
            ModDbl(QMontgomeryUInt(self.n), mod=self.mod): 1,
            ModAdd(self.n, mod=self.mod): 1,
            CModAdd(QMontgomeryUInt(self.n), mod=self.mod): 1,
            ModAdd(self.n, mod=self.mod).adjoint(): 1,
            ModDbl(QMontgomeryUInt(self.n), mod=self.mod).adjoint(): 1,
        }


@frozen
class _ECAddStepFour(Bloq):
    r"""Performs step four of the ECAdd bloq.

    Args:
        n: The bitsize of the two registers storing the elliptic curve point
        mod: The modulus of the field in which we do the addition.
        window_size: The number of bits in the ModMult window.

    Registers:
        x: The x component of the second input elliptic curve point of bitsize `n` in montgomery form, which
           will contain the x component of the resultant curve point.
        y: The y component of the second input elliptic curve point of bitsize `n` in montgomery form, which
           will contain the y component of the resultant curve point.
        lam: The lambda slope used in the addition operation.

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585)
        Fig 10.
    """

    n: 'SymbolicInt'
    mod: 'SymbolicInt'
    window_size: 'SymbolicInt' = 1

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('x', QMontgomeryUInt(self.n)),
                Register('y', QMontgomeryUInt(self.n)),
                Register('lam', QMontgomeryUInt(self.n)),
            ]
        )

    def on_classical_vals(
        self, x: 'ClassicalValT', y: 'ClassicalValT', lam: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        x = (
            x - QMontgomeryUInt(self.n, self.mod).montgomery_product(int(lam), int(lam))
        ) % self.mod
        if lam > 0:
            y = QMontgomeryUInt(self.n, self.mod).montgomery_product(int(x), int(lam))
        return {'x': x, 'y': y, 'lam': lam}

    def build_composite_bloq(
        self, bb: 'BloqBuilder', x: Soquet, y: Soquet, lam: Soquet
    ) -> Dict[str, 'SoquetT']:
        if is_symbolic(self.n):
            raise DecomposeTypeError(f"Cannot decompose {self} with symbolic `n`.")

        # Initialize z4 = lam.
        z4 = bb.add(IntState(bitsize=self.n, val=0))
        lam_split = bb.split(lam)
        z4_split = bb.split(z4)
        for i in range(int(self.n)):
            lam_split[i], z4_split[i] = bb.add(CNOT(), ctrl=lam_split[i], target=z4_split[i])
        lam = bb.join(lam_split, QMontgomeryUInt(self.n))
        z4 = bb.join(z4_split, QMontgomeryUInt(self.n))

        # z3 = lam * lam % p.
        z4, lam, z3, z2, reduced = bb.add(
            DirtyOutOfPlaceMontgomeryModMul(
                bitsize=self.n, window_size=self.window_size, mod=self.mod
            ),
            x=z4,
            y=lam,
        )

        # x = a - x_r % p.
        z3, x = bb.add(ModSub(QMontgomeryUInt(self.n), mod=self.mod), x=z3, y=x)

        # Uncompute the multiplication and initialization of z4.
        z4, lam = bb.add(
            DirtyOutOfPlaceMontgomeryModMul(
                bitsize=self.n, window_size=self.window_size, mod=self.mod
            ).adjoint(),
            x=z4,
            y=lam,
            target=z3,
            qrom_indices=z2,
            reduced=reduced,
        )
        lam_split = bb.split(lam)
        z4_split = bb.split(z4)
        for i in range(int(self.n)):
            lam_split[i], z4_split[i] = bb.add(CNOT(), ctrl=lam_split[i], target=z4_split[i])
        lam = bb.join(lam_split, QMontgomeryUInt(self.n))
        z4 = bb.join(z4_split, QMontgomeryUInt(self.n))
        bb.add(Free(QMontgomeryUInt(self.n)), reg=z4)

        # z3 = lam * x % p.
        x, lam, z3, z4, reduced = bb.add(
            DirtyOutOfPlaceMontgomeryModMul(
                bitsize=self.n, window_size=self.window_size, mod=self.mod
            ),
            x=x,
            y=lam,
        )

        # y = y_r + b % p.
        z3_split = bb.split(z3)
        y_split = bb.split(y)
        for i in range(int(self.n)):
            z3_split[i], y_split[i] = bb.add(CNOT(), ctrl=z3_split[i], target=y_split[i])
        z3 = bb.join(z3_split, QMontgomeryUInt(self.n))
        y = bb.join(y_split, QMontgomeryUInt(self.n))

        # Uncompute multiplication.
        x, lam = bb.add(
            DirtyOutOfPlaceMontgomeryModMul(
                bitsize=self.n, window_size=self.window_size, mod=self.mod
            ).adjoint(),
            x=x,
            y=lam,
            target=z3,
            qrom_indices=z4,
            reduced=reduced,
        )

        # Return the output registers.
        return {'x': x, 'y': y, 'lam': lam}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        return {
            ModSub(QMontgomeryUInt(self.n), mod=self.mod): 1,
            DirtyOutOfPlaceMontgomeryModMul(
                bitsize=self.n, window_size=self.window_size, mod=self.mod
            ): 2,
            DirtyOutOfPlaceMontgomeryModMul(
                bitsize=self.n, window_size=self.window_size, mod=self.mod
            ).adjoint(): 2,
            CNOT(): 3 * self.n,
        }


@frozen
class _ECAddStepFive(Bloq):
    r"""Performs step five of the ECAdd bloq.

    Includes a bugfix for the scenario where (a, b) = (x, y) and a - x_r = 0. In this situation,
    f_1 is set and f_2 - f_4 is cleared (which means that the ctrl qubit is set). Because a - x_r
    is 0, the computed λ is undefined (and with this construction the computed λ will be set to 0),
    however the λ is non-zero and should be cleared with λ_r. We accomplish this with a controled
    Xor bloq controlled on the ctrl qubit and the condition that the x register (a - x_r) = 0. In
    this ase we clear the λ register with λ_r.

    Args:
        n: The bitsize of the two registers storing the elliptic curve point
        mod: The modulus of the field in which we do the addition.
        window_size: The number of bits in the ModMult window.

    Registers:
        ctrl: Flag set if neither the input points nor the output point are (0, 0).
        a: The x component of the first input elliptic curve point of bitsize `n` in montgomery form.
        b: The y component of the first input elliptic curve point of bitsize `n` in montgomery form.
        x: The x component of the second input elliptic curve point of bitsize `n` in montgomery form, which
           will contain the x component of the resultant curve point.
        y: The y component of the second input elliptic curve point of bitsize `n` in montgomery form, which
           will contain the y component of the resultant curve point.
        lam_r: The precomputed lambda slope used in the addition operation if (a, b) = (x, y) in montgomery form.
        lam: The lambda slope used in the addition operation.

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585)
        Fig 10.
    """

    n: 'SymbolicInt'
    mod: 'SymbolicInt'
    window_size: 'SymbolicInt' = 1

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('ctrl', QBit()),
                Register('a', QMontgomeryUInt(self.n)),
                Register('b', QMontgomeryUInt(self.n)),
                Register('x', QMontgomeryUInt(self.n)),
                Register('y', QMontgomeryUInt(self.n)),
                Register('lam_r', QMontgomeryUInt(self.n)),
                Register('lam', QMontgomeryUInt(self.n), side=Side.LEFT),
            ]
        )

    def on_classical_vals(
        self,
        ctrl: 'ClassicalValT',
        a: 'ClassicalValT',
        b: 'ClassicalValT',
        x: 'ClassicalValT',
        y: 'ClassicalValT',
        lam_r: 'ClassicalValT',
        lam: 'ClassicalValT',
    ) -> Dict[str, 'ClassicalValT']:
        if ctrl == 1:
            x = (a - x) % self.mod
            y = (y - b) % self.mod
        else:
            x = (x + a) % self.mod
        return {'ctrl': ctrl, 'a': a, 'b': b, 'x': x, 'y': y, 'lam_r': lam_r}

    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        ctrl: Soquet,
        a: Soquet,
        b: Soquet,
        x: Soquet,
        y: Soquet,
        lam_r: Soquet,
        lam: Soquet,
    ) -> Dict[str, 'SoquetT']:
        if is_symbolic(self.n):
            raise DecomposeTypeError(f"Cannot decompose {self} with symbolic `n`.")

        # x = x ^ -1 % p.
        x, junk = bb.add(KaliskiModInverse(bitsize=self.n, mod=self.mod), x=x)

        # z4 = x * y % p.
        x, y, z4, z3, reduced = bb.add(
            DirtyOutOfPlaceMontgomeryModMul(
                bitsize=self.n, window_size=self.window_size, mod=self.mod
            ),
            x=x,
            y=y,
        )

        # If ctrl: lam = 0.
        z4_split = bb.split(z4)
        lam_split = bb.split(lam)
        for i in range(int(self.n)):
            ctrls = [ctrl, z4_split[i]]
            ctrls, lam_split[i] = bb.add(
                MultiControlX(cvs=[1, 1]), controls=ctrls, target=lam_split[i]
            )
            ctrl = ctrls[0]
            z4_split[i] = ctrls[1]
        z4 = bb.join(z4_split, dtype=QMontgomeryUInt(self.n))
        lam = bb.join(lam_split, dtype=QMontgomeryUInt(self.n))

        # If the denominator of lambda is 0, lam = lam_r so we clear lam with lam_r.
        clear_lam = (
            Xor(QMontgomeryUInt(self.n))
            .controlled(CtrlSpec(qdtypes=QMontgomeryUInt(self.n), cvs=0))
            .controlled()
        )
        ctrl, x, lam_r, lam = bb.add(clear_lam, ctrl1=ctrl, ctrl2=x, x=lam_r, y=lam)
        bb.add(Free(QMontgomeryUInt(self.n)), reg=lam)

        # Uncompute multiplication and inverse.
        x, y = bb.add(
            DirtyOutOfPlaceMontgomeryModMul(
                bitsize=self.n, window_size=self.window_size, mod=self.mod
            ).adjoint(),
            x=x,
            y=y,
            target=z4,
            qrom_indices=z3,
            reduced=reduced,
        )
        x = bb.add(KaliskiModInverse(bitsize=self.n, mod=self.mod).adjoint(), x=x, junk=junk)

        # If ctrl: x = x_r - a % p.
        ctrl, x = bb.add(CModNeg(QMontgomeryUInt(self.n), mod=self.mod), ctrl=ctrl, x=x)

        # Add a to x (x = x_r).
        a, x = bb.add(ModAdd(self.n, mod=self.mod), x=a, y=x)

        # If ctrl: subtract b from y (y = y_r).
        ctrl, b, y = bb.add(CModSub(QMontgomeryUInt(self.n), mod=self.mod), ctrl=ctrl, x=b, y=y)

        # Return the output registers.
        return {'ctrl': ctrl, 'a': a, 'b': b, 'x': x, 'y': y, 'lam_r': lam_r}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        clear_lam = (
            Xor(QMontgomeryUInt(self.n))
            .controlled(CtrlSpec(qdtypes=QMontgomeryUInt(self.n), cvs=0))
            .controlled()
        )
        return {
            CModSub(QMontgomeryUInt(self.n), mod=self.mod): 1,
            KaliskiModInverse(bitsize=self.n, mod=self.mod): 1,
            DirtyOutOfPlaceMontgomeryModMul(
                bitsize=self.n, window_size=self.window_size, mod=self.mod
            ): 1,
            DirtyOutOfPlaceMontgomeryModMul(
                bitsize=self.n, window_size=self.window_size, mod=self.mod
            ).adjoint(): 1,
            KaliskiModInverse(bitsize=self.n, mod=self.mod).adjoint(): 1,
            ModAdd(self.n, mod=self.mod): 1,
            MultiControlX(cvs=[1, 1]): self.n,
            clear_lam: 1,
            CModNeg(QMontgomeryUInt(self.n), mod=self.mod): 1,
        }


@frozen
class _ECAddStepSix(Bloq):
    r"""Performs step six of the ECAdd bloq.

    Include bugfixes for the following scenarios:
        1. f_2 is improperly cleared when ((x, y) = (0, 0) AND b = 0) OR ((a, b) = (0, 0) AND
            y = 0).
        2. f_4 is improperly cleared when P_1 = P_2 AND f_4 is set.

    The bugs are fixed respectively by:
        1. Clearing f_2 when x = y = b = 0 OR a = b = y = 0 using an XGate controlled on those
            registers.
        2. Moving the CModSub and CModAdd bloqs before the Equals bloq.

    Args:
        n: The bitsize of the two registers storing the elliptic curve point
        mod: The modulus of the field in which we do the addition.

    Registers:
        f1: Flag to set if a = x.
        f2: Flag to set if b = -y.
        f3: Flag to set if (a, b) = (0, 0).
        f4: Flag to set if (x, y) = (0, 0).
        ctrl: Flag to set if neither the input points nor the output point are (0, 0).
        a: The x component of the first input elliptic curve point of bitsize `n` in montgomery form.
        b: The y component of the first input elliptic curve point of bitsize `n` in montgomery form.
        x: The x component of the second input elliptic curve point of bitsize `n` in montgomery form, which
           will contain the x component of the resultant curve point.
        y: The y component of the second input elliptic curve point of bitsize `n` in montgomery form, which
           will contain the y component of the resultant curve point.

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585)
        Fig 10.
    """

    n: 'SymbolicInt'
    mod: 'SymbolicInt'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('f1', QBit(), side=Side.LEFT),
                Register('f2', QBit(), side=Side.LEFT),
                Register('f3', QBit(), side=Side.LEFT),
                Register('f4', QBit(), side=Side.LEFT),
                Register('ctrl', QBit(), side=Side.LEFT),
                Register('a', QMontgomeryUInt(self.n)),
                Register('b', QMontgomeryUInt(self.n)),
                Register('x', QMontgomeryUInt(self.n)),
                Register('y', QMontgomeryUInt(self.n)),
            ]
        )

    def on_classical_vals(
        self,
        f1: 'ClassicalValT',
        f2: 'ClassicalValT',
        f3: 'ClassicalValT',
        f4: 'ClassicalValT',
        ctrl: 'ClassicalValT',
        a: 'ClassicalValT',
        b: 'ClassicalValT',
        x: 'ClassicalValT',
        y: 'ClassicalValT',
    ) -> Dict[str, 'ClassicalValT']:
        if f4 == 1:
            x = a
            y = b
        if f1 and f2:
            x = 0
            y = 0
        return {'a': a, 'b': b, 'x': x, 'y': y}

    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        f1: Soquet,
        f2: Soquet,
        f3: Soquet,
        f4: Soquet,
        ctrl: Soquet,
        a: Soquet,
        b: Soquet,
        x: Soquet,
        y: Soquet,
    ) -> Dict[str, 'SoquetT']:
        if is_symbolic(self.n):
            raise DecomposeTypeError(f"Cannot decompose {self} with symbolic `n`.")

        # Unset control if f2, f3, and f4 flags are set.
        f_ctrls = [f2, f3, f4]
        f_ctrls, ctrl = bb.add(MultiControlX(cvs=[0] * 3), controls=f_ctrls, target=ctrl)
        f2 = f_ctrls[0]
        f3 = f_ctrls[1]
        f4 = f_ctrls[2]

        # Unset f2 if ((a, b) = (0, 0) AND y = 0) OR ((x, y) = (0, 0) AND b = 0).
        mcx = XGate().controlled(CtrlSpec(qdtypes=QMontgomeryUInt(self.n), cvs=[0, 0, 0]))
        [a, b, y], f2 = bb.add(mcx, ctrl=[a, b, y], q=f2)
        [x, y, b], f2 = bb.add(mcx, ctrl=[x, y, b], q=f2)

        # Set (x, y) to (a, b) if f4 is set.
        a_split = bb.split(a)
        x_split = bb.split(x)
        for i in range(int(self.n)):
            toff_ctrl = [f4, a_split[i]]
            toff_ctrl, x_split[i] = bb.add(Toffoli(), ctrl=toff_ctrl, target=x_split[i])
            f4 = toff_ctrl[0]
            a_split[i] = toff_ctrl[1]
        a = bb.join(a_split, QMontgomeryUInt(self.n))
        x = bb.join(x_split, QMontgomeryUInt(self.n))
        b_split = bb.split(b)
        y_split = bb.split(y)
        for i in range(int(self.n)):
            toff_ctrl = [f4, b_split[i]]
            toff_ctrl, y_split[i] = bb.add(Toffoli(), ctrl=toff_ctrl, target=y_split[i])
            f4 = toff_ctrl[0]
            b_split[i] = toff_ctrl[1]
        b = bb.join(b_split, QMontgomeryUInt(self.n))
        y = bb.join(y_split, QMontgomeryUInt(self.n))

        # If f1 and f2 are set, subtract a from x and add b to y.
        ancilla = bb.add(ZeroState())
        toff_ctrl = [f1, f2]
        toff_ctrl, ancilla = bb.add(Toffoli(), ctrl=toff_ctrl, target=ancilla)
        ancilla, a, x = bb.add(
            CModSub(QMontgomeryUInt(self.n), mod=self.mod), ctrl=ancilla, x=a, y=x
        )
        toff_ctrl, ancilla = bb.add(Toffoli(), ctrl=toff_ctrl, target=ancilla)
        f1 = toff_ctrl[0]
        f2 = toff_ctrl[1]
        bb.add(Free(QBit()), reg=ancilla)
        ancilla = bb.add(ZeroState())
        toff_ctrl = [f1, f2]
        toff_ctrl, ancilla = bb.add(Toffoli(), ctrl=toff_ctrl, target=ancilla)
        ancilla, b, y = bb.add(
            CModAdd(QMontgomeryUInt(self.n), mod=self.mod), ctrl=ancilla, x=b, y=y
        )
        toff_ctrl, ancilla = bb.add(Toffoli(), ctrl=toff_ctrl, target=ancilla)
        f1 = toff_ctrl[0]
        f2 = toff_ctrl[1]
        bb.add(Free(QBit()), reg=ancilla)

        # Unset f4 if (x, y) = (a, b).
        ab = bb.join(np.concatenate([bb.split(a), bb.split(b)]), dtype=QMontgomeryUInt(2 * self.n))
        xy = bb.join(np.concatenate([bb.split(x), bb.split(y)]), dtype=QMontgomeryUInt(2 * self.n))
        ab, xy, f4 = bb.add(Equals(QMontgomeryUInt(2 * self.n)), x=ab, y=xy, target=f4)
        ab_split = bb.split(ab)
        a = bb.join(ab_split[: int(self.n)], dtype=QMontgomeryUInt(self.n))
        b = bb.join(ab_split[int(self.n) :], dtype=QMontgomeryUInt(self.n))
        xy_split = bb.split(xy)
        x = bb.join(xy_split[: int(self.n)], dtype=QMontgomeryUInt(self.n))
        y = bb.join(xy_split[int(self.n) :], dtype=QMontgomeryUInt(self.n))

        # Unset f3 if (a, b) = (0, 0).
        ab_arr = np.concatenate([bb.split(a), bb.split(b)])
        ab_arr, f3 = bb.add(MultiControlX(cvs=[0] * 2 * self.n), controls=ab_arr, target=f3)
        ab_arr = np.split(ab_arr, 2)
        a = bb.join(ab_arr[0], dtype=QMontgomeryUInt(self.n))
        b = bb.join(ab_arr[1], dtype=QMontgomeryUInt(self.n))

        # Unset f1 and f2 if (x, y) = (0, 0).
        xy_arr = np.concatenate([bb.split(x), bb.split(y)])
        xy_arr, junk, out = bb.add(MultiAnd(cvs=[0] * 2 * self.n), ctrl=xy_arr)
        targets = bb.join(np.array([f1, f2]))
        out, targets = bb.add(MultiTargetCNOT(2), control=out, targets=targets)
        targets = bb.split(targets)
        f1 = targets[0]
        f2 = targets[1]
        xy_arr = bb.add(
            MultiAnd(cvs=[0] * 2 * self.n).adjoint(), ctrl=xy_arr, junk=junk, target=out
        )
        xy_arr = np.split(xy_arr, 2)
        x = bb.join(xy_arr[0], dtype=QMontgomeryUInt(self.n))
        y = bb.join(xy_arr[1], dtype=QMontgomeryUInt(self.n))

        # Free all ancilla qubits in the zero state.
        bb.add(Free(QBit()), reg=f1)
        bb.add(Free(QBit()), reg=f2)
        bb.add(Free(QBit()), reg=f3)
        bb.add(Free(QBit()), reg=f4)
        bb.add(Free(QBit()), reg=ctrl)

        # Return the output registers.
        return {'a': a, 'b': b, 'x': x, 'y': y}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        cvs2: Union[list[int], HasLength]
        if isinstance(self.n, int):
            cvs2 = [0] * 2 * self.n
        else:
            cvs2 = HasLength(2 * self.n)
        return {
            MultiControlX(cvs=cvs2): 1,
            XGate().controlled(CtrlSpec(qdtypes=QMontgomeryUInt(self.n), cvs=[0, 0, 0])): 2,
            MultiControlX(cvs=[0] * 3): 1,
            CModSub(QMontgomeryUInt(self.n), mod=self.mod): 1,
            CModAdd(QMontgomeryUInt(self.n), mod=self.mod): 1,
            Toffoli(): 2 * self.n + 4,
            Equals(QMontgomeryUInt(2 * self.n)): 1,
            MultiAnd(cvs=cvs2): 1,
            MultiTargetCNOT(2): 1,
            MultiAnd(cvs=cvs2).adjoint(): 1,
        }


@frozen
class ECAdd(Bloq):
    r"""Add two elliptic curve points.

    This takes elliptic curve points given by (a, b) and (x, y)
    and outputs the sum (x_r, y_r) in the second pair of registers.

    Because the decomposition of this Bloq is complex, we split it into six separate parts
    corresponding to the parts described in figure 10 of the Litinski paper cited below. We follow
    the signature from figure 5 and break down the further decompositions based on the steps in
    figure 10.

    Args:
        n: The bitsize of the two registers storing the elliptic curve point
        mod: The modulus of the field in which we do the addition.
        window_size: The number of bits in the ModMult window.

    Registers:
        a: The x component of the first input elliptic curve point of bitsize `n` in montgomery form.
        b: The y component of the first input elliptic curve point of bitsize `n` in montgomery form.
        x: The x component of the second input elliptic curve point of bitsize `n` in montgomery form, which
           will contain the x component of the resultant curve point.
        y: The y component of the second input elliptic curve point of bitsize `n` in montgomery form, which
           will contain the y component of the resultant curve point.
        lam_r: The precomputed lambda slope used in the addition operation if (a, b) = (x, y) in montgomery form.

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585).
        Litinski. 2023. Fig 5.
    """

    n: 'SymbolicInt'
    mod: 'SymbolicInt'
    window_size: 'SymbolicInt' = 1

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('a', QMontgomeryUInt(self.n)),
                Register('b', QMontgomeryUInt(self.n)),
                Register('x', QMontgomeryUInt(self.n)),
                Register('y', QMontgomeryUInt(self.n)),
                Register('lam_r', QMontgomeryUInt(self.n)),
            ]
        )

    def build_composite_bloq(
        self, bb: 'BloqBuilder', a: Soquet, b: Soquet, x: Soquet, y: Soquet, lam_r: Soquet
    ) -> Dict[str, 'SoquetT']:
        f1, f2, f3, f4, ctrl, a, b, x, y = bb.add(
            _ECAddStepOne(n=self.n, mod=self.mod), a=a, b=b, x=x, y=y
        )
        f1, ctrl, a, b, x, y, lam, lam_r = bb.add(
            _ECAddStepTwo(n=self.n, mod=self.mod, window_size=self.window_size),
            f1=f1,
            ctrl=ctrl,
            a=a,
            b=b,
            x=x,
            y=y,
            lam_r=lam_r,
        )
        ctrl, a, b, x, y, lam = bb.add(
            _ECAddStepThree(n=self.n, mod=self.mod, window_size=self.window_size),
            ctrl=ctrl,
            a=a,
            b=b,
            x=x,
            y=y,
            lam=lam,
        )
        x, y, lam = bb.add(
            _ECAddStepFour(n=self.n, mod=self.mod, window_size=self.window_size), x=x, y=y, lam=lam
        )
        ctrl, a, b, x, y, lam_r = bb.add(
            _ECAddStepFive(n=self.n, mod=self.mod, window_size=self.window_size),
            ctrl=ctrl,
            a=a,
            b=b,
            x=x,
            y=y,
            lam_r=lam_r,
            lam=lam,
        )
        a, b, x, y = bb.add(
            _ECAddStepSix(n=self.n, mod=self.mod),
            f1=f1,
            f2=f2,
            f3=f3,
            f4=f4,
            ctrl=ctrl,
            a=a,
            b=b,
            x=x,
            y=y,
        )

        return {'a': a, 'b': b, 'x': x, 'y': y, 'lam_r': lam_r}

    def on_classical_vals(self, a, b, x, y, lam_r) -> Dict[str, Union['ClassicalValT', sympy.Expr]]:
        dtype = QMontgomeryUInt(self.n, self.mod)
        curve_a = (
            dtype.montgomery_to_uint(lam_r) * 2 * dtype.montgomery_to_uint(b)
            - (3 * dtype.montgomery_to_uint(a) ** 2)
        ) % self.mod
        p1 = ECPoint(
            dtype.montgomery_to_uint(a), dtype.montgomery_to_uint(b), mod=self.mod, curve_a=curve_a
        )
        p2 = ECPoint(
            dtype.montgomery_to_uint(x), dtype.montgomery_to_uint(y), mod=self.mod, curve_a=curve_a
        )
        result = p1 + p2
        return {
            'a': a,
            'b': b,
            'x': dtype.uint_to_montgomery(result.x),
            'y': dtype.uint_to_montgomery(result.y),
            'lam_r': lam_r,
        }

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {
            _ECAddStepOne(n=self.n, mod=self.mod): 1,
            _ECAddStepTwo(n=self.n, mod=self.mod, window_size=self.window_size): 1,
            _ECAddStepThree(n=self.n, mod=self.mod, window_size=self.window_size): 1,
            _ECAddStepFour(n=self.n, mod=self.mod, window_size=self.window_size): 1,
            _ECAddStepFive(n=self.n, mod=self.mod, window_size=self.window_size): 1,
            _ECAddStepSix(n=self.n, mod=self.mod): 1,
        }


@bloq_example
def _ec_add() -> ECAdd:
    n, p = sympy.symbols('n p')
    ec_add = ECAdd(n, mod=p)
    return ec_add


@bloq_example
def _ec_add_small() -> ECAdd:
    ec_add_small = ECAdd(5, mod=7)
    return ec_add_small


_EC_ADD_DOC = BloqDocSpec(bloq_cls=ECAdd, examples=[_ec_add, _ec_add_small])
