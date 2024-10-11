#  Copyright 2024 Google LLC
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
from typing import Dict, Set, TYPE_CHECKING, Union, Optional

import numpy as np
import sympy
from attrs import field, frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    QBit,
    QAny,
    QMontgomeryUInt,
    Register,
    Side,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.arithmetic.controlled_addition import CAdd
from qualtran.bloqs.arithmetic.bitwise import BitwiseNot
from qualtran.bloqs.arithmetic.addition import AddK
from qualtran.bloqs.arithmetic.subtraction import Subtract
from qualtran.bloqs.arithmetic.comparison import LinearDepthGreaterThan
from qualtran.bloqs.basic_gates import CNOT, TwoBitCSwap, XGate
from qualtran.bloqs.mcmt import And, MultiAnd
from qualtran.bloqs.mod_arithmetic.mod_multiplication import ModDbl
from qualtran.bloqs.swap_network import CSwapApprox
from qualtran.resource_counting import BloqCountDictT, BloqCountT
from qualtran.resource_counting._call_graph import SympySymbolAllocator
from qualtran.symbolics import HasLength, is_symbolic

if TYPE_CHECKING:
    from qualtran.simulation.classical_sim import ClassicalValT
    from qualtran.symbolics import SymbolicInt
    from qualtran.resource_counting import BloqCountDictT

@frozen
class _KaliskiIterationStep1(Bloq):
    bitsize: 'SymbolicInt'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('v', QMontgomeryUInt(self.bitsize)),
                Register('m', QBit()),
                Register('f', QBit()),
            ]
        )

    def on_classical_vals(self, v: int, m: int, f: int) -> Dict[str, 'ClassicalValT']:
        m ^= f & (v == 0)
        f ^= m
        return {'v': v, 'm': m, 'f': f}

    def build_composite_bloq(
        self, bb: 'BloqBuilder', v: Soquet, m: Soquet, f: Soquet
    ) -> Dict[str, 'SoquetT']:
        v_arr = bb.split(v)
        ctrls = np.concatenate([v_arr, [f]])
        ctrls, junk, target = bb.add(MultiAnd(cvs=[0] * self.bitsize + [1]), ctrl=ctrls)
        target, m = bb.add(CNOT(), ctrl=target, target=m)
        ctrls = bb.add(
            MultiAnd(cvs=[0] * self.bitsize + [1]).adjoint(), ctrl=ctrls, junk=junk, target=target
        )
        v_arr = ctrls[:-1]
        f = ctrls[-1]
        v = bb.join(v_arr)
        m, f = bb.add(CNOT(), ctrl=m, target=f)
        return {'v': v, 'm': m, 'f': f}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        if is_symbolic(self.bitsize):
            cvs = HasLength(self.bitsize)
        else:
            cvs = [0] * self.bitsize
        return {
            MultiAnd(cvs=cvs):1,
            MultiAnd(cvs=cvs).adjoint(): 1,
            CNOT(): 2,
        }
@frozen
class _KaliskiIterationStep2(Bloq):
    bitsize: 'SymbolicInt'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('u', QMontgomeryUInt(self.bitsize)),
                Register('v', QMontgomeryUInt(self.bitsize)),
                Register('b', QBit()),
                Register('a', QBit()),
                Register('m', QBit()),
                Register('f', QBit()),
            ]
        )

    def on_classical_vals(
        self, u: int, v: int, b: int, a: int, m: int, f: int
    ) -> Dict[str, 'ClassicalValT']:
        a ^= ((u & 1) == 0) & f
        m ^= ((v & 1) == 0) & (a == 0) & f
        b ^= a
        b ^= m
        return {'u': u, 'v': v, 'b': b, 'a': a, 'm': m, 'f': f}

    def build_composite_bloq(
        self, bb: 'BloqBuilder', u: Soquet, v: Soquet, b: Soquet, a: Soquet, m: Soquet, f: Soquet
    ) -> Dict[str, 'SoquetT']:
        u_arr = bb.split(u)
        v_arr = bb.split(v)

        (f, u_arr[-1]), c = bb.add(And(1, 0), ctrl=(f, u_arr[-1]))
        c, a = bb.add(CNOT(), ctrl=c, target=a)
        f, u_arr[-1] = bb.add(And(1, 0).adjoint(), ctrl=(f, u_arr[-1]), target=c)

        (f, v_arr[-1], a), junk, c = bb.add(MultiAnd(cvs=(1, 0, 0)), ctrl=(f, v_arr[-1], a))
        c, m = bb.add(CNOT(), ctrl=c, target=m)
        f, v_arr[-1], a = bb.add(
            MultiAnd(cvs=(1, 0, 0)).adjoint(), ctrl=(f, v_arr[-1], a), junk=junk, target=c
        )

        a, b = bb.add(CNOT(), ctrl=a, target=b)
        m, b = bb.add(CNOT(), ctrl=m, target=b)
        u = bb.join(u_arr)
        v = bb.join(v_arr)
        return {'u': u, 'v': v, 'b': b, 'a': a, 'm': m, 'f': f}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {
            And(1, 0): 1,
            And(1, 0).adjoint(): 1,
            CNOT(): 4,  
            MultiAnd((1, 0, 0)): 1, 
            MultiAnd((1, 0, 0)).adjoint(): 1, 
        }

@frozen
class _KaliskiIterationStep3(Bloq):
    bitsize: 'SymbolicInt'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('u', QMontgomeryUInt(self.bitsize)),
                Register('v', QMontgomeryUInt(self.bitsize)),
                Register('b', QBit()),
                Register('a', QBit()),
                Register('m', QBit()),
                Register('f', QBit()),
            ]
        )

    def on_classical_vals(
        self, u: int, v: int, b: int, a: int, m: int, f: int
    ) -> Dict[str, 'ClassicalValT']:
        c = (u > v) & (b == 0) & f
        a ^= c
        m ^= c
        return {'u': u, 'v': v, 'b': b, 'a': a, 'm': m, 'f': f}

    def build_composite_bloq(self, bb: 'BloqBuilder', u: Soquet, v: Soquet, b: Soquet, a: Soquet, m: Soquet, f: Soquet) -> Dict[str, 'SoquetT']:
        greater_than = bb.allocate(1)
        u, v, greater_than = bb.add(LinearDepthGreaterThan(self.bitsize, signed=False), a=u, b=v, target=greater_than)

        (greater_than, f, b), junk, ctrl = bb.add(MultiAnd(cvs=(1, 1, 0)), ctrl=(greater_than, f, b))
        
        ctrl, a = bb.add(CNOT(), ctrl=ctrl, target=a)
        ctrl, m = bb.add(CNOT(), ctrl=ctrl, target=m)

        greater_than, f, b = bb.add(MultiAnd(cvs=(1, 1, 0)).adjoint(), ctrl=(greater_than, f, b), junk=junk, target=ctrl)
        u, v, greater_than = bb.add(LinearDepthGreaterThan(self.bitsize), a=u, b=v, target=greater_than)
        bb.free(greater_than)
        return {
            'u': u,
            'v': v,
            'b': b,
            'a': a,
            'm': m,
            'f': f,
        }

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {
            LinearDepthGreaterThan(self.bitsize, signed=False): 2,
            MultiAnd((1, 1, 0)): 1,
            MultiAnd((1, 1, 0)).adjoint(): 1,
            CNOT(): 2,
        }

@frozen
class _KaliskiIterationStep4(Bloq):
    bitsize: 'SymbolicInt'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('u', QMontgomeryUInt(self.bitsize)),
                Register('v', QMontgomeryUInt(self.bitsize)),
                Register('r', QMontgomeryUInt(self.bitsize)),
                Register('s', QMontgomeryUInt(self.bitsize)),
                Register('a', QBit()),
            ]
        )

    def on_classical_vals(
        self, u: int, v: int, r: int, s: int, a: int
    ) -> Dict[str, 'ClassicalValT']:
        if a:
            u, v = v, u
            r, s = s, r
        return {'u': u, 'v': v, 'r': r, 's': s, 'a': a}

    def build_composite_bloq(
        self, bb: 'BloqBuilder', u: Soquet, v: Soquet, r: Soquet, s: Soquet, a: Soquet
    ) -> Dict[str, 'SoquetT']:
        # CSwapApprox is a CSWAP with a phase flip.
        # Since we are doing two SWAPs the overal phase is correct.
        a, u, v = bb.add(CSwapApprox(self.bitsize), ctrl=a, x=u, y=v)
        a, r, s = bb.add(CSwapApprox(self.bitsize), ctrl=a, x=r, y=s)
        return {'u': u, 'v': v, 'r': r, 's': s, 'a': a}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> 'BloqCountDictT':
        return {CSwapApprox(self.bitsize): 2}


@frozen
class _KaliskiIterationStep5(Bloq):
    bitsize: 'SymbolicInt'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('u', QMontgomeryUInt(self.bitsize)),
                Register('v', QMontgomeryUInt(self.bitsize)),
                Register('r', QMontgomeryUInt(self.bitsize)),
                Register('s', QMontgomeryUInt(self.bitsize)),
                Register('b', QBit()),
                Register('f', QBit()),
            ]
        )

    def on_classical_vals(
        self, u: int, v: int, r: int, s: int, b: int, f: int
    ) -> Dict[str, 'ClassicalValT']:
        if f and b == 0:
            v -= u
            s += r
        return {'u': u, 'v': v, 'r': r, 's': s, 'b': b, 'f': f}

    def build_composite_bloq(
        self, bb: 'BloqBuilder', u: Soquet, v: Soquet, r: Soquet, s: Soquet, b: Soquet, f: Soquet
    ) -> Dict[str, 'SoquetT']:
        (f, b), c = bb.add(And(1, 0), ctrl=(f, b))
        v = bb.add(BitwiseNot(QMontgomeryUInt(self.bitsize)), x=v)
        c, u, v = bb.add(CAdd(QMontgomeryUInt(self.bitsize)), ctrl=c, a=u, b=v)
        v = bb.add(BitwiseNot(QMontgomeryUInt(self.bitsize)), x=v)
        c, r, s = bb.add(CAdd(QMontgomeryUInt(self.bitsize)), ctrl=c, a=r, b=s)
        f, b = bb.add(And(1, 0).adjoint(), ctrl=(f, b), target=c)
        return {'u': u, 'v': v, 'r': r, 's': s, 'b': b, 'f': f}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {
            And(1, 0) : 1,
            And(1, 0).adjoint(): 1,
            BitwiseNot(QMontgomeryUInt(self.bitsize)): 2,
            CAdd(QMontgomeryUInt(self.bitsize)): 2,
        }

@frozen
class _KaliskiIterationStep6(Bloq):
    bitsize: 'SymbolicInt'
    mod: 'SymbolicInt'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('u', QMontgomeryUInt(self.bitsize)),
                Register('v', QMontgomeryUInt(self.bitsize)),
                Register('r', QMontgomeryUInt(self.bitsize)),
                Register('s', QMontgomeryUInt(self.bitsize)),
                Register('b', QBit()),
                Register('a', QBit()),
                Register('m', QBit()),
                Register('f', QBit()),
            ]
        )

    def on_classical_vals(
        self, u: int, v: int, r: int, s: int, b: int, a: int, m: int, f: int
    ) -> Dict[str, 'ClassicalValT']:
        b ^= m
        b ^= a
        if f:
            v >>= 1
        r = (2 * r) % self.mod
        if a:
            r, s = s, r
            u, v = v, u
        if s % 2 == 0:
            a ^= 1
        return {'u': u, 'v': v, 'r': r, 's': s, 'b': b, 'a': a, 'm': m, 'f': f}

    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        u: Soquet,
        v: Soquet,
        r: Soquet,
        s: Soquet,
        b: Soquet,
        a: Soquet,
        m: Soquet,
        f: Soquet,
    ) -> Dict[str, 'SoquetT']:
        m, b = bb.add(CNOT(), ctrl=m, target=b)
        a, b = bb.add(CNOT(), ctrl=a, target=b)

        # Controlled Divison by 2. The control bit is set only iff the number is even so the divison becomes equivalent to a cyclic right shift.
        v_arr = bb.split(v)
        for i in reversed(range(self.bitsize - 1)):
            f, v_arr[i], v_arr[i + 1] = bb.add(TwoBitCSwap(), ctrl=f, x=v_arr[i], y=v_arr[i + 1])
        v = bb.join(v_arr)

        r = bb.add(ModDbl(QMontgomeryUInt(self.bitsize), self.mod), x=r)

        a, u, v = bb.add(CSwapApprox(self.bitsize), ctrl=a, x=u, y=v)
        a, r, s = bb.add(CSwapApprox(self.bitsize), ctrl=a, x=r, y=s)

        s_arr = bb.split(s)
        s_arr[-1] = bb.add(XGate(), q=s_arr[-1])
        s_arr[-1], a = bb.add(CNOT(), ctrl=s_arr[-1], target=a)
        s_arr[-1] = bb.add(XGate(), q=s_arr[-1])
        s = bb.join(s_arr)

        return {'u': u, 'v': v, 'r': r, 's': s, 'b': b, 'a': a, 'm': m, 'f': f}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {
            CNOT(): 4,
            XGate(): 2,
            ModDbl(QMontgomeryUInt(self.bitsize), self.mod): 1,
            CSwapApprox(self.bitsize): 2,
            TwoBitCSwap(): self.bitsize - 1,
        }

@frozen
class KaliskiIteration(Bloq):
    bitsize: 'SymbolicInt'
    mod: 'SymbolicInt'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('u', QMontgomeryUInt(self.bitsize)),
                Register('v', QMontgomeryUInt(self.bitsize)),
                Register('r', QMontgomeryUInt(self.bitsize)),
                Register('s', QMontgomeryUInt(self.bitsize)),
                Register('m', QBit()),
                Register('f', QBit()),
            ]
        )

    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        u: Soquet,
        v: Soquet,
        r: Soquet,
        s: Soquet,
        m: Soquet,
        f: Soquet,
    ) -> Dict[str, 'SoquetT']:
        a = bb.allocate(1)
        b = bb.allocate(1)

        v, m, f = bb.add(_KaliskiIterationStep1(self.bitsize), v=v, m=m, f=f)
        u, v, b, a, m, f = bb.add(_KaliskiIterationStep2(self.bitsize), u=u, v=v, b=b, a=a, m=m, f=f)
        u, v, b, a, m, f = bb.add(_KaliskiIterationStep3(self.bitsize), u=u, v=v, b=b, a=a, m=m, f=f)
        u, v, r, s, a = bb.add(_KaliskiIterationStep4(self.bitsize), u=u, v=v, r=r, s=s, a=a)
        u, v, r, s, b, f = bb.add(_KaliskiIterationStep5(self.bitsize), u=u, v=v, r=r, s=s, b=b, f=f)
        u, v, r, s, b, a, m, f = bb.add(_KaliskiIterationStep6(self.bitsize, self.mod), u=u, v=v, r=r, s=s, b=b, a=a, m=m, f=f)

        bb.free(a)
        bb.free(b)
        return {
            'u': u, 'v': v, 'r': r, 's': s, 'm': m, 'f': f,
        }


@frozen
class _KaliskiModInverseImpl(Bloq):
    bitsize: 'SymbolicInt'
    mod: 'SymbolicInt'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('u', QMontgomeryUInt(self.bitsize)),
                Register('v', QMontgomeryUInt(self.bitsize)),
                Register('r', QMontgomeryUInt(self.bitsize)),
                Register('s', QMontgomeryUInt(self.bitsize)),
                Register('m', QAny(2*self.bitsize)),
                Register('f', QBit()),
            ]
        )
    

    @cached_property
    def _kaliski_iteration(self):
        return KaliskiIteration(self.bitsize, self.mod)


    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        u: Soquet,
        v: Soquet,
        r: Soquet,
        s: Soquet,
        m: Soquet,
        f: Soquet,
    ) -> Dict[str, 'SoquetT']:
        f = bb.add(XGate(), q = f)
        m_arr = bb.split(m)

        for i in range(2*self.bitsize):
            u, v, r, s, m_arr[i], f = bb.add(self._kaliski_iteration, u=u, v=v, r=r, s=s, m=m_arr[i], f=f)

        r = bb.add(BitwiseNot(QMontgomeryUInt(self.bitsize)), x=r)
        r = bb.add(AddK(self.bitsize, self.mod + 1, signed=False), x=r)

        m = bb.join(m_arr)
        return {
            'u': u,
            'v': v,
            'r': r,
            's': s,
            'm': m,
            'f': f,
        }
    
    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {
            self._kaliski_iteration:2*self.bitsize,
            BitwiseNot(QMontgomeryUInt(self.bitsize)): 1,
            AddK(self.bitsize, self.mod + 1, signed=False): 1,
            XGate(): 1,
        }

@frozen
class KaliskiModInverse(Bloq):
    bitsize: 'SymbolicInt'
    mod: 'SymbolicInt'
    uncompute: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        side = Side.LEFT if self.uncompute else Side.RIGHT
        return Signature(
            [
                Register('u', QMontgomeryUInt(self.bitsize)),
                Register('v', QMontgomeryUInt(self.bitsize)),
                Register('r', QMontgomeryUInt(self.bitsize)),
                Register('s', QMontgomeryUInt(self.bitsize)),
                Register('m', QAny(2*self.bitsize), side=side),
                Register('f', QBit(), side=side),
            ]
        )
    

    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        u: Soquet,
        v: Soquet,
        r: Soquet,
        s: Soquet,
        m: Optional[Soquet] = None,
        f: Optional[Soquet] = None,
    ) -> Dict[str, 'SoquetT']:

        if self.uncompute:
            u, v, r, s, m, f = bb.add_from(_KaliskiModInverseImpl(self.bitsize, self.mod).adjoint(), u=u, v=v, r=r, s=s, m=m, f=f)
            bb.free(m)
            bb.free(f)
            return {
                'u': u,
                'v': v,
                'r': r,
                's': s,
            }

        m = bb.allocate(2*self.bitsize)
        # m = bb.split(m)
        f = bb.allocate(1)
        u, v, r, s, m, f = bb.add_from(_KaliskiModInverseImpl(self.bitsize, self.mod), u=u, v=v, r=r, s=s, m=m, f=f)
        return {
            'u': u,
            'v': v,
            'r': r,
            's': s,
            'm': m,
            'f': f,
        }

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return _KaliskiModInverseImpl(self.bitsize, self.mod).build_call_graph(ssa)