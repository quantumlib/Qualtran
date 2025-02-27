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
from typing import cast, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
import sympy
from attrs import evolve, field, frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    DecomposeTypeError,
    QAny,
    QBit,
    QMontgomeryUInt,
    Register,
    Side,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.arithmetic.addition import AddK
from qualtran.bloqs.arithmetic.bitwise import BitwiseNot, XorK
from qualtran.bloqs.arithmetic.comparison import LinearDepthHalfGreaterThan
from qualtran.bloqs.arithmetic.controlled_addition import CAdd
from qualtran.bloqs.basic_gates import CNOT, CSwap, TwoBitCSwap, XGate
from qualtran.bloqs.mcmt import And, MultiAnd
from qualtran.bloqs.mod_arithmetic.mod_multiplication import ModDbl
from qualtran.symbolics import HasLength, is_symbolic

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT
    from qualtran.symbolics import SymbolicInt


@frozen
class _KaliskiIterationStep1(Bloq):
    """The first layer of operations in figure 15 of https://arxiv.org/pdf/2302.06639."""

    bitsize: 'SymbolicInt'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('v', QMontgomeryUInt(self.bitsize)),
                Register('m', QBit()),
                Register('f', QBit()),
                Register('is_terminal', QBit()),
            ]
        )

    def on_classical_vals(
        self, v: int, m: int, f: int, is_terminal: int
    ) -> Dict[str, 'ClassicalValT']:
        m ^= f & (v == 0)
        assert is_terminal == 0
        is_terminal ^= m
        f ^= m
        return {'v': v, 'm': m, 'f': f, 'is_terminal': is_terminal}

    def build_composite_bloq(
        self, bb: 'BloqBuilder', v: Soquet, m: Soquet, f: Soquet, is_terminal: Soquet
    ) -> Dict[str, 'SoquetT']:
        if is_symbolic(self.bitsize):
            raise DecomposeTypeError(f'symbolic decomposition is not supported for {self}')
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
        m, is_terminal = bb.add(CNOT(), ctrl=m, target=is_terminal)
        return {'v': v, 'm': m, 'f': f, 'is_terminal': is_terminal}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        if is_symbolic(self.bitsize):
            cvs: Union[HasLength, List[int]] = HasLength(self.bitsize + 1)
        else:
            cvs = [0] * int(self.bitsize) + [1]
        return {MultiAnd(cvs=cvs): 1, MultiAnd(cvs=cvs).adjoint(): 1, CNOT(): 3}


@frozen
class _KaliskiIterationStep2(Bloq):
    """The second layer of operations in figure 15 of https://arxiv.org/pdf/2302.06639."""

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
        if is_symbolic(self.bitsize):
            raise DecomposeTypeError(f"Cannot decompose {self} with symbolic `bitsize`.")

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
    """The third layer of operations in figure 15 of https://arxiv.org/pdf/2302.06639."""

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

    def build_composite_bloq(
        self, bb: 'BloqBuilder', u: Soquet, v: Soquet, b: Soquet, a: Soquet, m: Soquet, f: Soquet
    ) -> Dict[str, 'SoquetT']:
        u, v, junk_c, greater_than = bb.add(
            LinearDepthHalfGreaterThan(QMontgomeryUInt(self.bitsize)), a=u, b=v
        )

        (greater_than, f, b), junk_m, ctrl = bb.add(
            MultiAnd(cvs=(1, 1, 0)), ctrl=(greater_than, f, b)
        )

        ctrl, a = bb.add(CNOT(), ctrl=ctrl, target=a)
        ctrl, m = bb.add(CNOT(), ctrl=ctrl, target=m)

        greater_than, f, b = bb.add(
            MultiAnd(cvs=(1, 1, 0)).adjoint(), ctrl=(greater_than, f, b), junk=junk_m, target=ctrl
        )
        u, v = bb.add(
            LinearDepthHalfGreaterThan(QMontgomeryUInt(self.bitsize)).adjoint(),
            a=u,
            b=v,
            c=junk_c,
            target=greater_than,
        )
        return {'u': u, 'v': v, 'b': b, 'a': a, 'm': m, 'f': f}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {
            LinearDepthHalfGreaterThan(QMontgomeryUInt(self.bitsize)): 1,
            LinearDepthHalfGreaterThan(QMontgomeryUInt(self.bitsize)).adjoint(): 1,
            MultiAnd((1, 1, 0)): 1,
            MultiAnd((1, 1, 0)).adjoint(): 1,
            CNOT(): 2,
        }


@frozen
class _KaliskiIterationStep4(Bloq):
    """The fourth layer of operations in figure 15 of https://arxiv.org/pdf/2302.06639."""

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
        a, u, v = bb.add(CSwap(self.bitsize), ctrl=a, x=u, y=v)
        a, r, s = bb.add(CSwap(self.bitsize), ctrl=a, x=r, y=s)
        return {'u': u, 'v': v, 'r': r, 's': s, 'a': a}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {CSwap(self.bitsize): 2}


@frozen
class _KaliskiIterationStep5(Bloq):
    """The fifth layer of operations in figure 15 of https://arxiv.org/pdf/2302.06639."""

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
            And(1, 0): 1,
            And(1, 0).adjoint(): 1,
            BitwiseNot(QMontgomeryUInt(self.bitsize)): 2,
            CAdd(QMontgomeryUInt(self.bitsize)): 2,
        }


@frozen
class _KaliskiIterationStep6(Bloq):
    """The sixth layer of operations in figure 15 of https://arxiv.org/pdf/2302.06639."""

    bitsize: 'SymbolicInt'
    mod: 'SymbolicInt'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('u', QMontgomeryUInt(self.bitsize, self.mod)),
                Register('v', QMontgomeryUInt(self.bitsize, self.mod)),
                Register('r', QMontgomeryUInt(self.bitsize, self.mod)),
                Register('s', QMontgomeryUInt(self.bitsize, self.mod)),
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
        if is_symbolic(self.bitsize, self.mod):
            raise DecomposeTypeError(f'symbolic decomposition is not supported for {self}')
        m, b = bb.add(CNOT(), ctrl=m, target=b)
        a, b = bb.add(CNOT(), ctrl=a, target=b)

        # Controlled Divison by 2. The control bit is set only iff the number is even so the divison becomes equivalent to a cyclic right shift.
        v_arr = bb.split(v)
        for i in reversed(range(self.bitsize - 1)):
            f, v_arr[i], v_arr[i + 1] = bb.add(TwoBitCSwap(), ctrl=f, x=v_arr[i], y=v_arr[i + 1])
        v = bb.join(v_arr)

        r = bb.add(ModDbl(QMontgomeryUInt(self.bitsize), self.mod), x=r)

        a, u, v = bb.add(CSwap(self.bitsize), ctrl=a, x=u, y=v)
        a, r, s = bb.add(CSwap(self.bitsize), ctrl=a, x=r, y=s)

        s_arr = bb.split(s)
        s_arr[-1] = bb.add(XGate(), q=s_arr[-1])
        s_arr[-1], a = bb.add(CNOT(), ctrl=s_arr[-1], target=a)
        s_arr[-1] = bb.add(XGate(), q=s_arr[-1])
        s = bb.join(s_arr)

        return {'u': u, 'v': v, 'r': r, 's': s, 'b': b, 'a': a, 'm': m, 'f': f}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {
            CNOT(): 3,
            XGate(): 2,
            ModDbl(QMontgomeryUInt(self.bitsize), self.mod): 1,
            CSwap(self.bitsize): 2,
            TwoBitCSwap(): self.bitsize - 1,
        }


@frozen
class _KaliskiIteration(Bloq):
    """The single full iteration of Kaliski. see figure 15 of https://arxiv.org/pdf/2302.06639."""

    bitsize: 'SymbolicInt'
    mod: 'SymbolicInt'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('u', QMontgomeryUInt(self.bitsize, self.mod)),
                Register('v', QMontgomeryUInt(self.bitsize, self.mod)),
                Register('r', QMontgomeryUInt(self.bitsize, self.mod)),
                Register('s', QMontgomeryUInt(self.bitsize, self.mod)),
                Register('m', QBit()),
                Register('f', QBit()),
                Register('is_terminal', QBit()),
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
        is_terminal: Soquet,
    ) -> Dict[str, 'SoquetT']:
        a = bb.allocate(1)
        b = bb.allocate(1)

        v, m, f, is_terminal = bb.add(
            _KaliskiIterationStep1(self.bitsize), v=v, m=m, f=f, is_terminal=is_terminal
        )
        u, v, b, a, m, f = bb.add(
            _KaliskiIterationStep2(self.bitsize), u=u, v=v, b=b, a=a, m=m, f=f
        )
        u, v, b, a, m, f = bb.add(
            _KaliskiIterationStep3(self.bitsize), u=u, v=v, b=b, a=a, m=m, f=f
        )
        u, v, r, s, a = bb.add(_KaliskiIterationStep4(self.bitsize), u=u, v=v, r=r, s=s, a=a)
        u, v, r, s, b, f = bb.add(
            _KaliskiIterationStep5(self.bitsize), u=u, v=v, r=r, s=s, b=b, f=f
        )
        u, v, r, s, b, a, m, f = bb.add(
            _KaliskiIterationStep6(self.bitsize, self.mod), u=u, v=v, r=r, s=s, b=b, a=a, m=m, f=f
        )

        bb.free(a)
        bb.free(b)
        return {'u': u, 'v': v, 'r': r, 's': s, 'm': m, 'f': f, 'is_terminal': is_terminal}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {
            _KaliskiIterationStep1(self.bitsize): 1,
            _KaliskiIterationStep2(self.bitsize): 1,
            _KaliskiIterationStep3(self.bitsize): 1,
            _KaliskiIterationStep4(self.bitsize): 1,
            _KaliskiIterationStep5(self.bitsize): 1,
            _KaliskiIterationStep6(self.bitsize, self.mod): 1,
        }

    def on_classical_vals(
        self, u: int, v: int, r: int, s: int, m: int, f: int, is_terminal: int
    ) -> Dict[str, 'ClassicalValT']:
        """This is the Kaliski algorithm as described in Fig7 of https://arxiv.org/pdf/2001.09580.

        The following implementation merges together the pseudocode from Fig7 of https://arxiv.org/pdf/2001.09580
        and the circuit in figure 15 of https://arxiv.org/pdf/2302.06639; This is in order to compute the values
        of `f` and `m`.
        """
        assert m == 0
        is_terminal = int(f == 1 and v == 0)
        if f == 0:
            # When `f = 0` this means that the algorithm is nearly over and that we just need to
            # double the value of `r`.
            r = (r << 1) % self.mod
        elif v == 0:
            # `v = 0` is the termination condition of the algorithm and it means that the only
            # remaining step is multiplying `r` by 2 raised to the number of remaining iterations.
            # Classically this translates into a `r = (r * pow(2, k, p))%p` where k is the number
            # of iterations left followed by a break statement.
            m = u & 1
            f = 0
            r = (r << 1) % self.mod
        else:
            m = ((u % 2 == 1) & (v % 2 == 0)) or (u % 2 == 1 and v % 2 == 1 and u > v)
            m = int(m)
            # Kaliski iteration as described in Fig7 of https://arxiv.org/pdf/2001.09580.
            swap = (u % 2 == 0 and v % 2 == 1) or (u % 2 == 1 and v % 2 == 1 and u > v)
            if swap:
                u, v = v, u
                r, s = s, r
            if u % 2 == 1 and v % 2 == 1:
                v -= u
                s += r
            assert v % 2 == 0, f'{u=} {v=} {swap=}'
            v >>= 1
            r = (r << 1) % self.mod
            if swap:
                u, v = v, u
                r, s = s, r
        return {'u': u, 'v': v, 'r': r, 's': s, 'm': m, 'f': f, 'is_terminal': is_terminal}


@frozen
class _KaliskiModInverseImpl(Bloq):
    """The full KaliskiIteration algorithm. see C5 https://arxiv.org/pdf/2302.06639"""

    bitsize: 'SymbolicInt'
    mod: 'SymbolicInt'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('u', QMontgomeryUInt(self.bitsize, self.mod)),
                Register('v', QMontgomeryUInt(self.bitsize, self.mod)),
                Register('r', QMontgomeryUInt(self.bitsize, self.mod)),
                Register('s', QMontgomeryUInt(self.bitsize, self.mod)),
                Register('m', QAny(2 * self.bitsize)),
                Register('f', QBit()),
                Register('terminal_condition', QAny(2 * self.bitsize)),
            ]
        )

    @cached_property
    def _kaliski_iteration(self):
        return _KaliskiIteration(self.bitsize, self.mod)

    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        u: Soquet,
        v: Soquet,
        r: Soquet,
        s: Soquet,
        m: Soquet,
        f: Soquet,
        terminal_condition: Soquet,
    ) -> Dict[str, 'SoquetT']:
        if is_symbolic(self.bitsize):
            raise DecomposeTypeError(f"Cannot decompose {self} with symbolic `bitsize`.")

        f = bb.add(XGate(), q=f)
        u = bb.add(XorK(QMontgomeryUInt(self.bitsize), self.mod), x=u)
        s = bb.add(XorK(QMontgomeryUInt(self.bitsize), 1), x=s)

        m_arr = bb.split(m)
        terminal_condition_arr = bb.split(terminal_condition)

        for i in range(2 * self.bitsize):
            u, v, r, s, m_arr[i], f, terminal_condition_arr[i] = bb.add(
                self._kaliski_iteration,
                u=u,
                v=v,
                r=r,
                s=s,
                m=m_arr[i],
                f=f,
                is_terminal=terminal_condition_arr[i],
            )

        r = bb.add(BitwiseNot(QMontgomeryUInt(self.bitsize)), x=r)
        r = bb.add(AddK(QMontgomeryUInt(self.bitsize), self.mod + 1), x=r)

        u = bb.add(XorK(QMontgomeryUInt(self.bitsize), 1), x=u)
        s = bb.add(XorK(QMontgomeryUInt(self.bitsize), self.mod), x=s)

        # This is an extra step not present in the original Kaliski algorithm in order to
        # handle the case of x=0. The invariant of the Kaliski algorithm is that that end of the
        # algorithm u=1, s=0, r=mod inverse. This happens for all cases where the modular inverse
        # exists (i.e. gcd(x, mod) = 1).
        # The case where the input is zero is important. Although mathematically the inverse
        # doesn't exist. For the bloq to be unitary it needs to map zero to itself.
        # When the input is zero, the terminal values of the registers are r=mod, u=v=mod^1=mod-1
        # (assuming odd modulus).
        # So we clean those registers conditioned on the first terminal qubit which is set
        # if and only if the input is zero.
        terminal_condition_arr[0], r = bb.add(
            XorK(QMontgomeryUInt(self.bitsize), self.mod).controlled(),
            ctrl=terminal_condition_arr[0],
            x=r,
        )
        terminal_condition_arr[0], u = bb.add(
            XorK(QMontgomeryUInt(self.bitsize), self.mod - 1).controlled(),
            ctrl=terminal_condition_arr[0],
            x=u,
        )
        terminal_condition_arr[0], s = bb.add(
            XorK(QMontgomeryUInt(self.bitsize), self.mod - 1).controlled(),
            ctrl=terminal_condition_arr[0],
            x=s,
        )

        m = bb.join(m_arr)
        terminal_condition = bb.join(terminal_condition_arr)
        return {
            'u': u,
            'v': v,
            'r': r,
            's': s,
            'm': m,
            'f': f,
            'terminal_condition': terminal_condition,
        }

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {
            self._kaliski_iteration: 2 * self.bitsize,
            BitwiseNot(QMontgomeryUInt(self.bitsize)): 1,
            AddK(QMontgomeryUInt(self.bitsize), self.mod + 1): 1,
            XGate(): 1,
            XorK(QMontgomeryUInt(self.bitsize), self.mod): 2,
            XorK(QMontgomeryUInt(self.bitsize), 1): 2,
            XorK(QMontgomeryUInt(self.bitsize), self.mod).controlled(): 1,
            XorK(QMontgomeryUInt(self.bitsize), self.mod - 1).controlled(): 2,
        }


@frozen
class KaliskiModInverse(Bloq):
    r"""Compute modular multiplicative inverse -inplace- of numbers in montgomery form.

    Applies the transformation
    $$
        \ket{x} \ket{0} \rightarrow \ket{x^{-1} 2^{2n} \mod p} \ket{\mathrm{garbage}}
    $$

    Args:
        bitsize: size of the number.
        mod: The integer modulus.
        uncompute: whether to compute or uncompute.

    Registers:
        x: The register for which we compute the multiplicative inverse.
        m: A 2*bitsize register of intermediate values needed for uncomputation.

    References:
        [Performance Analysis of a Repetition Cat Code Architecture: Computing 256-bit Elliptic Curve Logarithm in 9 Hours with 126 133 Cat Qubits](https://arxiv.org/abs/2302.06639)
            Appendix C5.

        [Improved quantum circuits for elliptic curve discrete logarithms](https://arxiv.org/abs/2001.09580)
            Fig 7(b)

        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585)
            page 8.
    """

    bitsize: 'SymbolicInt'
    mod: 'SymbolicInt' = field(validator=lambda _, __, v: is_symbolic(v) or v % 2 == 1)
    uncompute: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        side = Side.LEFT if self.uncompute else Side.RIGHT
        return Signature(
            [
                Register('x', QMontgomeryUInt(self.bitsize, self.mod)),
                Register('junk', QAny(4 * self.bitsize), side=side),
            ]
        )

    def build_composite_bloq(
        self, bb: 'BloqBuilder', x: Soquet, junk: Optional[Soquet] = None
    ) -> Dict[str, 'SoquetT']:
        if is_symbolic(self.bitsize):
            raise DecomposeTypeError(f"Cannot decompose {self} with symbolic `bitsize`.")

        u = bb.allocate(self.bitsize, QMontgomeryUInt(self.bitsize))
        r = bb.allocate(self.bitsize, QMontgomeryUInt(self.bitsize))
        s = bb.allocate(self.bitsize, QMontgomeryUInt(self.bitsize))
        f = bb.allocate(1)

        if self.uncompute:
            assert junk is not None
            junk_arr = bb.split(junk)
            m = bb.join(junk_arr[: 2 * self.bitsize])
            terminal_condition = bb.join(junk_arr[2 * self.bitsize :])
            u, x, r, s, m, f, terminal_condition = cast(
                Tuple[Soquet, Soquet, Soquet, Soquet, Soquet, Soquet, Soquet],
                bb.add_from(
                    _KaliskiModInverseImpl(self.bitsize, self.mod).adjoint(),
                    u=u,
                    v=r,
                    r=x,
                    s=s,
                    m=m,
                    f=f,
                    terminal_condition=terminal_condition,
                ),
            )
            bb.free(u)
            bb.free(r)
            bb.free(s)
            bb.free(m)
            bb.free(f)
            bb.free(terminal_condition)
            return {'x': x}

        m = bb.allocate(2 * self.bitsize)
        terminal_condition = bb.allocate(2 * self.bitsize)
        u, v, x, s, m, f, terminal_condition = cast(
            Tuple[Soquet, Soquet, Soquet, Soquet, Soquet, Soquet, Soquet],
            bb.add_from(
                _KaliskiModInverseImpl(self.bitsize, self.mod),
                u=u,
                v=x,
                r=r,
                s=s,
                m=m,
                f=f,
                terminal_condition=terminal_condition,
            ),
        )

        bb.free(u)
        bb.free(v)
        bb.free(s)
        bb.free(f)
        junk = bb.join(np.concatenate([bb.split(m), bb.split(terminal_condition)]))
        return {'x': x, 'junk': junk}

    def adjoint(self) -> 'KaliskiModInverse':
        return evolve(self, uncompute=not self.uncompute)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return _KaliskiModInverseImpl(self.bitsize, self.mod).build_call_graph(ssa)

    def on_classical_vals(self, x: int, junk: int = 0) -> Dict[str, 'ClassicalValT']:
        mod = int(self.mod)
        u, v, r, s, f = mod, x, 0, 1, 1
        terminal_condition = m = 0
        iteration = _KaliskiModInverseImpl(self.bitsize, self.mod)._kaliski_iteration
        for _ in range(2 * int(self.bitsize)):
            u, v, r, s, m_i, f, is_terminal = iteration.call_classically(
                u=u, v=v, r=r, s=s, m=0, f=f, is_terminal=0
            )
            m = (m << 1) | m_i
            terminal_condition = (terminal_condition << 1) | is_terminal
        assert u == 1 or (x == 0 and u == mod)
        assert s == self.mod or (x == 0 and s == 1)
        assert f == 0
        assert v == 0
        return {
            'x': (self.mod - r) if r else 0,
            'junk': m * 2 ** (2 * self.bitsize) + terminal_condition,
        }


@bloq_example
def _kaliskimodinverse_example() -> KaliskiModInverse:
    kaliskimodinverse_example = KaliskiModInverse(32, 10**9 + 7)
    return kaliskimodinverse_example


@bloq_example
def _kaliskimodinverse_symbolic() -> KaliskiModInverse:
    n, p = sympy.symbols('n p')
    kaliskimodinverse_symbolic = KaliskiModInverse(n, p)
    return kaliskimodinverse_symbolic


_KALISKI_MOD_INVERSE_DOC = BloqDocSpec(
    bloq_cls=KaliskiModInverse, examples=[_kaliskimodinverse_example, _kaliskimodinverse_symbolic]
)
