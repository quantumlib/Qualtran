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
from typing import Dict, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
import sympy
from attrs import field, frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    DecomposeTypeError,
    GateWithRegisters,
    QBit,
    QMontgomeryUInt,
    QUInt,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.arithmetic.addition import Add, AddK
from qualtran.bloqs.arithmetic.comparison import CLinearDepthGreaterThan, LinearDepthGreaterThan
from qualtran.bloqs.arithmetic.controlled_addition import CAdd
from qualtran.bloqs.basic_gates import IntEffect, IntState, XGate
from qualtran.bloqs.bookkeeping import Cast
from qualtran.bloqs.mcmt.and_bloq import And
from qualtran.drawing import Circle, Text, TextBox, WireSymbol
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.simulation.classical_sim import ClassicalValT
from qualtran.symbolics import is_symbolic

if TYPE_CHECKING:
    from qualtran import BloqBuilder
    from qualtran.symbolics import SymbolicInt


@frozen
class ModAdd(Bloq):
    r"""An n-bit modular addition gate.

    Implements |x>|y> => |x>|y + x % p> using $4n$ Toffoli
    gates.

    This gate can also operate on integers in the Montgomery form.

    Args:
        bitsize: Number of bits used to represent each integer.
        mod: The modulus for the addition.

    Registers:
        x: A bitsize-sized input register (register x above).
        y: A bitsize-sized input/output register (register y above).

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585).
        Construction from Figure 6a and cost summary in Figure 8.
    """

    bitsize: 'SymbolicInt'
    mod: 'SymbolicInt'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('x', QMontgomeryUInt(self.bitsize, self.mod)),
                Register('y', QMontgomeryUInt(self.bitsize, self.mod)),
            ]
        )

    def on_classical_vals(
        self, x: 'ClassicalValT', y: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        # The construction still works when at most one of inputs equals `mod`.
        special_case = (x == self.mod) ^ (y == self.mod)
        if not (0 <= x < self.mod or special_case):
            raise ValueError(
                f'{x=} is outside the valid interval for modular addition [0, {self.mod}]'
            )
        if not (0 <= y < self.mod or special_case):
            raise ValueError(
                f'{y=} is outside the valid interval for modular addition [0, {self.mod}]'
            )

        y = (x + y) % self.mod
        return {'x': x, 'y': y}

    def build_composite_bloq(self, bb: 'BloqBuilder', x: Soquet, y: Soquet) -> Dict[str, 'SoquetT']:
        if is_symbolic(self.bitsize):
            raise DecomposeTypeError(f"Cannot decompose {self} with symbolic `bitsize`.")
        # Allocate ancilla bits for use in addition.
        junk_bit = bb.allocate(n=1)
        sign = bb.allocate(n=1)

        # Join ancilla bits to x and y registers in order to be able to compute addition of
        # bitsize+1 registers. This allows us to keep track of the sign of the y register after a
        # constant subtraction circuit.
        x_split = bb.split(x)
        y_split = bb.split(y)
        x = bb.join(
            np.concatenate([[junk_bit], x_split]), dtype=QMontgomeryUInt(bitsize=self.bitsize + 1)
        )
        y = bb.join(
            np.concatenate([[sign], y_split]), dtype=QMontgomeryUInt(bitsize=self.bitsize + 1)
        )

        # Perform in-place addition on quantum register y.
        x, y = bb.add(Add(QMontgomeryUInt(bitsize=self.bitsize + 1)), a=x, b=y)

        # Temporary solution to equalize the bitlength of the x and y registers for Add().
        x_split = bb.split(x)
        junk_bit = x_split[0]
        x = bb.join(x_split[1:], dtype=QMontgomeryUInt(bitsize=self.bitsize))

        # Add constant -p to the y register.
        y = bb.add(AddK(QMontgomeryUInt(self.bitsize + 1), k=-1 * self.mod), x=y)

        # Controlled addition of classical constant p if the sign of y after the last addition is
        # negative.
        y_split = bb.split(y)
        sign = y_split[0]
        y = bb.join(y_split[1:], dtype=QMontgomeryUInt(bitsize=self.bitsize))

        sign, y = bb.add(
            AddK(QMontgomeryUInt(self.bitsize), k=self.mod).controlled(), ctrl=sign, x=y
        )

        # Check if y < x; if yes flip the bit of the signed ancilla bit. Then bitflip the sign bit
        # again before freeing.
        x, y, sign = bb.add(
            LinearDepthGreaterThan(bitsize=self.bitsize, signed=False), a=x, b=y, target=sign
        )
        sign = bb.add(XGate(), q=sign)

        # Free the ancilla qubits.
        bb.free(junk_bit)
        bb.free(sign)

        # Return the output registers.
        return {'x': x, 'y': y}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {
            Add(QUInt(self.bitsize + 1)): 1,
            AddK(QUInt(self.bitsize + 1), k=-self.mod): 1,
            AddK(QUInt(self.bitsize), k=self.mod).controlled(): 1,
            LinearDepthGreaterThan(self.bitsize): 1,
            XGate(): 1,
        }

    def short_name(self) -> str:
        return f'y = y + x mod {self.mod}'


@bloq_example
def _mod_add() -> ModAdd:
    n, p = sympy.symbols('n p')
    mod_add = ModAdd(n, mod=p)
    return mod_add


_MOD_ADD_DOC = BloqDocSpec(bloq_cls=ModAdd, examples=[_mod_add])


@frozen(auto_attribs=True)
class ModAddK(GateWithRegisters):
    """Applies U(add, M)|x> = |(x + add) % M> if x < M else |x>.

    Applies modular addition to input register `|x>` given parameters `mod` and `add_val` s.t.
     1. If integer `x` < `mod`: output is `|(x + add) % M>`
     2. If integer `x` >= `mod`: output is `|x>`.

    This condition is needed to ensure that the mapping of all input basis states (i.e. input
    states |0>, |1>, ..., |2 ** bitsize - 1) to corresponding output states is bijective and thus
    the gate is reversible.

    Also supports controlled version of the gate by specifying a per qubit control value as a tuple
    of integers passed as `cvs`.
    """

    bitsize: int
    mod: int = field()
    add_val: int = 1
    cvs: Tuple[int, ...] = field(
        converter=lambda v: (v,) if isinstance(v, int) else tuple(v), default=()
    )

    @mod.validator
    def _validate_mod(self, attribute, value):
        if isinstance(value, sympy.Expr) or isinstance(self.bitsize, sympy.Expr):
            return
        if not 1 <= value <= 2**self.bitsize:
            raise ValueError(f"mod: {value} must be between [1, {2 ** self.bitsize}].")

    @cached_property
    def signature(self) -> Signature:
        if self.cvs:
            return Signature(
                [
                    Register('ctrl', QBit(), shape=(len(self.cvs),)),
                    Register('x', QUInt(self.bitsize)),
                ]
            )
        return Signature([Register('x', QUInt(self.bitsize))])

    def _classical_unctrled(self, target_val: int):
        if target_val < self.mod:
            return (target_val + self.add_val) % self.mod
        return target_val

    def on_classical_vals(
        self, *, x: int, ctrl: Optional[int] = None
    ) -> Dict[str, 'ClassicalValT']:
        out = self._classical_unctrled(x)
        if self.cvs:
            assert ctrl is not None
            if ctrl == int(''.join(str(x) for x in self.cvs), 2):
                return {'ctrl': ctrl, 'x': out}
            else:
                return {'ctrl': ctrl, 'x': x}

        assert ctrl is None
        return {'x': out}

    def __pow__(self, power: int) -> 'ModAddK':
        return ModAddK(self.bitsize, self.mod, add_val=self.add_val * power, cvs=self.cvs)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {Add(QUInt(self.bitsize), QUInt(self.bitsize)): 5}


@bloq_example
def _mod_add_k() -> ModAddK:
    n, m, k = sympy.symbols('n m k')
    mod_add_k = ModAddK(bitsize=n, mod=m, add_val=k)
    return mod_add_k


@bloq_example
def _mod_add_k_small() -> ModAddK:
    mod_add_k_small = ModAddK(bitsize=4, mod=7, add_val=1)
    return mod_add_k_small


@bloq_example
def _mod_add_k_large() -> ModAddK:
    mod_add_k_large = ModAddK(bitsize=64, mod=500, add_val=23)
    return mod_add_k_large


_MOD_ADD_K_DOC = BloqDocSpec(
    bloq_cls=ModAddK, examples=[_mod_add_k, _mod_add_k_small, _mod_add_k_large]
)


@frozen
class CModAddK(Bloq):
    """Perform x += k mod m for constant k, m and quantum x.

    Args:
        k: The integer to add to `x`.
        mod: The modulus for the addition.
        bitsize: The bitsize of the `x` register.

    Registers:
        ctrl: The control bit
        x: The register to perform the in-place modular addition.

    References:
        [How to factor 2048 bit RSA integers in 8 hours using 20 million noisy qubits](https://arxiv.org/abs/1905.09749).
        Gidney and Ekerå 2019.
        The reference implementation in section 2.2 uses CModAddK, but the circuit that it points
        to is just ModAdd (not ModAddK). This ModAdd is less efficient than the circuit later
        introduced in the Litinski paper so we choose to use that since it is more efficient and
        already implemented in Qualtran.

        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585).
        Litinski et al. 2023.
        This CModAdd circuit uses 2 fewer additions than the implementation referenced in the paper
        above. Because of this we choose to use this CModAdd bloq instead.
    """

    k: Union[int, sympy.Expr]
    mod: Union[int, sympy.Expr]
    bitsize: Union[int, sympy.Expr]

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('ctrl', QBit()), Register('x', QUInt(self.bitsize))])

    def build_composite_bloq(
        self, bb: 'BloqBuilder', ctrl: 'Soquet', x: 'Soquet'
    ) -> Dict[str, 'SoquetT']:
        k = bb.add(IntState(bitsize=self.bitsize, val=self.k))
        ctrl, k, x = bb.add(CModAdd(QUInt(self.bitsize), mod=self.mod), ctrl=ctrl, x=k, y=x)
        bb.add(IntEffect(bitsize=self.bitsize, val=self.k), val=k)
        return {'ctrl': ctrl, 'x': x}

    def on_classical_vals(
        self, ctrl: 'ClassicalValT', x: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        if ctrl == 0:
            return {'ctrl': 0, 'x': x}

        assert ctrl == 1, 'Bad ctrl value.'

        if not (0 <= x < self.mod):
            raise ValueError(
                f'{x=} is outside the valid interval for modular addition [0, {self.mod}]'
            )

        x = (x + self.k) % self.mod
        return {'ctrl': ctrl, 'x': x}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {CModAdd(QUInt(self.bitsize), mod=self.mod): 1}

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text(f"mod {self.mod}")
        if reg.name == 'ctrl':
            return Circle()
        if reg.name == 'x':
            return TextBox(f'x += {self.k}')
        raise ValueError(f"Unknown register {reg}")


@bloq_example(generalizer=ignore_split_join)
def _cmod_add_k() -> CModAddK:
    n, m, k = sympy.symbols('n m k')
    cmod_add_k = CModAddK(bitsize=n, mod=m, k=k)
    return cmod_add_k


@bloq_example
def _cmod_add_k_small() -> CModAddK:
    cmod_add_k_small = CModAddK(bitsize=4, mod=7, k=1)
    return cmod_add_k_small


_C_MOD_ADD_K_DOC = BloqDocSpec(bloq_cls=CModAddK, examples=[_cmod_add_k, _cmod_add_k_small])


@frozen
class CtrlScaleModAdd(Bloq):
    """Perform y += x*k mod m for constant k, m and quantum x, y.

    Args:
        k: The constant integer to scale `x` before adding into `y`.
        mod: The modulus of the addition
        bitsize: The size of the two registers.

    Registers:
        ctrl: The control bit
        x: The 'source' quantum register containing the integer to be scaled and added to `y`.
        y: The 'destination' quantum register to which the addition will apply.

    References:
        [How to factor 2048 bit RSA integers in 8 hours using 20 million noisy qubits](https://arxiv.org/abs/1905.09749).
        Construction based on description in section 2.2 paragraph 4. We add n And/And† bloqs
        because the bloq is controlled, but the construction also involves modular addition
        controlled on the qubits comprising register x.
    """

    k: Union[int, sympy.Expr]
    mod: Union[int, sympy.Expr]
    bitsize: Union[int, sympy.Expr]

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('ctrl', QBit()),
                Register('x', QUInt(self.bitsize)),
                Register('y', QUInt(self.bitsize)),
            ]
        )

    def build_composite_bloq(
        self, bb: 'BloqBuilder', ctrl: 'Soquet', x: 'Soquet', y: 'Soquet'
    ) -> Dict[str, 'SoquetT']:
        if is_symbolic(self.bitsize):
            raise DecomposeTypeError(f"Cannot decompose {self} with symbolic `bitsize`.")
        x_split = bb.split(x)
        for i in range(int(self.bitsize)):
            and_ctrl = [ctrl, x_split[i]]
            and_ctrl, ancilla = bb.add(And(), ctrl=and_ctrl)
            ancilla, y = bb.add(
                CModAddK(
                    k=((self.k * 2 ** (self.bitsize - 1 - i)) % self.mod),
                    bitsize=self.bitsize,
                    mod=self.mod,
                ),
                ctrl=ancilla,
                x=y,
            )
            and_ctrl = bb.add(And().adjoint(), ctrl=and_ctrl, target=ancilla)
            ctrl = and_ctrl[0]
            x_split[i] = and_ctrl[1]
        x = bb.join(x_split, dtype=QUInt(self.bitsize))

        return {'ctrl': ctrl, 'x': x, 'y': y}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        k = ssa.new_symbol('k')
        return {
            CModAddK(k=k, bitsize=self.bitsize, mod=self.mod): self.bitsize,
            And(): self.bitsize,
            And().adjoint(): self.bitsize,
        }

    def on_classical_vals(
        self, ctrl: 'ClassicalValT', x: 'ClassicalValT', y: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        if ctrl == 0:
            return {'ctrl': 0, 'x': x, 'y': y}

        assert ctrl == 1, 'Bad ctrl value.'
        y_out = (y + x * self.k) % self.mod
        return {'ctrl': ctrl, 'x': x, 'y': y_out}

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text(f"mod {self.mod}")
        if reg.name == 'ctrl':
            return Circle()
        if reg.name == 'x':
            return TextBox('x')
        if reg.name == 'y':
            return TextBox(f'y += x*{self.k}')
        raise ValueError(f"Unknown register {reg}")


@bloq_example(generalizer=ignore_split_join)
def _ctrl_scale_mod_add() -> CtrlScaleModAdd:
    n, m, k = sympy.symbols('n m k')
    ctrl_scale_mod_add = CtrlScaleModAdd(bitsize=n, mod=m, k=k)
    return ctrl_scale_mod_add


@bloq_example
def _ctrl_scale_mod_add_small() -> CtrlScaleModAdd:
    ctrl_scale_mod_add_small = CtrlScaleModAdd(bitsize=4, mod=7, k=1)
    return ctrl_scale_mod_add_small


_CTRL_SCALE_MOD_ADD_DOC = BloqDocSpec(
    bloq_cls=CtrlScaleModAdd, examples=[_ctrl_scale_mod_add, _ctrl_scale_mod_add_small]
)


@frozen
class CModAdd(Bloq):
    r"""Controlled Modular Addition.

    Implements $\ket{c}\ket{x}\ket{y} \rightarrow \ket{c}\ket{x}\ket{(cx+y)\%p}$
    using $5n+1$ Toffoli gates.

    Note: The reference reports $5n$ toffolis. Our construction has an extra toffoli gate due
    to the current implementaiton of `OutOfPlaceAdder`.

    Args:
        dtype: Type of the input registers.
        mod: The modulus for the addition.
        cv: Control value for which the gate is active.

    Registers:
        ctrl: The control qubit.
        x: A dtype register.
        y: A dtype register.

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585).
        Construction from Figure 6a and cost summary in Figure 8.
    """

    dtype: Union[QUInt, QMontgomeryUInt]
    mod: 'SymbolicInt'
    cv: int = 1

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [Register('ctrl', QBit()), Register('x', self.dtype), Register('y', self.dtype)]
        )

    def on_classical_vals(
        self, ctrl: 'ClassicalValT', x: 'ClassicalValT', y: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        if ctrl != self.cv:
            return {'ctrl': ctrl, 'x': x, 'y': y}

        # The construction still works when at most one of inputs equals `mod`.
        special_case = (x == self.mod) ^ (y == self.mod)
        if not (0 <= x < self.mod or special_case):
            raise ValueError(
                f'{x=} is outside the valid interval for modular addition [0, {self.mod}]'
            )
        if not (0 <= y < self.mod or special_case):
            raise ValueError(
                f'{y=} is outside the valid interval for modular addition [0, {self.mod}]'
            )

        y = (x + y) % self.mod
        return {'ctrl': ctrl, 'x': x, 'y': y}

    def build_composite_bloq(
        self, bb: 'BloqBuilder', ctrl, x: Soquet, y: Soquet
    ) -> Dict[str, 'SoquetT']:
        if is_symbolic(self.dtype.bitsize):
            raise DecomposeTypeError(f'symbolic decomposition is not supported for {self}')
        y_arr = bb.split(y)
        ancilla = bb.allocate(1)
        x = bb.add(Cast(self.dtype, QUInt(self.dtype.bitsize)), reg=x)
        y = bb.join(np.concatenate([[ancilla], y_arr]), QUInt(self.dtype.bitsize + 1))

        ctrl, x, y = bb.add(
            CAdd(QUInt(self.dtype.bitsize), QUInt(self.dtype.bitsize + 1), cv=self.cv),
            ctrl=ctrl,
            a=x,
            b=y,
        )
        y = bb.add(AddK(QUInt(self.dtype.bitsize + 1), -self.mod), x=y)
        y_arr = bb.split(y)
        ancilla, y_arr = y_arr[0], y_arr[1:]
        y = bb.join(y_arr)
        ancilla, y = bb.add(
            AddK(QUInt(self.dtype.bitsize), self.mod).controlled(), ctrl=ancilla, x=y
        )

        ctrl, x, y, ancilla = bb.add(
            CLinearDepthGreaterThan(QUInt(self.dtype.bitsize), cv=self.cv),
            ctrl=ctrl,
            a=x,
            b=y,
            target=ancilla,
        )
        ancilla = bb.add(XGate(), q=ancilla)
        bb.free(ancilla)

        x = bb.add(Cast(QUInt(self.dtype.bitsize), self.dtype), reg=x)
        y = bb.add(Cast(QUInt(self.dtype.bitsize), self.dtype), reg=y)

        return {'ctrl': ctrl, 'x': x, 'y': y}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {
            CAdd(QUInt(self.dtype.bitsize), QUInt(self.dtype.bitsize + 1), cv=self.cv): 1,
            AddK(QUInt(self.dtype.bitsize + 1), -self.mod): 1,
            AddK(QUInt(self.dtype.bitsize), self.mod).controlled(): 1,
            CLinearDepthGreaterThan(QUInt(self.dtype.bitsize), cv=self.cv): 1,
            XGate(): 1,
        }


@bloq_example(generalizer=ignore_split_join)
def _cmodadd_symbolic() -> CModAdd:
    n, p = sympy.symbols('n p')
    cmodadd_symbolic = CModAdd(QUInt(n), p)
    return cmodadd_symbolic


@bloq_example
def _cmodadd_example() -> CModAdd:
    cmodadd_example = CModAdd(QUInt(32), 10**9 + 7)
    return cmodadd_example


_C_MOD_ADD_DOC = BloqDocSpec(bloq_cls=CModAdd, examples=[_cmodadd_example, _cmodadd_symbolic])
