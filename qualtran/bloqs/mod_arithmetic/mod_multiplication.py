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

import math
import numbers
from functools import cached_property
from typing import cast, Dict, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union

import attrs
import numpy as np
import sympy
from attrs import frozen
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    DecomposeNotImplementedError,
    QBit,
    QMontgomeryUInt,
    QUInt,
    Register,
    Side,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.arithmetic import Add, AddK, CAdd, Xor
from qualtran.bloqs.arithmetic.comparison import LessThanConstant
from qualtran.bloqs.basic_gates import CNOT, CSwap, XGate
from qualtran.bloqs.data_loading.qroam_clean import QROAMClean
from qualtran.bloqs.mod_arithmetic.mod_addition import CtrlScaleModAdd
from qualtran.drawing import Circle, directional_text_box, Text, WireSymbol
from qualtran.resource_counting import BloqCountDictT, BloqCountT, SympySymbolAllocator
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join
from qualtran.simulation.classical_sim import ClassicalValT
from qualtran.symbolics import is_symbolic, Shaped

if TYPE_CHECKING:
    from qualtran.symbolics import SymbolicInt


@frozen
class ModDbl(Bloq):
    r"""An n-bit modular doubling gate.

    Implements $\ket{x} \rightarrow \ket{2x \mod p}$ using $2n$ Toffoli gates.

    Args:
        dtype: Dtype of the number to double.
        p: The modulus for the doubling.

    Registers:
        x: The register containing the number to double.

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585)
        Fig 6d and 8
    """

    dtype: Union[QUInt, QMontgomeryUInt]
    mod: int = attrs.field()

    @mod.validator
    def _validate_mod(self, attribute, value):
        assert isinstance(value, numbers.Integral) or is_symbolic(value)
        if isinstance(value, numbers.Integral):
            assert value % 2 == 1

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', self.dtype)])

    def on_classical_vals(self, x: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        if x < self.mod:
            x = (x + x) % self.mod
        return {'x': x}

    def build_composite_bloq(self, bb: 'BloqBuilder', x: Soquet) -> Dict[str, 'SoquetT']:
        # Allocate ancilla bits for sign and double.
        lower_bit = bb.allocate(n=1)
        sign = bb.allocate(n=1)

        # Convert x to an n + 2-bit integer by attaching two |0⟩ qubits as the least and most
        # significant bits.
        x_split = bb.split(x)
        x = bb.join(
            np.concatenate([[sign], x_split, [lower_bit]]),
            dtype=attrs.evolve(self.dtype, bitsize=self.dtype.bitsize + 2),
        )

        # Add constant -p to the x register.
        x = bb.add(AddK(bitsize=self.dtype.bitsize + 2, k=-self.mod, signed=False), x=x)

        # Split the three bit pieces again so that we can use the sign to control our constant
        # addition circuit.
        x_split = bb.split(x)
        sign = x_split[0]
        x = bb.join(x_split[1:], dtype=attrs.evolve(self.dtype, bitsize=self.dtype.bitsize + 1))

        # Add constant p to the x register if the result of the last modular reduction is negative.
        (sign,), x = bb.add(
            AddK(bitsize=self.dtype.bitsize + 1, k=self.mod, signed=False, cvs=(1,)),
            ctrls=(sign,),
            x=x,
        )

        # Split the lower bit ancilla from the x register for use in resetting the other ancilla bit
        # before freeing them both.
        x_split = bb.split(x)
        lower_bit = x_split[-1]
        lower_bit = bb.add(XGate(), q=lower_bit)
        lower_bit, sign = bb.add(CNOT(), ctrl=lower_bit, target=sign)
        lower_bit = bb.add(XGate(), q=lower_bit)

        free_bit = x_split[0]
        x = bb.join(np.concatenate([x_split[1:-1], [lower_bit]]), dtype=self.dtype)

        # Free the ancilla bits.
        bb.free(free_bit)
        bb.free(sign)

        # Return the output registers.
        return {'x': x}

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text(f'x = 2 * x mod {self.mod}')
        return super().wire_symbol(reg, idx)

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        return {
            AddK(self.dtype.bitsize + 2, -self.mod, signed=False): 1,
            AddK(self.dtype.bitsize + 1, self.mod, cvs=(1,), signed=False): 1,
            CNOT(): 1,
            XGate(): 2,
        }


@bloq_example
def _moddbl_small() -> ModDbl:
    moddbl_small = ModDbl(QUInt(4), 13)
    return moddbl_small


@bloq_example
def _moddbl_large() -> ModDbl:
    prime = 10**9 + 7
    moddbl_large = ModDbl(QUInt(32), prime)
    return moddbl_large


_MOD_DBL_DOC = BloqDocSpec(bloq_cls=ModDbl, examples=[_moddbl_small, _moddbl_large])


@frozen
class CModMulK(Bloq):
    r"""Perform controlled modular multiplication by a constant.

    Applies $\ket{c}\ket{c} \rightarrow \ket{c} \ket{x*k^c \mod p}$.

    Args:
        dtype: Dtype of the register.
        k: The integer multiplicative constant.
        mod: The integer modulus.

    Registers:
        ctrl: The control bit
        x: The integer being multiplied
    """

    dtype: Union[QUInt, QMontgomeryUInt]
    k: Union[int, sympy.Expr]
    mod: Union[int, sympy.Expr]

    def __attrs_post_init__(self):
        if is_symbolic(self.k, self.mod):
            return
        assert 0 < self.k < self.mod
        assert math.gcd(cast(int, self.k), cast(int, self.mod)) == 1

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('ctrl', QBit()), Register('x', self.dtype)])

    def _Add(self, k: Union[int, sympy.Expr]):
        """Helper method to forward attributes to `CtrlScaleModAdd`."""
        return CtrlScaleModAdd(k=k, bitsize=self.dtype.bitsize, mod=self.mod)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', ctrl: 'SoquetT', x: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        k = self.k
        if isinstance(self.mod, sympy.Expr) or isinstance(k, sympy.Expr):
            neg_k_inv = sympy.Mod(sympy.Pow(k, -1), self.mod)
        else:
            neg_k_inv = -pow(k, -1, mod=self.mod)

        # We store the result of the CtrlScaleModAdd into this new register
        # and then clear the original `x` register by multiplying in the inverse.
        y = bb.allocate(self.dtype.bitsize)

        # y += x*k
        ctrl, x, y = bb.add(self._Add(k=k), ctrl=ctrl, x=x, y=y)
        # x += y * (-k^-1)
        ctrl, y, x = bb.add(self._Add(k=neg_k_inv), ctrl=ctrl, x=y, y=x)

        # y contains the answer and x is empty.
        # In [GE2019], it is asserted that the registers can be swapped via bookkeeping.
        # This is not correct: we do not want to swap the registers if the control bit
        # is not set.
        ctrl, x, y = bb.add(CSwap(self.dtype.bitsize), ctrl=ctrl, x=x, y=y)
        bb.free(y)
        return {'ctrl': ctrl, 'x': x}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        k = ssa.new_symbol('k')
        return {self._Add(k=k): 2, CSwap(self.dtype.bitsize): 1}

    def on_classical_vals(self, ctrl, x) -> Dict[str, ClassicalValT]:
        if ctrl and x < self.mod:
            return {'ctrl': ctrl, 'x': (x * self.k) % self.mod}
        return {'ctrl': ctrl, 'x': x}

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text(f'x *= {self.k} % {self.mod}')
        if reg.name == 'ctrl':
            return Circle(filled=True)
        if reg.name == 'x':
            return directional_text_box(f'*={self.k}', side=reg.side)
        raise ValueError(f"Unknown register name: {reg.name}")


_K = sympy.Symbol('k_mul')


def _generalize_k(b: Bloq) -> Optional[Bloq]:
    if isinstance(b, CtrlScaleModAdd):
        return attrs.evolve(b, k=_K)

    return b


@bloq_example(generalizer=(ignore_split_join, ignore_alloc_free, _generalize_k))
def _modmul() -> CModMulK:
    modmul = CModMulK(QUInt(8), k=123, mod=13 * 17)
    return modmul


@bloq_example(generalizer=(ignore_split_join, ignore_alloc_free, _generalize_k))
def _modmul_symb() -> CModMulK:
    import sympy

    k, N, n_x = sympy.symbols('k N n_x')
    modmul_symb = CModMulK(QUInt(n_x), k=k, mod=N)
    return modmul_symb


_C_MOD_MUL_K_DOC = BloqDocSpec(bloq_cls=CModMulK, examples=(_modmul_symb, _modmul))


@frozen
class SingleWindowModMul(Bloq):
    r"""Performs modular multiplication on a single windowed.

    This bloq is used as a subroutine in DirtyOutOfPlaceMontgomeryModMul.

    Applies
    $$
        \ket{x}\key{y}\key{t}\ket{0} \rightarrow \ket{x}\key{y}\ket{t+xy \mod p} \ket{xy \mod 2^w}
    $$

    Where:

    - $w$ is the window size.
    - $p$ is the modulus.

    Args:
        window_size: size of the window (=size of the first register).
        bitsize: size of the second register.
        mod: The integer modulus.

    Registers:
        x: The first integer as an array of bits (`window_size` bits).
        y: The second integer (`bitsize` bits)
        target: product accumulation array of bits.
        qrom_index: contains the value $xy \mod 2^w$ (starts at 0).
    """

    window_size: 'SymbolicInt'
    bitsize: 'SymbolicInt'
    mod: 'SymbolicInt'

    def __attrs_post_init__(self):
        if not is_symbolic(self.bitsize, self.window_size):
            assert self.bitsize % self.window_size == 0

    @property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('x', QBit(), shape=(self.window_size,)),
                Register('y', QMontgomeryUInt(self.bitsize)),
                Register('target', QBit(), shape=(self.window_size + self.bitsize,)),
                Register('qrom_index', QMontgomeryUInt(self.window_size)),
            ]
        )

    @cached_property
    def qrom(self) -> QROAMClean:
        if is_symbolic(self.bitsize) or is_symbolic(self.window_size) or is_symbolic(self.mod):
            log_block_sizes = None
            if is_symbolic(self.bitsize) and not is_symbolic(self.window_size):
                # We assume that bitsize is much larger than window_size
                log_block_sizes = (0,)
            return QROAMClean(
                [Shaped((2**self.window_size,))],
                selection_bitsizes=(self.window_size,),
                target_bitsizes=(self.window_size + self.bitsize,),
                log_block_sizes=log_block_sizes,
            )
        inv_mod = pow(self.mod, 2 ** (self.window_size - 1) - 1, 2**self.window_size)
        N = 2**self.window_size
        data = (-np.arange(N) * inv_mod) % N
        data *= self.mod
        return QROAMClean(
            [data],
            selection_bitsizes=(self.window_size,),
            target_bitsizes=(self.window_size + self.bitsize,),
        )

    def on_classical_vals(self, x: Sequence[int], y: int, target: Sequence[int], qrom_index: int):
        if is_symbolic(self.bitsize) or is_symbolic(self.window_size):
            raise ValueError(f'classical action is not supported for {self}')
        dtype = QMontgomeryUInt(self.window_size + self.bitsize)
        target_val = QMontgomeryUInt.from_bits(dtype, target)
        for i in range(self.window_size):
            if x[i]:
                target_val += y << i
        qrom_index = target_val & (2**self.window_size - 1)
        Tm = self.qrom.data[0][qrom_index]
        target_val = (target_val + Tm) >> self.window_size
        target = QMontgomeryUInt.to_bits(dtype, target_val)
        return {'target': target, 'qrom_index': qrom_index, 'x': x, 'y': y}

    def build_composite_bloq(
        self, bb: 'BloqBuilder', x: NDArray[Soquet], y: Soquet, target: NDArray[Soquet], qrom_index: Soquet  # type: ignore[type-var]
    ):
        if is_symbolic(self.window_size):
            raise DecomposeNotImplementedError(f'symbolic decomposition not supported for {self}')
        for i in range(self.window_size):
            z = bb.join(target[-self.bitsize - 1 - i : len(target) - i])
            x[i], y, z = bb.add(
                CAdd(QMontgomeryUInt(self.bitsize), QMontgomeryUInt(self.bitsize + 1)),
                ctrl=x[i],
                a=y,
                b=z,
            )
            z_arr = bb.split(z)
            target[-self.bitsize - 1 - i : len(target) - i] = z_arr

        m = bb.join(target[-self.window_size :], QMontgomeryUInt(self.window_size))
        m, qrom_index = bb.add(Xor(QMontgomeryUInt(self.window_size)), x=m, y=qrom_index)
        target[-self.window_size :] = bb.split(m)

        qrom_index, qrom_target, *junk = bb.add(self.qrom, selection=qrom_index)
        z = bb.join(target)
        qrom_target, z = bb.add(
            Add(QMontgomeryUInt(self.bitsize + self.window_size)), a=qrom_target, b=z
        )
        if junk:
            assert len(junk) == 1
            qrom_index = bb.add(
                self.qrom.adjoint(),
                selection=qrom_index,
                target0_=qrom_target,
                junk_target0_=junk[0],
            )
        else:
            qrom_index = bb.add(self.qrom.adjoint(), selection=qrom_index, target0_=qrom_target)
        target_arr = bb.split(z)
        target_arr = np.roll(target_arr, self.window_size)

        return {'x': x, 'y': y, 'target': target_arr, 'qrom_index': qrom_index}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> BloqCountDictT:
        return {
            CAdd(
                QMontgomeryUInt(self.bitsize), QMontgomeryUInt(self.bitsize + 1)
            ): self.window_size,
            Xor(QMontgomeryUInt(self.window_size)): 1,
            Add(QMontgomeryUInt(self.bitsize + self.window_size)): 1,
            self.qrom: 1,
            self.qrom.adjoint(): 1,
        }


@frozen
class _DirtyOutOfPlaceMontgomeryModMulImpl(Bloq):
    r"""Perform windowed montgomery modular multiplication.

    Applies the trasformation
    $$
        \ket{x}\ket{y}\ket{0}\ket{0}\ket{0} \rightarrow \ket{x}\ket{y}\ket{xy2^{-n}}\ket{h}\ket{c}
    $$

    Where:

    - $n$ is the bitsize.
    - $x, y$ are in montgomery form
    - $h$ is an ancilla register that represents intermediate values.
    - $c$ is whether a final modular reduction was applied or not.

    Note: this is an internal implementation class that assumes the target registers (see above) are clean.

    Args:
        bitsize: size of the numbers.
        window_size: size of the window.
        mod: The integer modulus.
        uncompute: whether to compute or uncompute.

    Registers:
        x: The first integer
        y: The second integer
        target: product in montgomery form $xy 2^{-n}$
        qrom_indices: concatination of the indicies used to query QROM.
        reduced: whether a final modular reduction was applied.

    References:
        [Performance Analysis of a Repetition Cat Code Architecture: Computing 256-bit Elliptic Curve Logarithm in 9 Hours with 126 133 Cat Qubits](https://arxiv.org/abs/2302.06639)
            Appendix C4.

        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585)
            page 8.
    """

    bitsize: 'SymbolicInt'
    window_size: 'SymbolicInt'
    mod: 'SymbolicInt'

    def __attrs_post_init__(self):
        if isinstance(self.mod, int):
            assert self.mod > 1 and self.mod % 2 == 1  # Must be an odd integer greater than 1.

        if isinstance(self.mod, int) and isinstance(self.bitsize, int):
            assert 2 * self.mod - 1 < 2**self.bitsize, f'bitsize={self.bitsize} is too small'

        if isinstance(self.window_size, int) and isinstance(self.bitsize, int):
            assert self.window_size <= self.bitsize

    @cached_property
    def signature(self) -> 'Signature':
        num_windows = (
            self.bitsize + self.window_size - 1
        ) // self.window_size  # = ceil(self.bitsize/self.window_size)
        return Signature(
            [
                Register('x', QMontgomeryUInt(self.bitsize)),
                Register('y', QMontgomeryUInt(self.bitsize)),
                Register('target', QMontgomeryUInt(self.bitsize)),
                Register('qrom_indices', QMontgomeryUInt(num_windows * self.window_size)),
                Register('reduced', QBit()),
            ]
        )

    @cached_property
    def _window(self):
        return SingleWindowModMul(self.window_size, self.bitsize, self.mod)

    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        x: Soquet,
        y: Soquet,
        target: Soquet,
        qrom_indices: Soquet,
        reduced: Soquet,
    ) -> Dict[str, 'SoquetT']:
        if is_symbolic(self.window_size) or is_symbolic(self.bitsize) or is_symbolic(self.mod):
            raise DecomposeNotImplementedError(f'symbolic decomposition not supported for {self}')
        x_arr = bb.split(x)
        x_arr = np.flip(x_arr)

        target_arr = np.concatenate([bb.split(bb.allocate(self.window_size)), bb.split(target)])
        qrom_indices_arr = bb.split(qrom_indices)

        for i in range(0, self.bitsize, self.window_size):
            (x_arr[i : i + self.window_size], y, target_arr, qrom_index) = bb.add(
                self._window,
                x=x_arr[i : i + self.window_size],
                y=y,
                target=target_arr,
                qrom_index=bb.join(qrom_indices_arr[i : i + self.window_size]),
            )
            qrom_indices_arr[i : i + self.window_size] = bb.split(qrom_index)

        # Free ancillas and join
        bb.free(bb.join(target_arr[: -self.bitsize]))
        x_arr = np.flip(x_arr[: self.bitsize])
        x = bb.join(x_arr)
        qrom_indices = bb.join(qrom_indices_arr)

        # Modular reduction
        target = bb.join(target_arr[-self.bitsize :])
        reduced = bb.add(XGate(), q=reduced)
        target, reduced = bb.add(LessThanConstant(self.bitsize, self.mod), x=target, target=reduced)
        (reduced,), target = bb.add(
            AddK(self.bitsize, self.mod, cvs=(1,), signed=False), ctrls=(reduced,), x=target
        )

        return {'x': x, 'y': y, 'target': target, 'qrom_indices': qrom_indices, 'reduced': reduced}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> BloqCountDictT:
        num_windows = (self.bitsize + self.window_size - 1) // self.window_size
        return {
            AddK(self.bitsize, self.mod, cvs=(1,), signed=False): 1,
            LessThanConstant(bitsize=self.bitsize, less_than_val=self.mod): 1,
            XGate(): 1,
            self._window: num_windows,
        }


@frozen
class DirtyOutOfPlaceMontgomeryModMul(Bloq):
    r"""Perform windowed montgomery modular multiplication.

    Applies the trasformation
    $$
        \ket{x}\ket{y}\ket{0}\ket{0}\ket{0} \rightarrow \ket{x}\ket{y}\ket{xy2^{-n}}\ket{h}\ket{c}
    $$

    Where:

    - $n$ is the bitsize.
    - $x, y$ are in montgomery form
    - $h$ is an ancilla register that represents intermidate values.
    - $c$ is whether a final modular reduction was applied or not.

    Args:
        bitsize: size of the numbers.
        window_size: size of the window.
        mod: The integer modulus.
        uncompute: whether to compute or uncompute.

    Registers:
        x: The first integer
        y: The second integer
        target: product in montgomery form $xy 2^{-n}$
        qrom_indices: concatination of the indicies used to query QROM.
        reduced: whether a final modular reduction was applied.

    References:
        [Performance Analysis of a Repetition Cat Code Architecture: Computing 256-bit Elliptic Curve Logarithm in 9 Hours with 126 133 Cat Qubits](https://arxiv.org/abs/2302.06639)
            Appendix C4.

        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585)
            page 8.
    """

    bitsize: 'SymbolicInt'
    window_size: 'SymbolicInt'
    mod: 'SymbolicInt'
    uncompute: bool = False

    def __attrs_post_init__(self):
        if isinstance(self.mod, int):
            assert self.mod > 1 and self.mod % 2 == 1  # Must be an odd integer greater than 1.

        if isinstance(self.mod, int) and isinstance(self.bitsize, int):
            assert 2 * self.mod - 1 < 2**self.bitsize, f'bitsize={self.bitsize} is too small'

        if isinstance(self.window_size, int) and isinstance(self.bitsize, int):
            assert self.bitsize % self.window_size == 0

    @cached_property
    def signature(self) -> 'Signature':
        num_windows = (
            self.bitsize + self.window_size - 1
        ) // self.window_size  # = ceil(self.bitsize/self.window_size)
        side = Side.LEFT if self.uncompute else Side.RIGHT
        return Signature(
            [
                Register('x', QMontgomeryUInt(self.bitsize)),
                Register('y', QMontgomeryUInt(self.bitsize)),
                Register('target', QMontgomeryUInt(self.bitsize), side=side),
                Register(
                    'qrom_indices', QMontgomeryUInt(num_windows * self.window_size), side=side
                ),
                Register('reduced', QBit(), side=side),
            ]
        )

    def adjoint(self) -> 'DirtyOutOfPlaceMontgomeryModMul':
        return attrs.evolve(self, uncompute=self.uncompute ^ True)

    @cached_property
    def _inversion_data(self) -> np.typing.NDArray:
        inv_mod = pow(self.mod, 2 ** (self.window_size - 1) - 1, 2**self.window_size)
        N = 2**self.window_size
        data = (-np.arange(N) * inv_mod) % N
        data *= self.mod
        return data

    def _classical_action_window(
        self,
        x: 'ClassicalValT',
        y: 'ClassicalValT',
        target: 'ClassicalValT',
        qrom_indices: 'ClassicalValT',
    ):
        # This method implements same logic as SingleWindowModMul.on_classical_vals except that it works on integers rather than bit arrays.
        # Calls to this function are equivalent to calls to self._window.call_classically given the appropiate conversion int <-> bitarray.
        if is_symbolic(self.bitsize) or is_symbolic(self.window_size) or is_symbolic(self.mod):
            raise ValueError(f'classical action is not supported for {self}')
        for i in range(self.window_size):
            if (x >> i) & 1:
                target += y << i
        m = target & (2**self.window_size - 1)
        Tm = self._inversion_data[m]
        target += Tm
        target >>= self.window_size
        qrom_indices = (qrom_indices << self.window_size) | m
        return target, qrom_indices

    def on_classical_vals(
        self,
        x: 'ClassicalValT',
        y: 'ClassicalValT',
        target: Optional['ClassicalValT'] = None,
        qrom_indices: Optional['ClassicalValT'] = None,
        reduced: Optional['ClassicalValT'] = None,
    ) -> Dict[str, ClassicalValT]:
        if is_symbolic(self.bitsize) or is_symbolic(self.window_size) or is_symbolic(self.mod):
            raise ValueError(f'classical action is not supported for {self}')
        if self.uncompute:
            assert (
                target is not None and target == (x * y * pow(2, self.bitsize, self.mod)) % self.mod
            )
            assert qrom_indices is not None
            assert reduced is not None
            return {'x': x, 'y': y}
        assert target is None
        assert qrom_indices is None
        assert reduced is None

        if not (0 < x < self.mod and 0 < y < self.mod):
            return {'x': x, 'y': y, 'target': 0, 'qrom_indices': 0, 'reduced': 0}

        target = 0
        qrom_indices = 0
        reduced = 0
        for i in range(0, self.bitsize, self.window_size):
            target, qrom_indices = self._classical_action_window(x >> i, y, target, qrom_indices)

        if target >= self.mod:
            target -= self.mod
            reduced = 1

        montgomery_prod = (x * y * pow(2, self.bitsize * (self.mod - 2), self.mod)) % self.mod
        assert target == montgomery_prod
        return {'x': x, 'y': y, 'target': target, 'qrom_indices': qrom_indices, 'reduced': reduced}

    @cached_property
    def _mod_mul_impl(self) -> Bloq:
        b: Bloq = _DirtyOutOfPlaceMontgomeryModMulImpl(self.bitsize, self.window_size, self.mod)
        if self.uncompute:
            b = b.adjoint()
        return b

    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        x: Soquet,
        y: Soquet,
        target: Optional[Soquet] = None,
        qrom_indices: Optional[Soquet] = None,
        reduced: Optional[Soquet] = None,
    ) -> Dict[str, 'SoquetT']:
        if self.uncompute:
            assert target is not None
            assert qrom_indices is not None
            assert reduced is not None

            x, y, target, qrom_indices, reduced = bb.add_from(  # type: ignore
                self._mod_mul_impl,
                x=x,
                y=y,
                target=target,
                qrom_indices=qrom_indices,
                reduced=reduced,
            )

            bb.free(reduced)
            bb.free(qrom_indices)
            bb.free(target)
            return {'x': x, 'y': y}

        target = bb.allocate(self.bitsize, QMontgomeryUInt(self.bitsize))
        num_windows = (self.bitsize + self.window_size - 1) // self.window_size
        qrom_indices = bb.allocate(
            num_windows * self.window_size, QMontgomeryUInt(num_windows * self.window_size)
        )
        reduced = bb.allocate(1)

        x, y, target, qrom_indices, reduced = bb.add_from(
            self._mod_mul_impl, x=x, y=y, target=target, qrom_indices=qrom_indices, reduced=reduced
        )
        return {'x': x, 'y': y, 'target': target, 'qrom_indices': qrom_indices, 'reduced': reduced}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union[Set['BloqCountT'], BloqCountDictT]:
        return self._mod_mul_impl.build_call_graph(ssa)


@bloq_example(generalizer=[ignore_alloc_free, ignore_split_join])
def _dirtyoutofplacemontgomerymodmul_small() -> DirtyOutOfPlaceMontgomeryModMul:
    dirtyoutofplacemontgomerymodmul_small = DirtyOutOfPlaceMontgomeryModMul(6, 2, 7)
    return dirtyoutofplacemontgomerymodmul_small


@bloq_example(generalizer=[ignore_alloc_free, ignore_split_join])
def _dirtyoutofplacemontgomerymodmul_medium() -> DirtyOutOfPlaceMontgomeryModMul:
    dirtyoutofplacemontgomerymodmul_medium = DirtyOutOfPlaceMontgomeryModMul(
        bitsize=16, window_size=4, mod=2**15 - 1
    )
    return dirtyoutofplacemontgomerymodmul_medium


_DIRTY_OUT_OF_PLACE_MONTGOMERY_MOD_MUL_DOC = BloqDocSpec(
    bloq_cls=DirtyOutOfPlaceMontgomeryModMul,
    examples=(_dirtyoutofplacemontgomerymodmul_small, _dirtyoutofplacemontgomerymodmul_medium),
)
