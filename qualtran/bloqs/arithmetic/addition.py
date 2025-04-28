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
from collections import Counter
from functools import cached_property
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import cirq
import numpy as np
import sympy
from attrs import evolve, field, frozen
from numpy.typing import NDArray

from qualtran import (
    AddControlledT,
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    CompositeBloq,
    CtrlSpec,
    DecomposeTypeError,
    GateWithRegisters,
    QInt,
    QMontgomeryUInt,
    QUInt,
    Register,
    Side,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.basic_gates import CNOT
from qualtran.bloqs.bookkeeping import Always
from qualtran.bloqs.mcmt.and_bloq import And
from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv
from qualtran.cirq_interop import decompose_from_cirq_style_method
from qualtran.drawing import directional_text_box, Text
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.simulation.classical_sim import add_ints
from qualtran.symbolics import is_symbolic, SymbolicInt

if TYPE_CHECKING:
    from qualtran.drawing import WireSymbol
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class Add(Bloq):
    r"""An n-bit addition gate.

    Implements $U|a\rangle|b\rangle \rightarrow |a\rangle|a+b\rangle$ using $4n - 4 T$ gates.

    Args:
        a_dtype: Quantum datatype used to represent the integer a.
        b_dtype: Quantum datatype used to represent the integer b. Must be large
            enough to hold the result in the output register of a + b, or else it simply
            drops the most significant bits. If not specified, b_dtype is set to a_dtype.

    Registers:
        a: A a_dtype.bitsize-sized input register (register a above).
        b: A b_dtype.bitsize-sized input/output register (register b above).

    References:
        [Halving the cost of quantum addition](https://arxiv.org/abs/1709.06648)
    """

    a_dtype: Union[QInt, QUInt, QMontgomeryUInt] = field()
    b_dtype: Union[QInt, QUInt, QMontgomeryUInt] = field()

    @b_dtype.default
    def b_dtype_default(self):
        return self.a_dtype

    @a_dtype.validator
    def _a_dtype_validate(self, field, val):
        if not isinstance(val, (QInt, QUInt, QMontgomeryUInt)):
            raise ValueError("Only QInt, QUInt and QMontgomerUInt types are supported.")
        if isinstance(val.num_qubits, sympy.Expr):
            return
        if val.bitsize > self.b_dtype.bitsize:
            raise ValueError("a_dtype bitsize must be less than or equal to b_dtype bitsize")

    @b_dtype.validator
    def _b_dtype_validate(self, field, val):
        if not isinstance(val, (QInt, QUInt, QMontgomeryUInt)):
            raise ValueError("Only QInt, QUInt and QMontgomerUInt types are supported.")

    @property
    def dtype(self):
        if self.a_dtype != self.b_dtype:
            raise ValueError(
                "Add.dtype is only supported when both operands have the same dtype: "
                f"{self.a_dtype=}, {self.b_dtype=}"
            )
        return self.a_dtype

    @property
    def signature(self):
        return Signature([Register("a", self.a_dtype), Register("b", self.b_dtype)])

    def decompose_bloq(self) -> 'CompositeBloq':
        return decompose_from_cirq_style_method(self)

    def on_classical_vals(
        self, a: 'ClassicalValT', b: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        unsigned = isinstance(self.a_dtype, (QUInt, QMontgomeryUInt))
        b_bitsize = self.b_dtype.bitsize
        return {
            'a': a,
            'b': add_ints(int(a), int(b), num_bits=int(b_bitsize), is_signed=not unsigned),
        }

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["In(x)"] * int(self.a_dtype.bitsize)
        wire_symbols += ["In(y)/Out(x+y)"] * int(self.b_dtype.bitsize)
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text("")
        if reg.name == 'a':
            return directional_text_box('a', side=reg.side)
        elif reg.name == 'b':
            return directional_text_box('a+b', side=reg.side)
        else:
            raise ValueError()

    def _left_building_block(self, inp, out, anc, depth):
        if depth == self.b_dtype.bitsize - 1:
            return
        else:
            if depth < 1:
                raise ValueError(f"{depth=} is not a positive integer")
            if depth < len(inp):
                yield CNOT().on(anc[depth - 1], inp[depth])
                control = inp[depth]
            else:
                # If inp[depth] doesn't exist, we treat it as a |0>,
                # and therefore applying CNOT().on(anc[depth - 1], inp[depth])
                # essentially "copies" anc[depth - 1] into inp[depth]
                # in the classical basis. So therefore, on future operations,
                # we can use anc[depth - 1] in its place.
                control = anc[depth - 1]
            yield CNOT().on(anc[depth - 1], out[depth])
            yield And().on(control, out[depth], anc[depth])
            yield CNOT().on(anc[depth - 1], anc[depth])
            yield from self._left_building_block(inp, out, anc, depth + 1)

    def _right_building_block(self, inp, out, anc, depth):
        if depth == 0:
            return
        else:
            yield CNOT().on(anc[depth - 1], anc[depth])
            if depth < len(inp):
                yield And().adjoint().on(inp[depth], out[depth], anc[depth])
                yield CNOT().on(anc[depth - 1], inp[depth])
                yield CNOT().on(inp[depth], out[depth])
            else:
                yield And().adjoint().on(anc[depth - 1], out[depth], anc[depth])
            yield from self._right_building_block(inp, out, anc, depth - 1)

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]  # type: ignore[type-var]
    ) -> Iterator[cirq.OP_TREE]:
        # reverse the order of qubits for big endian-ness.
        input_bits = quregs['a'][::-1]
        output_bits = quregs['b'][::-1]
        if self.b_dtype.bitsize == 1:
            yield CNOT().on(input_bits[0], output_bits[0])
            return
        ancillas = context.qubit_manager.qalloc(self.b_dtype.bitsize - 1)[::-1]
        # Start off the addition by anding into the ancilla
        yield And().on(input_bits[0], output_bits[0], ancillas[0])
        # Left part of Fig.2
        yield from self._left_building_block(input_bits, output_bits, ancillas, 1)
        yield CNOT().on(ancillas[-1], output_bits[-1])
        if len(input_bits) == len(output_bits):
            yield CNOT().on(input_bits[-1], output_bits[-1])
        # right part of Fig.2
        yield from self._right_building_block(
            input_bits, output_bits, ancillas, self.b_dtype.bitsize - 2
        )
        yield And().adjoint().on(input_bits[0], output_bits[0], ancillas[0])
        yield CNOT().on(input_bits[0], output_bits[0])
        context.qubit_manager.qfree(ancillas)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        n = self.b_dtype.bitsize
        n_cnot = (n - 2) * 6 + 3
        return {And(): n - 1, And().adjoint(): n - 1, CNOT(): n_cnot}

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> Tuple['Bloq', 'AddControlledT']:
        from qualtran.bloqs.arithmetic import CAdd

        return get_ctrl_system_1bit_cv(
            bloq=self,
            ctrl_spec=ctrl_spec,
            current_ctrl_bit=None,
            get_ctrl_bloq_and_ctrl_reg_name=lambda cv: (
                CAdd(a_dtype=self.a_dtype, b_dtype=self.b_dtype, cv=cv),
                "ctrl",
            ),
        )


@bloq_example(generalizer=ignore_split_join)
def _add_symb() -> Add:
    n = sympy.Symbol('n')
    add_symb = Add(QInt(bitsize=n))
    return add_symb


@bloq_example(generalizer=ignore_split_join)
def _add_small() -> Add:
    add_small = Add(QUInt(bitsize=4))
    return add_small


@bloq_example(generalizer=ignore_split_join)
def _add_large() -> Add:
    add_large = Add(QUInt(bitsize=64))
    return add_large


@bloq_example(generalizer=ignore_split_join)
def _add_diff_size_regs() -> Add:
    add_diff_size_regs = Add(QUInt(bitsize=4), QUInt(bitsize=16))
    return add_diff_size_regs


_ADD_DOC = BloqDocSpec(
    bloq_cls=Add, examples=[_add_symb, _add_small, _add_large, _add_diff_size_regs]
)


@frozen
class OutOfPlaceAdder(GateWithRegisters, cirq.ArithmeticGate):  # type: ignore[misc]
    r"""An n-bit addition gate.

    Implements $U|a\rangle|b\rangle 0\rangle \rightarrow |a\rangle|b\rangle|a+b\rangle$
    using $4n - 4 T$ gates. Uncomputation requires 0 T-gates.

    Args:
        bitsize: Number of bits used to represent each input integer. The allocated output register
            is of size `bitsize+1` so it has enough space to hold the sum of `a+b`.
        is_adjoint: Whether this is compute or uncompute version.
        include_most_significant_bit: Whether to add an extra most significant (i.e. carry) bit.

    Registers:
        a: A bitsize-sized input register (register a above).
        b: A bitsize-sized input register (register b above).
        c: The LEFT/RIGHT register depending on whether the gate adjoint or not.
            This register size is either bitsize or bitsize+1 depending on
            the value of `include_most_significant_bit`.

    References:
        [Halving the cost of quantum addition](https://arxiv.org/abs/1709.06648)
    """

    bitsize: 'SymbolicInt'
    is_adjoint: bool = False
    include_most_significant_bit: bool = True

    @property
    def out_bitsize(self):
        return self.bitsize + (1 if self.include_most_significant_bit else 0)

    @property
    def signature(self):
        side = Side.LEFT if self.is_adjoint else Side.RIGHT
        return Signature(
            [
                Register('a', QUInt(self.bitsize)),
                Register('b', QUInt(self.bitsize)),
                Register('c', QUInt(self.out_bitsize), side=side),
            ]
        )

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        if not isinstance(self.bitsize, int):
            raise ValueError(f'Symbolic bitsize {self.bitsize} not supported')
        return [2] * self.bitsize, [2] * self.bitsize, [2] * self.out_bitsize

    def apply(self, a: int, b: int, c: int) -> Tuple[int, int, int]:
        return a, b, c + a + b

    def adjoint(self) -> 'OutOfPlaceAdder':
        return evolve(self, is_adjoint=not self.is_adjoint)

    def on_classical_vals(
        self, *, a: 'ClassicalValT', b: 'ClassicalValT', c: Optional['ClassicalValT'] = None
    ) -> Dict[str, 'ClassicalValT']:
        if isinstance(self.bitsize, sympy.Expr):
            raise ValueError(f'Classical simulation is not support for symbolic bloq {self}')
        if self.is_adjoint:
            assert c is not None
            return {'a': a, 'b': b}
        assert c is None
        return {
            'a': a,
            'b': b,
            'c': add_ints(int(a), int(b), num_bits=self.out_bitsize, is_signed=False),
        }

    def with_registers(self, *new_registers: Union[int, Sequence[int]]):
        raise NotImplementedError("no need to implement with_registers.")

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        if not isinstance(self.bitsize, int):
            raise ValueError(f'Symbolic bitsize {self.bitsize} not supported')
        a, b, c = quregs['a'][::-1], quregs['b'][::-1], quregs['c'][::-1]
        optree: List[List[cirq.Operation]] = [
            [
                cirq.CX(a[i], b[i]),
                cirq.CX(a[i], c[i]),
                And().on(b[i], c[i], c[i + 1]),
                cirq.CX(a[i], b[i]),
                cirq.CX(a[i], c[i + 1]),
                cirq.CX(b[i], c[i]),
            ]
            for i in range(self.out_bitsize - 1)
        ]
        if not self.include_most_significant_bit:
            # Update c[-1] as c[-1] ^= a[-1]^b[-1]
            i = self.bitsize - 1
            optree.append([cirq.CX(a[i], c[i]), cirq.CX(b[i], c[i])])
        return cirq.inverse(optree) if self.is_adjoint else optree

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {
            And(uncompute=self.is_adjoint): self.out_bitsize - 1,
            CNOT(): 5 * (self.bitsize - 1) + 2 + (3 if self.include_most_significant_bit else 0),
        }

    def __pow__(self, power: int):
        if power == 1:
            return self
        if power == -1:
            return OutOfPlaceAdder(self.bitsize, is_adjoint=not self.is_adjoint)
        raise NotImplementedError("OutOfPlaceAdder.__pow__ defined only for +1/-1.")

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('c=a+b')
        return super().wire_symbol(reg, idx)


@bloq_example(generalizer=ignore_split_join)
def _add_oop_symb() -> OutOfPlaceAdder:
    n = sympy.Symbol('n')
    add_oop_symb = OutOfPlaceAdder(bitsize=n)
    return add_oop_symb


@bloq_example(generalizer=ignore_split_join)
def _add_oop_small() -> OutOfPlaceAdder:
    add_oop_small = OutOfPlaceAdder(bitsize=4)
    return add_oop_small


@bloq_example(generalizer=ignore_split_join)
def _add_oop_large() -> OutOfPlaceAdder:
    add_oop_large = OutOfPlaceAdder(bitsize=64)
    return add_oop_large


_ADD_OOP_DOC = BloqDocSpec(
    bloq_cls=OutOfPlaceAdder, examples=[_add_oop_symb, _add_oop_small, _add_oop_large]
)


def _cvs_converter(vv):
    if isinstance(vv, (int, np.integer)):
        return (int(vv),)
    return tuple(int(v) for v in vv)


@frozen
class AddK(Bloq):
    r"""Takes |x> to |x + k> for a classical integer `k`.

    This construction simply XORs the classical constant into a quantum register and
    applies quantum-quantum addition. This is the lowest T-count algorithm at the expense
    of $n$ auxiliary qubits. This construction also permits an inexpensive controlled version:
    you only need to control the loading of the classical constant which can be done with
    only clifford operations.

    Args:
        dtype: data type of the input register `x`
        k: The classical integer value to be added to x.

    Registers:
        x: register of type `self.dtype`

    References:
        [Improved quantum circuits for elliptic curve discrete logarithms](https://arxiv.org/abs/2001.09580).
        Haner et al. 2020. Section 3: Components. "Integer addition" and Fig 2a.
    """

    dtype: Union[QInt, QUInt, QMontgomeryUInt]
    k: 'SymbolicInt'

    def __attrs_post_init__(self):
        if not isinstance(self.dtype, (QInt, QUInt, QMontgomeryUInt)):
            raise NotImplementedError(
                "Only QInt, QUInt and QMontgomeryUInt types are supported for composite addition."
            )

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(x=self.dtype)

    def on_classical_vals(
        self, x: 'ClassicalValT', **vals: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        if is_symbolic(self.k) or is_symbolic(self.dtype):
            raise ValueError(f"Classical simulation isn't supported for symbolic block {self}")

        is_signed = isinstance(self.dtype, QInt)
        x = add_ints(int(x), int(self.k), num_bits=self.dtype.num_qubits, is_signed=is_signed)

        return {'x': x}

    @cached_property
    def _load_k_bloq(self) -> Bloq:
        from qualtran.bloqs.arithmetic.bitwise import XorK

        k = self.k
        if not is_symbolic(k) and k < 0 and isinstance(self.dtype, (QUInt, QMontgomeryUInt)):
            # Since this is unsigned addition, adding `-v` is equivalent to adding `2**bitsize - v`
            k %= 2**self.dtype.bitsize

        return XorK(self.dtype, k)

    def build_composite_bloq(self, bb: 'BloqBuilder', x: Soquet) -> Dict[str, 'SoquetT']:
        if is_symbolic(self.k) or is_symbolic(self.dtype):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self}.")

        # load `k`
        k = bb.allocate(dtype=self.dtype)
        k = bb.add(self._load_k_bloq, x=k)

        # perform the quantum-quantum addition
        # we always perform this addition (even when controlled), so we wrap in `Always`
        # controlling the data loading is sufficient to control this bloq.
        k, x = bb.add(Always(Add(self.dtype, self.dtype)), a=k, b=x)

        # unload `k`
        k = bb.add(self._load_k_bloq.adjoint(), x=k)
        bb.free(k)

        return {'x': x}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        counts = Counter[Bloq]()

        counts[self._load_k_bloq] += 1
        counts[Always(Add(self.dtype, self.dtype))] += 1
        counts[self._load_k_bloq.adjoint()] += 1

        return counts


@bloq_example(generalizer=ignore_split_join)
def _add_k() -> AddK:
    n, k = sympy.symbols('n k')
    add_k = AddK(QUInt(n), k=k)
    return add_k


@bloq_example(generalizer=ignore_split_join)
def _add_k_small() -> AddK:
    add_k_small = AddK(QUInt(4), k=2)
    return add_k_small


@bloq_example(generalizer=ignore_split_join)
def _add_k_large() -> AddK:
    add_k_large = AddK(QInt(64), k=-23)
    return add_k_large


_ADD_K_DOC = BloqDocSpec(bloq_cls=AddK, examples=[_add_k, _add_k_small, _add_k_large])
