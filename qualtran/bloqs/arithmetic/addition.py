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
import itertools
import math
from functools import cached_property
from typing import Any, Dict, Sequence, Set, Tuple, TYPE_CHECKING, Union

import cirq
import numpy as np
import sympy
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    CompositeBloq,
    GateWithRegisters,
    QBit,
    QInt,
    QUInt,
    Register,
    Side,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran._infra.data_types import QMontgomeryUInt
from qualtran.bloqs.basic_gates import CNOT, XGate
from qualtran.bloqs.mcmt.and_bloq import And
from qualtran.bloqs.mcmt.multi_control_multi_target_pauli import MultiControlX
from qualtran.bloqs.util_bloqs import ArbitraryClifford
from qualtran.cirq_interop import decompose_from_cirq_style_method
from qualtran.cirq_interop.bit_tools import iter_bits, iter_bits_twos_complement
from qualtran.cirq_interop.t_complexity_protocol import TComplexity

if TYPE_CHECKING:
    import quimb.tensor as qtn

    from qualtran.drawing import WireSymbol
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
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

    def add_my_tensors(
        self,
        tn: 'qtn.TensorNetwork',
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        import quimb.tensor as qtn

        if isinstance(self.a_dtype, QInt) or isinstance(self.b_dtype, QInt):
            raise TypeError("Tensor contraction for addition is only supported for unsigned ints.")
        N_a = 2**self.a_dtype.bitsize
        N_b = 2**self.b_dtype.bitsize
        inds = (incoming['a'], incoming['b'], outgoing['a'], outgoing['b'])
        unitary = np.zeros((N_a, N_b, N_a, N_b), dtype=np.complex128)
        # TODO: Add a value-to-index method on dtype to make this easier.
        for a, b in itertools.product(range(N_a), range(N_b)):
            unitary[a, b, a, int(math.fmod(a + b, N_b))] = 1

        tn.add(qtn.Tensor(data=unitary, inds=inds, tags=[self.short_name(), tag]))

    def decompose_bloq(self) -> 'CompositeBloq':
        return decompose_from_cirq_style_method(self)

    def on_classical_vals(
        self, a: 'ClassicalValT', b: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        unsigned = isinstance(self.a_dtype, (QUInt, QMontgomeryUInt))
        b_bitsize = self.b_dtype.bitsize
        N = 2**b_bitsize if unsigned else 2 ** (b_bitsize - 1)
        return {'a': a, 'b': int(math.fmod(a + b, N))}

    def short_name(self) -> str:
        return "a+b"

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["In(x)"] * int(self.a_dtype.bitsize)
        wire_symbols += ["In(y)/Out(x+y)"] * int(self.b_dtype.bitsize)
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        from qualtran.drawing import directional_text_box

        if soq.reg.name == 'a':
            return directional_text_box('a', side=soq.reg.side)
        elif soq.reg.name == 'b':
            return directional_text_box('a+b', side=soq.reg.side)
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
    ) -> cirq.OP_TREE:
        # reverse the order of qubits for big endian-ness.
        input_bits = quregs['a'][::-1]
        output_bits = quregs['b'][::-1]
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

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        n = self.b_dtype.bitsize
        n_cnot = (n - 2) * 6 + 3
        return {(And(), n - 1), (And().adjoint(), n - 1), (CNOT(), n_cnot)}


@bloq_example
def _add_symb() -> Add:
    n = sympy.Symbol('n')
    add_symb = Add(QInt(bitsize=n))
    return add_symb


@bloq_example
def _add_small() -> Add:
    add_small = Add(QUInt(bitsize=4))
    return add_small


@bloq_example
def _add_large() -> Add:
    add_large = Add(QUInt(bitsize=64))
    return add_large


@bloq_example
def _add_diff_size_regs() -> Add:
    add_diff_size_regs = Add(QUInt(bitsize=4), QUInt(bitsize=16))
    return add_diff_size_regs


_ADD_DOC = BloqDocSpec(
    bloq_cls=Add, examples=[_add_symb, _add_small, _add_large, _add_diff_size_regs]
)


@frozen
class OutOfPlaceAdder(GateWithRegisters, cirq.ArithmeticGate):
    r"""An n-bit addition gate.

    Implements $U|a\rangle|b\rangle 0\rangle \rightarrow |a\rangle|b\rangle|a+b\rangle$
    using $4n - 4 T$ gates. Uncomputation requires 0 T-gates.

    Args:
        bitsize: Number of bits used to represent each input integer. The allocated output register
            is of size `bitsize+1` so it has enough space to hold the sum of `a+b`.

    Registers:
        a: A bitsize-sized input register (register a above).
        b: A bitsize-sized input register (register b above).
        c: A bitize+1-sized LEFT/RIGHT register depending on whether the gate adjoint or not.

    References:
        [Halving the cost of quantum addition](https://arxiv.org/abs/1709.06648)
    """

    bitsize: int
    adjoint: bool = False

    @property
    def signature(self):
        side = Side.LEFT if self.adjoint else Side.RIGHT
        return Signature(
            [
                Register('a', QUInt(self.bitsize)),
                Register('b', QUInt(self.bitsize)),
                Register('c', QUInt(self.bitsize + 1), side=side),
            ]
        )

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return [2] * self.bitsize, [2] * self.bitsize, [2] * (self.bitsize + 1)

    def apply(self, a: int, b: int, c: int) -> Tuple[int, int, int]:
        return a, b, c + a + b

    def on_classical_vals(
        self, *, a: 'ClassicalValT', b: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        return {'a': a, 'b': b, 'c': a + b}

    def with_registers(self, *new_registers: Union[int, Sequence[int]]):
        raise NotImplementedError("no need to implement with_registers.")

    def short_name(self) -> str:
        return "c = a + b"

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        a, b, c = quregs['a'][::-1], quregs['b'][::-1], quregs['c'][::-1]
        optree = [
            [
                [cirq.CX(a[i], b[i]), cirq.CX(a[i], c[i])],
                And().on(b[i], c[i], c[i + 1]),
                [cirq.CX(a[i], b[i]), cirq.CX(a[i], c[i + 1]), cirq.CX(b[i], c[i])],
            ]
            for i in range(self.bitsize)
        ]
        return cirq.inverse(optree) if self.adjoint else optree

    def _t_complexity_(self) -> TComplexity:
        and_t = And(uncompute=self.adjoint).t_complexity()
        num_clifford = self.bitsize * (5 + and_t.clifford)
        num_t = self.bitsize * and_t.t
        return TComplexity(t=num_t, clifford=num_clifford)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {
            (And(uncompute=self.adjoint), self.bitsize),
            (ArbitraryClifford(n=2), 5 * self.bitsize),
        }

    def __pow__(self, power: int):
        if power == 1:
            return self
        if power == -1:
            return OutOfPlaceAdder(self.bitsize, adjoint=not self.adjoint)
        raise NotImplementedError("OutOfPlaceAdder.__pow__ defined only for +1/-1.")


@bloq_example
def _add_oop_symb() -> OutOfPlaceAdder:
    n = sympy.Symbol('n')
    add_oop_symb = OutOfPlaceAdder(bitsize=n)
    return add_oop_symb


@bloq_example
def _add_oop_small() -> OutOfPlaceAdder:
    add_oop_small = OutOfPlaceAdder(bitsize=4)
    return add_oop_small


@bloq_example
def _add_oop_large() -> OutOfPlaceAdder:
    add_oop_large = OutOfPlaceAdder(bitsize=64)
    return add_oop_large


_ADD_OOP_DOC = BloqDocSpec(
    bloq_cls=OutOfPlaceAdder, examples=[_add_oop_symb, _add_oop_small, _add_oop_large]
)


@frozen
class AddK(Bloq):
    r"""Takes |x> to |x + k> for a classical integer `k`.

    This construction simply XORs the classical constant into a quantum register and
    applies quantum-quantum addition. This is the lowest T-count algorithm at the expense
    of $n$ auxiliary qubits. This construction also permits an inexpensive controlled version:
    you only need to control the loading of the classical constant which can be done with
    only clifford operations.

    Args:
        bitsize: Number of bits used to represent each integer.
        k: The classical integer value to be added to x.
        cvs: A tuple of control values. Each entry specifies whether that control line is a
            "positive" control (`cv[i]=1`) or a "negative" control (`cv[i]=0`).
        signed: A boolean condition which controls whether the x register holds a value represented
            in 2's Complement or Unsigned. This affects the ability to add a negative constant.

    Registers:
        x: A bitsize-sized input register (register x above).

    References:
        [Improved quantum circuits for elliptic curve discrete logarithms](https://arxiv.org/abs/2001.09580).
        Haner et. al. 2020. Section 3: Components. "Integer addition" and Fig 2a.
    """

    bitsize: int
    k: int
    cvs: Tuple[int, ...] = field(
        converter=lambda v: (v,) if isinstance(v, int) else tuple(v), default=()
    )
    signed: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        if len(self.cvs) > 0:
            return Signature(
                [
                    Register('ctrls', QBit(), shape=(len(self.cvs),)),
                    Register('x', QInt(self.bitsize) if self.signed else QUInt(self.bitsize)),
                ]
            )
        else:
            return Signature(
                [Register('x', QInt(bitsize=self.bitsize) if self.signed else QUInt(self.bitsize))]
            )

    def on_classical_vals(
        self, x: 'ClassicalValT', **vals: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        if len(self.cvs) > 0:
            ctrls = vals['ctrls']
        else:
            return {'x': x + self.k}

        if (self.cvs == ctrls).all():
            x = x + self.k

        return {'ctrls': ctrls, 'x': x}

    def build_composite_bloq(
        self, bb: 'BloqBuilder', x: SoquetT, **regs: SoquetT
    ) -> Dict[str, 'SoquetT']:
        # Assign registers to variables and allocate ancilla bits for classical integer k.
        if len(self.cvs) > 0:
            ctrls = regs['ctrls']
        else:
            ctrls = None
        k = bb.allocate(dtype=x.reg.dtype)

        # Get binary representation of k and split k into separate wires.
        k_split = bb.split(k)
        if self.signed:
            binary_rep = list(iter_bits_twos_complement(self.k, self.bitsize))
        else:
            binary_rep = list(iter_bits(self.k, self.bitsize))

        # Apply XGates to qubits in k where the bitstring has value 1. Apply CNOTs when the gate is
        # controlled.
        for i in range(self.bitsize):
            if binary_rep[i] == 1:
                if len(self.cvs) > 0:
                    ctrls, k_split[i] = bb.add(
                        MultiControlX(cvs=self.cvs), ctrls=ctrls, x=k_split[i]
                    )
                else:
                    k_split[i] = bb.add(XGate(), q=k_split[i])

        # Rejoin the qubits representing k for in-place addition.
        k = bb.join(k_split, dtype=x.reg.dtype)
        k, x = bb.add(Add(x.reg.dtype, x.reg.dtype), a=k, b=x)

        # Resplit the k qubits in order to undo the original bit flips to go from the binary
        # representation back to the zero state.
        k_split = bb.split(k)
        for i in range(self.bitsize):
            if binary_rep[i] == 1:
                if len(self.cvs) > 0:
                    ctrls, k_split[i] = bb.add(
                        MultiControlX(cvs=self.cvs), ctrls=ctrls, x=k_split[i]
                    )
                else:
                    k_split[i] = bb.add(XGate(), q=k_split[i])

        # Free the ancilla qubits.
        k = bb.join(k_split, dtype=x.reg.dtype)
        bb.free(k)

        # Return the output registers.
        if len(self.cvs) > 0:
            return {'ctrls': ctrls, 'x': x}
        else:
            return {'x': x}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        if len(self.cvs) == 0:
            loading_cost = (XGate(), self.bitsize)  # upper bound; depends on the data.
        elif len(self.cvs) == 1:
            loading_cost = (CNOT(), self.bitsize)  # upper bound; depends on the data.
        else:
            # Otherwise, use the decomposition
            return super().build_call_graph(ssa=ssa)

        return {loading_cost, (Add(QUInt(self.bitsize)), 1)}

    def short_name(self) -> str:
        return f'x += {self.k}'


@bloq_example
def _add_k() -> AddK:
    n, k = sympy.symbols('n k')
    add_k = AddK(bitsize=n, k=k)
    return add_k


@bloq_example
def _add_k_small() -> AddK:
    add_k_small = AddK(bitsize=4, k=2, signed=False)
    return add_k_small


@bloq_example
def _add_k_large() -> AddK:
    add_k_large = AddK(bitsize=64, k=-23, signed=True)
    return add_k_large


_ADD_K_DOC = BloqDocSpec(bloq_cls=AddK, examples=[_add_k, _add_k_small, _add_k_large])
