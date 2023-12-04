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
from typing import Dict, Iterable, Sequence, Set, Tuple, TYPE_CHECKING, Union

import cirq
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import Bloq, BloqBuilder, GateWithRegisters, Register, Side, Signature, SoquetT
from qualtran.bloqs.and_bloq import And
from qualtran.bloqs.basic_gates import Toffoli, XGate
from qualtran.bloqs.multi_control_multi_target_pauli import MultiControlPauli
from qualtran.bloqs.util_bloqs import ArbitraryClifford
from qualtran.cirq_interop.bit_tools import iter_bits, iter_bits_twos_complement
from qualtran.cirq_interop.t_complexity_protocol import TComplexity

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class Add(GateWithRegisters, cirq.ArithmeticGate):
    r"""An n-bit addition gate.

    Implements $U|a\rangle|b\rangle \rightarrow |a\rangle|a+b\rangle$ using $4n - 4 T$ gates.

    Args:
        bitsize: Number of bits used to represent each integer. Must be large
            enough to hold the result in the output register of a + b, or else it simply
            drops the most significant bits.

    Registers:
        a: A bitsize-sized input register (register a above).
        b: A bitsize-sized input/output register (register b above).

    References:
        [Halving the cost of quantum addition](https://arxiv.org/abs/1709.06648)
    """

    bitsize: int

    @property
    def signature(self):
        return Signature.build(a=self.bitsize, b=self.bitsize)

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return [2] * self.bitsize, [2] * self.bitsize

    def with_registers(self, *new_registers) -> 'Add':
        return Add(len(new_registers[0]))

    def apply(self, *register_values: int) -> Union[int, Iterable[int]]:
        a, b = register_values
        return a, a + b

    def on_classical_vals(self, *args) -> Dict[str, 'ClassicalValT']:
        return dict(zip([reg.name for reg in self.signature], self.apply(*args)))

    def short_name(self) -> str:
        return "a+b"

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["In(x)"] * self.bitsize
        wire_symbols += ["In(y)/Out(x+y)"] * self.bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _has_unitary_(self):
        return True

    def _left_building_block(self, inp, out, anc, depth):
        if depth == self.bitsize - 1:
            return
        else:
            yield cirq.CX(anc[depth - 1], inp[depth])
            yield cirq.CX(anc[depth - 1], out[depth])
            yield And().on(inp[depth], out[depth], anc[depth])
            yield cirq.CX(anc[depth - 1], anc[depth])
            yield from self._left_building_block(inp, out, anc, depth + 1)

    def _right_building_block(self, inp, out, anc, depth):
        if depth == 0:
            return
        else:
            yield cirq.CX(anc[depth - 1], anc[depth])
            yield And(adjoint=True).on(inp[depth], out[depth], anc[depth])
            yield cirq.CX(anc[depth - 1], inp[depth])
            yield cirq.CX(inp[depth], out[depth])
            yield from self._right_building_block(inp, out, anc, depth - 1)

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        input_bits = quregs['a']
        output_bits = quregs['b']
        ancillas = context.qubit_manager.qalloc(self.bitsize - 1)
        # Start off the addition by anding into the ancilla
        yield And().on(input_bits[0], output_bits[0], ancillas[0])
        # Left part of Fig.2
        yield from self._left_building_block(input_bits, output_bits, ancillas, 1)
        yield cirq.CX(ancillas[-1], output_bits[-1])
        yield cirq.CX(input_bits[-1], output_bits[-1])
        # right part of Fig.2
        yield from self._right_building_block(input_bits, output_bits, ancillas, self.bitsize - 2)
        yield And(adjoint=True).on(input_bits[0], output_bits[0], ancillas[0])
        yield cirq.CX(input_bits[0], output_bits[0])
        context.qubit_manager.qfree(ancillas)

    def _t_complexity_(self):
        num_clifford = (self.bitsize - 2) * 19 + 16
        num_t_gates = 4 * self.bitsize - 4
        return TComplexity(t=num_t_gates, clifford=num_clifford)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        num_clifford = (self.bitsize - 2) * 19 + 16
        num_toffoli = self.bitsize - 1
        return {(Toffoli(), num_toffoli), (ArbitraryClifford(n=1), num_clifford)}


@frozen
class OutOfPlaceAdder(GateWithRegisters, cirq.ArithmeticGate):
    r"""An n-bit addition gate.

    Implements $U|a\rangle|b\rangle 0\rangle \rightarrow |a\rangle|b\rangle|a+b\rangle$
    using $4n - 4 T$ gates. Uncomputation requires 0 T-gates.

    Args:
        bitsize: Number of bits used to represent each input integer. The allocated output register
            is of size `bitsize+1` so it has enough space to hold the sum of `a+b`.

    Registers:
     - a: A bitsize-sized input register (register a above).
     - b: A bitsize-sized input register (register b above).
     - c: A bitize+1-sized LEFT/RIGHT register depending on whether the gate adjoint or not.

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
                Register('a', self.bitsize),
                Register('b', self.bitsize),
                Register('c', self.bitsize + 1, side=side),
            ]
        )

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return [2] * self.bitsize, [2] * self.bitsize, [2] * (self.bitsize + 1)

    def apply(self, a: int, b: int, c: int) -> Tuple[int, int, int]:
        return a, b, c + a + b

    def on_classical_vals(
        self, a: 'ClassicalValT', b: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        return dict(zip('abc', (a, b, a + b)))

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

    def t_complexity(self) -> TComplexity:
        and_t = And(adjoint=self.adjoint).t_complexity()
        num_clifford = self.bitsize * (5 + and_t.clifford)
        num_t = self.bitsize * and_t.t
        return TComplexity(t=num_t, clifford=num_clifford)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {
            (And(adjoint=self.adjoint), self.bitsize),
            (ArbitraryClifford(n=2), 5 * self.bitsize),
        }

    def __pow__(self, power: int):
        if power == 1:
            return self
        if power == -1:
            return OutOfPlaceAdder(self.bitsize, adjoint=not self.adjoint)
        raise NotImplementedError("OutOfPlaceAdder.__pow__ defined only for +1/-1.")


@frozen
class SimpleAddConstant(Bloq):
    r"""Applies U_{k}|x> = |x + k>.

    Applies addition to input register `|x>` given classical integer 'k'.

    This is the simple version of constant addition because it involves simply converting the
    classical integer into a quantum parameter and using quantum-quantum addition as opposed to
    designing a bespoke circuit for constant addition based on the classical parameter.

    Args:
        bitsize: Number of bits used to represent each integer.
        k: The classical integer value to be added to x.
        cvs: A tuple of control variable settings. Each entry specifies whether that
            control line is a "positive" control (`cv[i]=1`) or a "negative" control `0`.
        signed: A boolean condition which controls whether the x register holds a value represented
            in 2's Complement or Unsigned. This affects the ability to add a negative constant.

    Registers:
        x: A bitsize-sized input register (register x above).

    References:
        [Improved quantum circuits for elliptic curve discrete logarithms](https://arxiv.org/abs/2001.09580) Fig 2a
    """

    bitsize: int
    k: int
    cvs: Tuple[int, ...] = field(converter=lambda v: (v,) if isinstance(v, int) else tuple(v))
    signed: bool

    @cached_property
    def signature(self) -> 'Signature':
        if len(self.cvs) > 0:
            return Signature(
                [Register('ctrl', bitsize=len(self.cvs)), Register('x', bitsize=self.bitsize)]
            )
        else:
            return Signature([Register('x', bitsize=self.bitsize)])

    def apply(self, *args) -> Union[int, Iterable[int]]:
        target_val = args[-1]
        new_target_val = target_val + self.k

        if self.cvs and args[0] != int(''.join(str(x) for x in self.cvs), 2):
            new_target_val = target_val
        ret = (args[0], new_target_val) if self.cvs else (new_target_val,)
        return ret

    def on_classical_vals(self, *args) -> Dict[str, 'ClassicalValT']:
        return dict(zip([reg.name for reg in self.signature], self.apply(*args)))

    def build_composite_bloq(self, bb: 'BloqBuilder', **regs: SoquetT) -> Dict[str, 'SoquetT']:

        # Assign registers to variables and allocate ancilla bits for classical integer k.
        x = regs['x']
        if len(self.cvs) > 0:
            ctrls = regs['ctrl']
        else:
            ctrls = None
        k = bb.allocate(n=self.bitsize)

        # Get binary representation of k and split k into separate wires.
        k_split = bb.split(k)
        if self.signed:
            binary_rep = list(iter_bits_twos_complement(self.k, self.bitsize))
        else:
            binary_rep = list(iter_bits(self.k, self.bitsize))
        binary_rep.reverse()

        # Apply XGates to qubits in k where the bitstring has value 1. Apply CNOTs when the gate is
        # controlled.
        for i in range(self.bitsize):
            if binary_rep[i] == 1:
                if len(self.cvs) > 0:
                    ctrls, k_split[i] = bb.add(
                        MultiControlPauli(cvs=self.cvs, target_gate=XGate()),
                        controls=ctrls,
                        target=k_split[i],
                    )
                else:
                    k_split[i] = bb.add(XGate(), q=k_split[i])

        # Rejoin the qubits representing k for in-place addition.
        k = bb.join(k_split)
        k, x = bb.add(Add(bitsize=self.bitsize), a=k, b=x)

        # Resplit the k qubits in order to undo the original bit flips to go from the binary
        # representation back to the zero state.
        k_split = bb.split(k)
        for i in range(self.bitsize):
            if binary_rep[i] == 1:
                if len(self.cvs) > 0:
                    ctrls, k_split[i] = bb.add(
                        MultiControlPauli(cvs=self.cvs, target_gate=XGate()),
                        controls=ctrls,
                        target=k_split[i],
                    )
                else:
                    k_split[i] = bb.add(XGate(), q=k_split[i])

        # Free the ancilla qubits.
        k = bb.join(k_split)
        bb.free(k)

        # Return the output registers.
        if len(self.cvs) > 0:
            return {'x': x, 'ctrl': ctrls}
        else:
            return {'x': x}

    def short_name(self) -> str:
        return f'x = x + {self.k}'


@frozen(auto_attribs=True)
class AddConstantMod(GateWithRegisters, cirq.ArithmeticGate):
    """Applies U_{M}_{add}|x> = |(x + add) % M> if x < M else |x>.

    Applies modular addition to input register `|x>` given parameters `mod` and `add_val` s.t.
        1) If integer `x` < `mod`: output is `|(x + add) % M>`
        2) If integer `x` >= `mod`: output is `|x>`.

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
        if not 1 <= value <= 2**self.bitsize:
            raise ValueError(f"mod: {value} must be between [1, {2 ** self.bitsize}].")

    @cached_property
    def signature(self) -> Signature:
        if self.cvs:
            return Signature.build(ctrl=len(self.cvs), x=self.bitsize)
        return Signature.build(x=self.bitsize)

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        add_reg = (2,) * self.bitsize
        control_reg = (2,) * len(self.cvs)
        return (control_reg, add_reg) if control_reg else (add_reg,)

    def with_registers(self, *new_registers: Union[int, Sequence[int]]) -> "AddMod":
        raise NotImplementedError()

    def apply(self, *args) -> Union[int, Iterable[int]]:
        target_val = args[-1]
        if target_val < self.mod:
            new_target_val = (target_val + self.add_val) % self.mod
        else:
            new_target_val = target_val
        if self.cvs and args[0] != int(''.join(str(x) for x in self.cvs), 2):
            new_target_val = target_val
        ret = (args[0], new_target_val) if self.cvs else (new_target_val,)
        return ret

    def on_classical_vals(self, *args) -> Dict[str, 'ClassicalValT']:
        return dict(zip([reg.name for reg in self.signature], self.apply(*args)))

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ['@' if b else '@(0)' for b in self.cvs]
        wire_symbols += [f"Add_{self.add_val}_Mod_{self.mod}"] * self.bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __pow__(self, power: int) -> 'AddConstantMod':
        return AddConstantMod(self.bitsize, self.mod, add_val=self.add_val * power, cvs=self.cvs)

    def _t_complexity_(self) -> TComplexity:
        # Rough cost as given in https://arxiv.org/abs/1905.09749
        return 5 * Add(self.bitsize).t_complexity()
