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
from typing import Any, Dict, Set, TYPE_CHECKING, Union

import cirq
import numpy as np
import sympy
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import Bloq, CompositeBloq, QBit, QInt, QUInt, Register, Signature, Soquet, SoquetT
from qualtran._infra.data_types import QMontgomeryUInt
from qualtran.bloqs.basic_gates import CNOT
from qualtran.bloqs.mcmt.and_bloq import And
from qualtran.bloqs.mcmt.multi_control_multi_target_pauli import MultiControlPauli
from qualtran.cirq_interop import decompose_from_cirq_style_method
from qualtran.cirq_interop.t_complexity_protocol import TComplexity

if TYPE_CHECKING:
    import quimb.tensor as qtn

    from qualtran.drawing import WireSymbol
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class ControlledAdd(Bloq):
    r"""An n-bit controlled-addition gate.

    Args:
        a_dtype: Quantum datatype used to represent the integer a.
        b_dtype: Quantum datatype used to represent the integer b. Must be large
            enough to hold the result in the output register of a + b, or else it simply
            drops the most significant bits. If not specified, b_dtype is set to a_dtype.
        controlled: When controlled=0, this bloq is active when the ctrl register is 0. When
            controlled=1, this bloq is active when the ctrl register is 1.

    Registers:
        ctrl: the control bit for the addition
        a: A a_dtype.bitsize-sized input register (register a above).
        b: A b_dtype.bitsize-sized input/output register (register b above).

    References:
        [Halving the cost of quantum addition](https://arxiv.org/abs/1709.06648)
    """

    a_dtype: Union[QInt, QUInt, QMontgomeryUInt] = field()
    b_dtype: Union[QInt, QUInt, QMontgomeryUInt] = field()
    controlled: int = field(default=1)

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

    @controlled.validator
    def _controlled_validate(self, field, val):
        if val not in (0, 1):
            raise ValueError("controlled must be either 0 or 1")

    @property
    def signature(self):
        return Signature(
            [Register("ctrl", QBit()), Register("a", self.a_dtype), Register("b", self.b_dtype)]
        )

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
        inds = (
            incoming['ctrl'],
            incoming['a'],
            incoming['b'],
            outgoing['ctrl'],
            outgoing['a'],
            outgoing['b'],
        )
        unitary = np.zeros((2, N_a, N_b, 2, N_a, N_b), dtype=np.complex128)
        for c, a, b in itertools.product(range(2), range(N_a), range(N_b)):
            if c == self.controlled:
                unitary[c, a, b, c, a, int(math.fmod(a + b, N_b))] = 1
            else:
                unitary[c, a, b, c, a, b] = 1

        tn.add(qtn.Tensor(data=unitary, inds=inds, tags=[self.short_name(), tag]))

    def decompose_bloq(self) -> 'CompositeBloq':
        return decompose_from_cirq_style_method(self)

    def on_classical_vals(self, **kwargs) -> Dict[str, 'ClassicalValT']:
        a, b = kwargs['a'], kwargs['b']
        unsigned = isinstance(self.a_dtype, (QUInt, QMontgomeryUInt))
        b_bitsize = self.b_dtype.bitsize
        N = 2**b_bitsize if unsigned else 2 ** (b_bitsize - 1)
        ctrl = kwargs['ctrl']
        if ctrl != self.controlled:
            return {'ctrl': ctrl, 'a': a, 'b': b}
        else:
            return {'ctrl': ctrl, 'a': a, 'b': int(math.fmod(a + b, N))}

    def short_name(self) -> str:
        return "a+b"

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        if self.controlled is not None:
            wire_symbols = ["In(ctrl)"]
        else:
            wire_symbols = []
        wire_symbols += ["In(x)"] * self.a_dtype.bitsize
        wire_symbols += ["In(y)/Out(x+y)"] * self.b_dtype.bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        from qualtran.drawing import directional_text_box

        if soq.reg.name == 'ctrl':
            return directional_text_box('ctrl', side=soq.reg.side)
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

    def _right_building_block(self, inp, out, anc, control, depth):
        if depth == 0:
            return
        yield CNOT().on(anc[depth - 1], anc[depth])
        if depth < len(inp):
            yield And().adjoint().on(inp[depth], out[depth], anc[depth])
            yield MultiControlPauli((1, 1), cirq.X).on(control, inp[depth], out[depth])
            yield CNOT().on(anc[depth - 1], inp[depth])
        else:
            yield And().adjoint().on(anc[depth - 1], out[depth], anc[depth])
            yield MultiControlPauli((1, 1), cirq.X).on(control, anc[depth - 1], out[depth])
        yield CNOT().on(anc[depth - 1], out[depth])
        yield from self._right_building_block(inp, out, anc, control, depth - 1)

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        # reverse the order of qubits for big endian-ness.
        input_bits = quregs['a'][::-1]
        output_bits = quregs['b'][::-1]
        ancillas = context.qubit_manager.qalloc(self.b_dtype.bitsize - 1)[::-1]
        control = quregs['ctrl'][0]
        if self.controlled == 0:
            yield cirq.X(control)
        # Start off the addition by anding into the ancilla
        yield And().on(input_bits[0], output_bits[0], ancillas[0])
        # Left part of Fig.4
        yield from self._left_building_block(input_bits, output_bits, ancillas, 1)
        yield CNOT().on(ancillas[-1], output_bits[-1])
        if len(input_bits) == len(output_bits):
            yield MultiControlPauli((1, 1), cirq.X).on(control, input_bits[-1], output_bits[-1])
            yield CNOT().on(ancillas[-1], output_bits[-1])
        # right part of Fig.4
        yield from self._right_building_block(
            input_bits, output_bits, ancillas, control, self.b_dtype.bitsize - 2
        )
        yield And().adjoint().on(input_bits[0], output_bits[0], ancillas[0])
        yield MultiControlPauli((1, 1), cirq.X).on(control, input_bits[0], output_bits[0])
        if self.controlled == 0:
            yield cirq.X(control)
        context.qubit_manager.qfree(ancillas)

    def _t_complexity_(self):
        n = self.b_dtype.bitsize
        num_toffoli = n - 1
        num_clifford = 33 * (n - 2) + 43
        return TComplexity(t=8 * num_toffoli + 4, clifford=num_clifford)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        n = self.b_dtype.bitsize
        n_cnot = (n - 2) * 6 + 2
        return {
            (MultiControlPauli((1, 1), cirq.X), n),
            (And(), n - 1),
            (And().adjoint(), n - 1),
            (CNOT(), n_cnot),
        }
