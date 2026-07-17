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

"""Classes to apply single qubit bloq to multiple qubits."""

from functools import cached_property
from typing import cast, Dict, Optional, Tuple, TYPE_CHECKING

import attrs
import sympy

from qualtran import (
    Bloq,
    BloqBuilder,
    DecomposeTypeError,
    QAny,
    QDType,
    QVar,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.drawing import Text, WireSymbol
from qualtran.drawing.musical_score import TextBox
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.symbolics import SymbolicInt

if TYPE_CHECKING:
    from qualtran.simulation.classical_sim import ClassicalValT
    from qualtran.simulation.verification import ClassicalSimTestCase


@attrs.frozen
class OnEach(Bloq):
    """Add a single-qubit (unparameterized) bloq on each of n qubits.

    Args:
        n: the number of qubits to add the bloq to.
        gate: A single qubit gate. The single qubit register must be named q.
        target_dtype: optional dtype of the register. If not provided, default to QAny.

    Registers:
     - q: an n-qubit register.
    """

    n: SymbolicInt
    gate: Bloq
    target_dtype: Optional[QDType] = None

    def __attrs_post_init__(self):
        assert len(self.gate.signature) == 1, "Gate must only have a single register."
        assert self.gate.signature[0].bitsize == 1, "Must be single qubit gate."
        assert self.gate.signature[0].name == 'q', "Register must be named q."
        assert (
            self.target_dtype is None
            or not hasattr(self.target_dtype, 'bitsize')
            or self.target_dtype.bitsize == self.n
        )

    @cached_property
    def signature(self) -> Signature:
        reg = Register(
            'q', QAny(bitsize=self.n) if self.target_dtype is None else self.target_dtype
        )
        return Signature([reg])

    @classmethod
    def qcall(cls, q: 'QVar', *, gate: Bloq) -> 'QVar':
        return q.bb.add(cls(n=q.dtype.num_qubits, gate=gate, target_dtype=q.dtype), q=q)  # type: ignore[arg-type]

    def build_composite_bloq(self, bb: BloqBuilder, *, q: Soquet) -> Dict[str, SoquetT]:
        if isinstance(self.n, sympy.Expr):
            raise DecomposeTypeError(f'Cannote decompose {self} with symbolic bitsize {self.n}')
        qs = bb.split(q)
        for i in range(self.n):
            qs[i] = bb.add(self.gate, q=qs[i])
        return {'q': bb.join(qs, self.target_dtype)}

    def on_classical_vals(self, q: int) -> Dict[str, 'ClassicalValT']:
        n = self.n
        if isinstance(n, sympy.Expr):
            raise ValueError(f'Cannot simulate symbolic bloq {self}')
        dtype = self.signature[0].dtype
        bits = dtype.to_bits(q)
        out_bits = list(bits)
        for i in range(n):
            out = self.gate.on_classical_vals(q=bits[i])
            if out is NotImplemented:
                return NotImplemented
            out_bits[i] = int(cast(int, out['q']))
        return {'q': dtype.from_bits(out_bits)}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {self.gate: self.n}

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> WireSymbol:
        one_reg = self.gate.wire_symbol(reg=reg, idx=idx)
        if isinstance(one_reg, TextBox):
            new_text = f'{one_reg.text}⨂{self.n}'
            return TextBox(new_text)
        if isinstance(one_reg, Text):
            if one_reg.text == '':
                return Text('')
            new_text = f'{one_reg.text}⨂{self.n}'
            return Text(new_text)

        return super().wire_symbol(reg, idx)

    def __str__(self):
        return f'{self.gate}(oneach={self.n})'


def _get_on_each_classical_sim_test_cases() -> list['ClassicalSimTestCase']:
    """Test cases for the `OnEach` bloq.

    These specify concrete (non-symbolic) bloq instances with specific
    compile-time parameter combinations. Runtime quantum inputs are
    generated automatically by the verification framework.
    """
    from qualtran import QInt, QUInt
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.simulation.verification import ClassicalSimTestCase

    cases: list[ClassicalSimTestCase] = []
    for n in [2, 3, 4]:
        # Default dtype (QAny)
        cases.append(
            ClassicalSimTestCase(
                bloq=OnEach(n=n, gate=XGate()),
                name=f"OnEach(XGate, n={n})",
            )
        )
        # Unsigned
        cases.append(
            ClassicalSimTestCase(
                bloq=OnEach(n=n, gate=XGate(), target_dtype=QUInt(n)),
                name=f"OnEach(XGate, n={n}, QUInt)",
            )
        )
        # Signed
        cases.append(
            ClassicalSimTestCase(
                bloq=OnEach(n=n, gate=XGate(), target_dtype=QInt(n)),
                name=f"OnEach(XGate, n={n}, QInt)",
            )
        )
    return cases
