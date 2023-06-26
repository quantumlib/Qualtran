from functools import cached_property
from typing import Dict, List, Tuple

import sympy
from attrs import frozen

from cirq_qubitization.bloq_algos.basic_gates import CSwap
from cirq_qubitization.bloq_algos.factoring.mod_add import CtrlScaleModAdd
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.bloq_counts import SympySymbolAllocator
from cirq_qubitization.quantum_graph.classical_sim import ClassicalValT
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder, SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegisters
from cirq_qubitization.quantum_graph.musical_score import Circle, directional_text_box, WireSymbol
from cirq_qubitization.quantum_graph.quantum_graph import Soquet


@frozen
class CtrlModMul(Bloq):
    """Perform controlled `x *= k mod m` for constant k, m and variable x.

    Args:
        k: The integer multiplicative constant.
        mod: The integer modulus.
        bitsize: The size of the `x` register.

    Registers:
     - ctrl: The control bit
     - x: The integer being multiplied
    """

    k: int
    mod: int
    bitsize: int

    def __attrs_post_init__(self):
        if isinstance(self.k, sympy.Expr):
            return
        if isinstance(self.mod, sympy.Expr):
            return

        assert self.k < self.mod

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters.build(ctrl=1, x=self.bitsize)

    def Add(self, k: int):
        """Helper method to forward attributes to `CtrlScaleModAdd`."""
        return CtrlScaleModAdd(k=k, bitsize=self.bitsize, mod=self.mod)

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', ctrl: 'SoquetT', x: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        k = self.k
        neg_k_inv = -pow(k, -1, mod=self.mod)

        # We store the result of the CtrlScaleModAdd into this new register
        # and then clear the original `x` register by multiplying in the inverse.
        y = bb.allocate(self.bitsize)

        # y += x*k
        ctrl, x, y = bb.add(self.Add(k=k), ctrl=ctrl, x=x, y=y)
        # x += y * (-k^-1)
        ctrl, y, x = bb.add(self.Add(k=neg_k_inv), ctrl=ctrl, x=y, y=x)

        # y contains the answer and x is empty.
        # In [GE2019], it is asserted that the registers can be swapped via bookkeeping.
        # This is not correct: we do not want to swap the registers if the control bit
        # is not set.
        ctrl, x, y = bb.add(CSwap(self.bitsize), ctrl=ctrl, x=x, y=y)
        bb.free(y)
        return {'ctrl': ctrl, 'x': x}

    def bloq_counts(self, ssa: SympySymbolAllocator) -> List[Tuple[int, 'Bloq']]:
        k = ssa.new_symbol('k')
        return [(2, self.Add(k=k)), (1, CSwap(self.bitsize))]

    def on_classical_vals(self, ctrl, x) -> Dict[str, ClassicalValT]:
        if ctrl == 0:
            return {'ctrl': ctrl, 'x': x}

        assert ctrl == 1, ctrl
        return {'ctrl': ctrl, 'x': (x * self.k) % self.mod}

    def short_name(self) -> str:
        return f'x *= {self.k} % {self.mod}'

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        if soq.reg.name == 'ctrl':
            return Circle(filled=True)
        if soq.reg.name == 'x':
            return directional_text_box(f'*={self.k}', side=soq.reg.side)
