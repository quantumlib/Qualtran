from functools import cached_property
from typing import Dict, Union

import sympy
from attrs import frozen

from cirq_qubitization.bloq_algos.basic_gates.t_gate import TGate
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.bloq_counts import SympySymbolAllocator
from cirq_qubitization.quantum_graph.classical_sim import ClassicalValT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters


@frozen
class CtrlScaleModAdd(Bloq):
    """Perform y += x*k mod m for constant k, m and quantum x, y.

    Args:
        k: The constant integer to scale `x` before adding into `y`.
        mod: The modulus of the addition
        bitsize: The size of the two registers.

    Registers:
     - ctrl: The control bit
     - x: The 'source' quantum register containing the integer to be scaled and added to `y`.
     - y: The 'destination' quantum register to which the addition will apply.
    """

    k: Union[int, sympy.Expr]
    mod: Union[int, sympy.Expr]
    bitsize: Union[int, sympy.Expr]

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters(
            [
                FancyRegister('ctrl', bitsize=1),
                FancyRegister('x', bitsize=self.bitsize),
                FancyRegister('y', bitsize=self.bitsize),
            ]
        )

    def bloq_counts(self, ssa: SympySymbolAllocator):
        k = ssa.new_symbol('k')
        return [(self.bitsize, CtrlModAddK(k=k, bitsize=self.bitsize, mod=self.mod))]

    def on_classical_vals(
        self, ctrl: 'ClassicalValT', x: 'ClassicalValT', y: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        if ctrl == 0:
            return {'ctrl': 0, 'x': x, 'y': y}

        assert ctrl == 1, 'Bad ctrl value.'
        y_out = (y + x * self.k) % self.mod
        return {'ctrl': ctrl, 'x': x, 'y': y_out}

    def short_name(self) -> str:
        return f'y += x*{self.k} % {self.mod}'


@frozen
class CtrlModAddK(Bloq):
    """Perform x += k mod m for constant k, m and quantum x.

    Args:
        k: The integer to add to `x`.
        mod: The modulus for the addition.
        bitsize: The bitsize of the `x` register.

    Registers:
        ctrl: The control bit
        x: The register to perform the in-place modular addition.
    """

    k: Union[int, sympy.Expr]
    mod: Union[int, sympy.Expr]
    bitsize: Union[int, sympy.Expr]

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters(
            [FancyRegister('ctrl', bitsize=1), FancyRegister('x', bitsize=self.bitsize)]
        )

    def bloq_counts(self, ss):
        k = ss.new_symbol('k')
        return [(5, CtrlAddK(k=k, bitsize=self.bitsize))]

    def short_name(self) -> str:
        return f'x += {self.k} % {self.mod}'


@frozen
class CtrlAddK(Bloq):
    """Perform x += k for constant k and quantum x.

    Args:
        k: The integer to add to `x`.
        bitsize: The bitsize of the `x` register.

    Registers:
        ctrl: The control bit
        x: The register to perform the addition.
    """

    k: Union[int, sympy.Expr]
    bitsize: Union[int, sympy.Expr]

    def short_name(self) -> str:
        return f'x += {self.k}'

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters(
            [FancyRegister('ctrl', bitsize=1), FancyRegister('x', bitsize=self.bitsize)]
        )

    def bloq_counts(self, mgr):
        return [(2 * self.bitsize, TGate())]
