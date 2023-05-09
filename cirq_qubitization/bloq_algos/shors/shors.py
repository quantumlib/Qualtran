from functools import cached_property
from typing import Dict

import numpy as np
import sympy
from attrs import frozen

from cirq_qubitization.bloq_algos.basic_gates import CSwap, IntState
from cirq_qubitization.bloq_algos.basic_gates.t_gate import TGate
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.classical_sim import ClassicalValT
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder, SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from cirq_qubitization.t_complexity_protocol import TComplexity


@frozen
class ModExp(Bloq):
    """Perform $b^e mod m$ for constant `base` $b$, `mod` $m$, and register `exponent` $e$.

    This follows the reference implementation from Gidney and Ekera. Namely, it uses iterative
    controlled modular multiplication into a |1> register.

    Args:
        base: The integer base of the exponentiation
        mod: The integer modulus
        exp_bitsize: The size of the `exponent` thru-register
        x_bitsize: The size of the `x` right-register for returning the result of the
            exponentiation.

    Reference:
        [How to factor 2048 bit RSA integers in 8 hours using 20 million noisy qubits](https://arxiv.org/abs/1905.09749).
        Gidney and Ekera. 2019. Follows the 'reference implementation'.

    """

    base: int
    mod: int
    exp_bitsize: int
    x_bitsize: int

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters(
            [
                FancyRegister('exponent', bitsize=self.exp_bitsize),
                FancyRegister('x', bitsize=self.x_bitsize, side=Side.RIGHT),
            ]
        )

    @classmethod
    def make_for_shor(cls, big_n: int, g=None):
        """Factory method that sets up the modular exponentiation for Shor's algorithm.

        Args:
            big_n: The large composite number N. Used to set `mod`. Its bitsize is used
                to set `x_bitsize` and `exp_bitsize`.
            g: Optional base of the exponentiation. If `None`, we pick a random base.
        """
        little_n = int(np.ceil(np.log2(big_n)))
        if g is None:
            g = np.random.randint(big_n)
        return cls(base=g, mod=big_n, exp_bitsize=2 * little_n, x_bitsize=little_n)

    def on_classical_vals(self, exponent: int):
        return {'exponent': exponent, 'x': (self.base**exponent) % self.mod}

    def CtrlModMul(self, k: int):
        """Helper method to forward bitsize and mod arguments."""
        return CtrlModMul(k=k, bitsize=self.x_bitsize, mod=self.mod)

    def rough_decompose(self):
        k = sympy.Symbol('k')
        return [
            (1, IntState(val=1, bitsize=self.x_bitsize)),
            (self.exp_bitsize, self.CtrlModMul(k=k)),
        ]

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', exponent: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        (x,) = bb.add(IntState(val=1, bitsize=self.x_bitsize))
        exponent = bb.split(exponent)

        # https://en.wikipedia.org/wiki/Modular_exponentiation#Right-to-left_binary_method
        base = self.base
        for j in range(self.exp_bitsize - 1, 0 - 1, -1):
            exponent[j], x = bb.add(self.CtrlModMul(k=base), ctrl=exponent[j], x=x)
            base = base * base % self.mod

        return {'exponent': bb.join(exponent), 'x': x}

    def short_name(self) -> str:
        return f'{self.base}^e % {self.mod}'


@frozen
class CtrlModMul(Bloq):
    """Perform controlled `x *= k mod m`

    Args:
        k: The integer multiplicative constant.
        mod: The integer modulus
        bitsize: The size of the `x` register.
    """

    k: int
    mod: int
    bitsize: int

    def __attrs_post_init__(self):
        if isinstance(self.k, sympy.Expr) or isinstance(self.mod, sympy.Expr):
            return
        assert self.k < self.mod

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters.build(ctrl=1, x=self.bitsize)

    def on_classical_vals(self, ctrl, x) -> Dict[str, ClassicalValT]:
        if ctrl == 0:
            return {'ctrl': ctrl, 'x': x}

        assert ctrl == 1, ctrl
        return {'ctrl': ctrl, 'x': (x * self.k) % self.mod}

    def CSMAdd(self, k: int):
        """Helper method to forward bitsize and mod arguments."""
        return CtrlScaleModAdd(k=k, bitsize=self.bitsize, mod=self.mod)

    def rough_decompose(self):
        k = sympy.Symbol('k')
        return [(2, self.CSMAdd(k=k))]

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', ctrl: 'SoquetT', x: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        k = self.k
        neg_k_inv = self.mod - pow(k, -1, self.mod)

        y = bb.allocate(self.bitsize)

        # y += x*k
        # x += y * (-k^-1)
        ctrl, x, y = bb.add(self.CSMAdd(k=k), ctrl=ctrl, src=x, trg=y)
        ctrl, y, x = bb.add(self.CSMAdd(k=neg_k_inv), ctrl=ctrl, src=y, trg=x)

        # TODO: does this need to be quantum?
        ctrl, y, x = bb.add(CSwap(self.bitsize), ctrl=ctrl, x=y, y=x)

        bb.free(y)
        return {'ctrl': ctrl, 'x': x}

    def short_name(self) -> str:
        return f'x *= {self.k}'


@frozen
class CtrlScaleModAdd(Bloq):
    """Perform controlled `trg += src*k mod m`."""

    k: int
    bitsize: int
    mod: int

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters.build(ctrl=1, src=self.bitsize, trg=self.bitsize)

    def on_classical_vals(self, ctrl: int, src: int, trg: int) -> Dict[str, 'ClassicalValT']:
        if ctrl == 0:
            return {'ctrl': ctrl, 'src': src, 'trg': trg}
        assert ctrl == 1, 'Bad classical value for ctrl.'
        return {'ctrl': ctrl, 'src': src, 'trg': (trg + (src * self.k)) % self.mod}

    def rough_decompose(self):
        k = sympy.Symbol('k')
        return [(self.bitsize, CtrlModAdd(k=k, bitsize=self.bitsize, mod=self.mod))]

    def short_name(self) -> str:
        return f'y += x*{self.k} % {self.mod}'


@frozen
class CtrlModAdd(Bloq):
    """Perform controlled `x += k mod m`."""

    k: int
    bitsize: int
    mod: int

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters.build(ctrl=1, x=self.bitsize)

    def rough_decompose(self):
        k = sympy.Symbol('k')
        return [(5, CtrlAdd(k=k, bitsize=self.bitsize))]

    def t_complexity(self) -> 'TComplexity':
        ((n, bloq),) = self.rough_decompose()
        return n * bloq.t_complexity()

    def short_name(self) -> str:
        return f'x += {self.k} % {self.mod}'


@frozen
class CtrlAdd(Bloq):
    """Perform controlled `x += k`"""

    k: int
    bitsize: int

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters(
            [FancyRegister('ctrl', bitsize=1), FancyRegister('x', bitsize=self.bitsize)]
        )

    def rough_decompose(self):
        return [(2 * self.bitsize, TGate())]

    def t_complexity(self) -> 'TComplexity':
        return TComplexity(t=2 * self.bitsize)

    def short_name(self) -> str:
        return f'x += {self.k}'
