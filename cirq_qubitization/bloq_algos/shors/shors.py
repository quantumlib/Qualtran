"""

Period finding: find r of f(e) = g^e mod N.

The number of qubits in the exponent is 2n (why?)
The bitsize of the exponent is 2n (why?)

Briefly, $g^r - 1 = 0 mod N$ by definition. and $g^r - 1 = (g^{r/2} - 1)(g^{r/2} + 1)
and we can use the GCD to find the actual factors.

----
Discrete log (Ekera Hastad) has two exponents: e1, e2; bitsize 2m, m (resp).
m such that p + q < 2^m
N = pq

craig says m = 0.5n + O(1)

Period finding f(e1, e2) = g^e1 y^-e2
total exponent bitsize is 1.5n + O(1)

---

"""
from typing import Dict

import sympy
from numpy.typing import NDArray

from cirq_qubitization.quantum_graph.bloq import Bloq
from attrs import frozen
from functools import cached_property

from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder, SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegisters, FancyRegister, Side
from cirq_qubitization.quantum_graph.quantum_graph import Soquet


@frozen
class ModExp(Bloq):
    g: int
    mod_N: int
    exp_bitsize: int
    x_bitsize: int

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters([
            FancyRegister('exponent', bitsize=self.exp_bitsize),
            FancyRegister('x', bitsize=self.x_bitsize, side=Side.RIGHT)
        ])

    def CtrlModMul(self, k:int):
        return CtrlModMul(k=k, x_bitsize=self.x_bitsize, mod_N=self.mod_N)


    def build_composite_bloq(
            self, bb: 'CompositeBloqBuilder', exponent: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        x = bb.allocate(self.x_bitsize)  # TODO: initialize into "1" state.
        exponent = bb.split(exponent)

        for j in range(self.exp_bitsize):
            k = (self.g ** (2 ** j)) % self.mod_N
            exponent[j], x = bb.add(self.CtrlModMul(k=k), ctrl=exponent[j], x=x)

        return {'exponent': bb.join(exponent), 'x': x}


@frozen
class CtrlModMul(Bloq):
    k: int
    x_bitsize: int
    mod_N: int

    def __attrs_post_init__(self):
        pass
        # assert self.k < self.mod_N # todo: work with sympy

    def short_name(self) -> str:
        return f'x *= {self.k}'

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters([
            FancyRegister('ctrl', bitsize=1),
            FancyRegister('x', bitsize=self.x_bitsize),
        ])

    def Add(self, k:int):
        return CtrlScaleAdd(k=k, x_bitsize=self.x_bitsize, mod_N=self.mod_N)

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', ctrl: 'SoquetT', x: 'SoquetT',
    ) -> Dict[str, 'SoquetT']:
        k = self.k
        neg_k_inv = -k**-1 # TODO: is there modular stuff happening here?

        y = bb.allocate(self.x_bitsize)

        # y += x*k
        # x += y * (-k^-1)

        ctrl, x, y = bb.add(self.Add(k=k), ctrl=ctrl, src=x, trg=y)
        ctrl, y, x = bb.add(self.Add(k=neg_k_inv), ctrl=ctrl, src=y, trg=x)

        bb.free(x)
        return {'ctrl': ctrl, 'x': y}


@frozen
class CtrlScaleAdd(Bloq):
    k: int
    x_bitsize: int
    mod_N: int

    def short_name(self) -> str:
        return f'y += x*{self.k}'

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters([
            FancyRegister('ctrl', bitsize=1),
            FancyRegister('src', bitsize=self.x_bitsize),
            FancyRegister('trg', bitsize=self.x_bitsize),
        ])

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', ctrl:Soquet, src:Soquet, trg:Soquet
    ) -> Dict[str, 'SoquetT']:
        k = self.k

        src = bb.split(src)
        for j in range(self.x_bitsize):
            a = (k*2**j) % self.mod_N
            src[j], trg = bb.add(CtrlAdd(a=a), ctrl=src[j], trg=trg)

        return {'ctrl': ctrl, 'src': src, 'trg': trg}

