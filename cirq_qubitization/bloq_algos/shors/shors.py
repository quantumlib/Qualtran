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

https://arxiv.org/abs/1905.09749

---

"""
from functools import cached_property
from typing import Any, Dict, Iterable

import numpy as np
import sympy
from attrs import frozen
from numpy.typing import NDArray

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder, SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from cirq_qubitization.quantum_graph.quantum_graph import Soquet


def big_endian_bits_to_int_cirq(bits: Iterable[Any]) -> int:
    """Returns the big-endian integer specified by the given bits.
    Args:
        bits: Descending bits of the integer, with the 1s bit at the end.
    Returns:
        The integer.
    Examples:
        >>> cirq.big_endian_bits_to_int([0, 1])
        1
        >>> cirq.big_endian_bits_to_int([1, 0])
        2
        >>> cirq.big_endian_bits_to_int([0, 1, 0])
        2
        >>> cirq.big_endian_bits_to_int([1, 0, 0, 1, 0])
        18
    """
    result = 0
    for e in bits:
        result <<= 1
        if e:
            result |= 1
    return result


def big_endian_bits_to_int(bitstrings):
    bitstrings = np.atleast_2d(bitstrings)
    basis = 2 ** np.arange(bitstrings.shape[1] - 1, 0 - 1, -1)
    return np.sum(basis * bitstrings, axis=1)


def int_to_bits(x: int, w: int):
    return np.asarray([int(b) for b in f'{x:0{w}b}'])


@frozen
class ModExp(Bloq):
    """
    The reference implementation works the way most implementations of Shor’s algorithm do, by
    decomposing exponentiation into iterative controlled modular multiplication [6, 31, 42, 87, 90, 91].
    A register x is initialized to the |1> state, then a controlled modular multiplication of the classical
    constant g^2^j (mod N) into x is performed, controlled by the qubit e_j from the exponent e, for
    each integer j from n_e − 1 down to 0. After the multiplications are done, x is storing g^e (mod N)
    and measuring x completes the hard part of Shor’s algorithm.


    """

    g: int
    mod_N: int
    exp_bitsize: int
    x_bitsize: int

    @property
    def little_n(self):
        return int(np.ceil(self.mod_N))

    @classmethod
    def make_for_shor(cls, big_n: int, g=None):
        little_n = int(np.ceil(np.log2(big_n)))
        if g is None:
            g = np.random.randint(big_n)
        return cls(g=g, mod_N=big_n, exp_bitsize=2 * little_n, x_bitsize=little_n)

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters(
            [
                FancyRegister('exponent', bitsize=self.exp_bitsize),
                FancyRegister('x', bitsize=self.x_bitsize, side=Side.RIGHT),
            ]
        )

    def apply_classical(self, exponent):
        assert exponent.shape == (self.exp_bitsize,)
        (exp_number,) = big_endian_bits_to_int(exponent)
        res = self.g**exp_number
        res %= self.mod_N
        res = int_to_bits(res, self.x_bitsize)
        return exponent, res

    def CtrlModMul(self, k: int):
        return CtrlModMul(k=k, x_bitsize=self.x_bitsize, mod_N=self.mod_N)

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', exponent: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        x = bb.allocate(self.x_bitsize)  # TODO: initialize into "1" state.
        exponent = bb.split(exponent)

        for j in range(self.exp_bitsize):
            k = (self.g ** (2**j)) % self.mod_N
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
        return FancyRegisters(
            [FancyRegister('ctrl', bitsize=1), FancyRegister('x', bitsize=self.x_bitsize)]
        )

    def Add(self, k: int):
        return CtrlScaleAdd(k=k, x_bitsize=self.x_bitsize, mod_N=self.mod_N)

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', ctrl: 'SoquetT', x: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        k = self.k
        neg_k_inv = -(k**-1)  # TODO: is there modular stuff happening here?

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
        return FancyRegisters(
            [
                FancyRegister('ctrl', bitsize=1),
                FancyRegister('src', bitsize=self.x_bitsize),
                FancyRegister('trg', bitsize=self.x_bitsize),
            ]
        )

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', ctrl: Soquet, src: Soquet, trg: Soquet
    ) -> Dict[str, 'SoquetT']:
        k = self.k

        src = bb.split(src)
        for j in range(self.x_bitsize):
            a = (k * 2**j) % self.mod_N
            src[j], trg = bb.add(CtrlAdd(a=a), ctrl=src[j], trg=trg)

        return {'ctrl': ctrl, 'src': src, 'trg': trg}
