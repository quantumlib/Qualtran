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
from typing import Any, Dict, Iterable, Tuple

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
    return np.asarray([int(b) for b in f'{x:0{w}b}'], dtype=np.uint8)


@frozen
class AllocateInt(Bloq):
    """Allocate an `n` bit register storing `val` ."""

    val: int
    width: int

    def __attrs_post_init__(self):
        assert self.val >= 0
        assert self.val.bit_length() <= self.width

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters([FancyRegister('x', bitsize=self.width, side=Side.RIGHT)])

    def apply_classical(self) -> Dict[str, NDArray[np.uint8]]:
        return {'x': self.val}


@frozen
class ModExp(Bloq):
    """Perform $b^e mod m$ for constant `base` $b$, `mod` $m$, and register `exponent` $e$.

    The reference implementation works the way most implementations of Shor’s algorithm do, by
    decomposing exponentiation into iterative controlled modular multiplication [6, 31, 42, 87, 90, 91].
    A register x is initialized to the |1> state, then a controlled modular multiplication of the classical
    constant g^2^j (mod N) into x is performed, controlled by the qubit e_j from the exponent e, for
    each integer j from n_e − 1 down to 0. After the multiplications are done, x is storing g^e (mod N)
    and measuring x completes the hard part of Shor’s algorithm.

    Args:
        base: The integer base of the exponentiation
        mod: The integer modulus
        exp_bitsize: The size of the `exponent` thru-register
        x_bitsize: The size of the `x` right-register for returning the result of the
            exponentiation.
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

    def short_name(self) -> str:
        return f'{self.base}^e % {self.mod}'

    @classmethod
    def make_for_shor(cls, big_n: int, g=None):
        """Factory method that sets up the modular exponentiation for a run of shors.

        Args:
            big_n: The large composite number N. Used to set `mod`. Its bitsize is used
                to set `x_bitsize` and `exp_bitsize`.
            g: Optional base of the exponentiation. If `None`, pick a random base.
        """
        if isinstance(big_n, sympy.Expr):
            little_n = sympy.ceiling(sympy.log(big_n, 2))
            # little_n = little_n.subs(little_n, sympy.Symbol('n'))
        else:
            little_n = int(np.ceil(np.log2(big_n)))
        if g is None:
            g = np.random.randint(big_n)
        return cls(base=g, mod=big_n, exp_bitsize=2 * little_n, x_bitsize=little_n)

    def apply_classical(self, exponent: int):
        assert 0 <= exponent < 2**self.exp_bitsize
        return {'exponent': exponent, 'x': (self.base**exponent) % self.mod}

    def CtrlModMul(self, k: int):
        return CtrlModMul(k=k, x_bitsize=self.x_bitsize, mod=self.mod)

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', exponent: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        (x,) = bb.add(AllocateInt(val=1, width=self.x_bitsize))
        exponent = bb.split(exponent)

        # https://en.wikipedia.org/wiki/Modular_exponentiation#Right-to-left_binary_method
        base = self.base
        for j in range(self.exp_bitsize - 1, 0 - 1, -1):
            exponent[j], x = bb.add(self.CtrlModMul(k=base), ctrl=exponent[j], x=x)
            base = base * base % self.mod

        return {'exponent': bb.join(exponent), 'x': x}


@frozen
class CtrlModMul(Bloq):
    """Perform controlled `x *= k mod m`

    Args:
        k: The integer multiplicative constant.
        mod: The integer modulus
        x_bitsize: The size of the `x` register.
    """

    k: int
    mod: int
    x_bitsize: int

    def __attrs_post_init__(self):
        pass
        # assert self.k < self.mod_N # todo: work with sympy

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters(
            [FancyRegister('ctrl', bitsize=1), FancyRegister('x', bitsize=self.x_bitsize)]
        )

    def short_name(self) -> str:
        return f'x *= {self.k}'

    def apply_classical(self, ctrl, x) -> Dict[str, NDArray[np.uint8]]:
        if ctrl == 0:
            return {'ctrl': ctrl, 'x': x}

        assert ctrl == 1, ctrl
        return {'ctrl': ctrl, 'x': (x * self.k) % self.mod}

    def Add(self, k: int):
        return CtrlScaleAdd(k=k, x_bitsize=self.x_bitsize, mod_N=self.mod)

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
