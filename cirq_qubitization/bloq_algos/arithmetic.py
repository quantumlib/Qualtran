from attrs import frozen

from cirq_qubitization import t_complexity_protocol
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegisters


@frozen
class Add(Bloq):
    r"""An n-bit addition gate.

    Implements $U|a\rangle|b\rangle \rightarrow |a\rangle|a+b\rangle$ using $4n - 4 T$ gates.

    Args:
        nbits: Number of bits used to represent each integer. Must be large
            enough to hold the result in the output register of a + b.

    Registers:
     - a: A nbit-sized input register (register a above).
     - b: A nbit-sized input/ouput register (register b above).

    References:
        [Halving the cost of quantum addition](https://arxiv.org/abs/1709.06648)
    """

    nbits: int

    @property
    def registers(self):
        return FancyRegisters.build(a=self.nbits, b=self.nbits)

    def pretty_name(self) -> str:
        return "a + b"

    def t_complexity(self):
        num_clifford = (self.nbits - 2) * 19 + 16
        num_t_gates = 4 * self.nbits - 4
        return t_complexity_protocol.TComplexity(t=num_t_gates, clifford=num_clifford)


@frozen
class Square(Bloq):
    r"""Square an n-bit number.

    Implements $U|a\rangle|0\rangle -\rightarrow |a\rangle|a^2\rangle$ using $4n - 4 T$ gates.

    Args:
        nbits: Number of bits used to represent the integer and .

    Registers:
     - a: A nbit-sized input register (register a above).
     - result: A 2-nbit-sized input/ouput register.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First
        Quantization](https://arxiv.org/abs/2105.12767) pg 76 give a Toffoli
        complexity for squaring.
    """

    nbits: int

    @property
    def registers(self):
        return FancyRegisters.build(a=self.nbits, result=2 * self.nbits)

    def pretty_name(self) -> str:
        return "a^2"

    def t_complexity(self):
        # TODO actual gate implementation + determine cliffords.
        num_toff = self.nbits * (self.nbits - 1)
        return t_complexity_protocol.TComplexity(t=4 * num_toff)


@frozen
class SumOfSquares(Bloq):
    r"""Compute the sum of squares of k n-bit numbers.

    Implements $U|a\rangle|b\rangle...|k\rangle|0\rangle \rightarrow |a\rangle|b\rangle..|k\rangle|a^2+b^2+..k^2\rangle$ using $4 k n^2$ Ts.

    Args:
        nbits: Number of bits used to represent each of the k integers.

    Registers:
     - a_k: k n-bit registers.
     - result: 2 * nbits + 1 sized output register.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First
        Quantization](https://arxiv.org/abs/2105.12767) pg 80 give a Toffoli
        complexity for squaring.
    """

    nbits: int
    k: int

    @property
    def registers(self):
        regs = {f"a_{i}": self.nbits for i in range(self.k)}
        return FancyRegisters.build(**regs, result=2 * self.nbits + 1)

    def pretty_name(self) -> str:
        return "SOS"

    def t_complexity(self):
        # TODO actual gate implementation + determine cliffords.
        num_toff = self.k * self.nbits**2 - self.nbits
        if self.k % 3 == 0:
            num_toff -= 1
        return t_complexity_protocol.TComplexity(t=4 * num_toff)


@frozen
class Product(Bloq):
    r"""Compute the product of an `n` an `m` bit integer.

    Implements $U|a\rangle|b\rangle|0\rangle -\rightarrow
    |a\rangle|b\rangle|ab\rangle$ using $2nm-n$ Toffolis.

    Args:
        nbits: Number of bits used to represent the first integer.
        mbits: Number of bits used to represent the second integer.

    Registers:
     - a: nbit-sized input registers.
     - b: mbit-sized input registers.
     - result: A 2nbit-sized ouput register (register b above).

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First
        Quantization](https://arxiv.org/abs/2105.12767) pg 81 give a Toffoli
        complexity for squaring.
    """

    nbits: int
    mbits: int

    @property
    def registers(self):
        return FancyRegisters.build(
            a=self.nbits, b=self.mbits, result=2 * max(self.nbits, self.mbits)
        )

    def pretty_name(self) -> str:
        return "a*b"

    def t_complexity(self):
        # TODO actual gate implementation + determine cliffords.
        num_toff = 2 * self.nbits * self.mbits - max(self.nbits, self.mbits)
        return t_complexity_protocol.TComplexity(t=4 * num_toff)


@frozen
class GreaterThan(Bloq):
    r"""Compare to n-bit integers.

    Implements $U|a\rangle|b\rangle|0\rangle \rightarrow
    |a\rangle|b\rangle|a > b\rangle$ using $8n T$  gates.


    Args:
        nbits: Number of bits used to represent the two integers a and b.

    Registers:
     - a: n-bit-sized input registers.
     - b: n-bit-sized input registers.
     - result: A nbit-sized ouput register (register b above).

    References:
        [Improved techniques for preparing eigenstates of fermionic
        Hamiltonians](https://www.nature.com/articles/s41534-018-0071-5#additional-information),
        Comparison Oracle from SI: Appendix 2B (pg 3)
    """
    nbits: int

    @property
    def registers(self):
        return FancyRegisters.build(a=self.nbits, b=self.nbits, anc=1)

    def pretty_name(self) -> str:
        return "a gt b"

    def t_complexity(self):
        # TODO actual gate implementation + determine cliffords.
        return t_complexity_protocol.TComplexity(t=8 * self.nbits)
