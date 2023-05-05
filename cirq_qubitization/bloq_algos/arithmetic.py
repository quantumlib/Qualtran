from attrs import frozen
import numpy as np

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
    """

    nbits: int

    @property
    def registers(self):
        return FancyRegisters.build(a=self.nbits, b=self.nbits)

    def t_complexity(self):
        num_clifford = (self.nbits - 2) * 19 + 16
        num_t_gates = 4 * self.nbits - 4
        return t_complexity_protocol.TComplexity(t=num_t_gates, clifford=num_clifford)


@frozen
class Square(Bloq):
    r"""Square an n-bit number.

    Implements $U|a\rangle|0\rangle -\rightarrow |a\rangle|a^2\rangle$ using 4n - 4 T gates.

    Args:
        nbits: Number of bits used to represent the integer and .

    Registers:
     - a: A nbit-sized input register (register a above).
     - result: A 2-nbit-sized input/ouput register (register b above).

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First
        Quantization](https://arxiv.org/abs/2105.12767) pg 76 give a Toffoli
        complexity for squaring.
    """

    nbits: int

    @property
    def registers(self):
        return FancyRegisters.build(a=self.nbits, result=2 * self.nbits)

    def t_complexity(self):
        # TODO actual gate implementation + determine cliffords.
        num_toff = self.nbits * (self.nbits - 1)
        return t_complexity_protocol.TComplexity(t=4 * num_toff)


@frozen
class SumOfSquares(Bloq):
    r"""Compute the sum of squares of k n-bit numbers.

    Implements $U|a\rangle|b\rangle...|k\rangle|0\rangle \rightarrow |a\rangle|b\rangle..|k\rangle|a^2+b^2+..k^2\rangle$ using $4 k n^2$ Ts.

    Args:
        nbits_in: Number of bits used to represent each of the k integers.
        nbits_out: Number of bits used to represent the output result. Should be
            ~ 2 * nbits_in * log2(k).

    Registers:
     - inputs: nbits_in-sized input registers.
     - result: nbits_out-sized register holding the result.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First
        Quantization](https://arxiv.org/abs/2105.12767) pg 80 give a Toffoli
        complexity for squaring.
    """

    nbits_in: int
    nbits_out: int
    k: int

    @property
    def registers(self):
        return FancyRegisters.build(inputs=self.k * self.nbits_in, result=self.nbits_out)

    def t_complexity(self):
        # TODO actual gate implementation + determine cliffords.
        num_toff = self.k * self.nbits_in**2 - self.nbits_in
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
     - a: bit-sized input registers.
     - b: bit-sized input registers.
     - result: A nbit-sized ouput register (register b above).

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First
        Quantization](https://arxiv.org/abs/2105.12767) pg 81 give a Toffoli
        complexity for squaring.
    """

    nbits: int
    mbits: int

    @property
    def registers(self):
        return FancyRegisters.build(a=self.nbits, b=self.mbits, result=max(self.nbits, self.mbits))

    def t_complexity(self):
        # TODO actual gate implementation + determine cliffords.
        num_toff = 2 * self.nbits * self.mbits - max(self.nbits, self.mbits)
        return t_complexity_protocol.TComplexity(t=4 * num_toff)
