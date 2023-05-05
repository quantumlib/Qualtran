from attrs import frozen
import numpy as np

from cirq_qubitization import t_complexity_protocol
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.bloq_algos.arithmetic import GreaterThan
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegisters


@frozen
class Comparator(Bloq):
    r"""Compare and potentially swaps two n-bit numbers.

    Implements $U|a\rangle|b\rangle|0\rangle \rightarrow |\min(a,b)\rangle|\max(a,b)\rangle|a>b\rangle$,

    where $a$ and $b$ are n-qubit quantum registers. On output a and b are
    potentially swapped if a > b. Forms the base primitive for sorting.

    Args:
        nbits: Number of bits used to represent each integer.

    Registers:
     - a: A nbit-sized input register (register a above).
     - b: A nbit-sized input register (register b above).
     - anc: A nbit-sized input register (register anc above).

    References:
        [Improved techniques for preparing eigenstates of fermionic
        Hamiltonians](https://www.nature.com/articles/s41534-018-0071-5),
        Fig. 1 in main text.
    """

    nbits: int

    @property
    def registers(self):
        return FancyRegisters.build(a=self.nbits, b=self.nbits, anc=1)

    def pretty_name(self) -> str:
        return "Cmprtr"

    def t_complexity(self):
        # complexity is from less than on two n qubit numbers + controlled swap
        # Hard code for now until CSwap-Bloq is merged.
        t_complexity = GreaterThan(self.nbits).t_complexity()
        t_complexity += t_complexity_protocol.TComplexity(t=14 * self.nbits)
        return t_complexity


@frozen
class BitonicSort(Bloq):
    r"""Sort k n-bit numbers.

    TODO: actually implement the algorithm using comapritor Hiding ancilla cost
        for the moment.

    Args:
        nbits: Number of bits used to represent each integer.
        k: Number of integers to sort.

    Registers:
     - input: A nbit-sized input register (register a above).
     - output: A nbit-sized input register (register a above).

    References:
    """

    nbits: int
    k: int

    @property
    def registers(self):
        regs = {f"a_{i}": self.nbits for i in range(self.k)}
        return FancyRegisters.build(**regs)

    def pretty_name(self) -> str:
        return "BSort"

    def t_complexity(self):
        # Need k * log^2(k) comparisons.
        # TODO: This is Big-O complexity?
        return (
            self.k
            * int(np.ceil(max(np.log2(self.k) ** 2.0, 1)))
            * Comparator(self.nbits).t_complexity()
        )
