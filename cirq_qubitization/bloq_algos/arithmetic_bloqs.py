from attrs import frozen
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegisters, FancyRegister


@frozen
class Add(Bloq):
    """An n-bit addition gate.

    Implements U|a>|b> -> |a>|a+b> using 4n - 4 T gates.

    Args:
        bitsize: Number of bits used to represent each integer. Must be large
            enough to hold the result in the output register of a + b.

    Registers:
     - input: A bitsize-size input register (register a above).
     - inout: A bitsize-size input/ouput register (register b above).

    References:
    """

    bitsize: int

    @property
    def registers(self):
        return FancyRegisters(
            [
                FancyRegister('input', bitsize=1, wireshape=(self.bitsize,)),
                FancyRegister('target', bitsize=1, wireshape=(self.bitsize,)),
            ]
        )
