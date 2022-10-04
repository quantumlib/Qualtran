from dataclasses import dataclass
from functools import cached_property

from cirq_qubitization.gate_with_registers import Registers, Register
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.bloq_builder import BloqBuilder
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloq
from cirq_qubitization.quantum_graph.fancy_registers import (
    ApplyFRegister,
)
from cirq_qubitization.quantum_graph.quantum_graph import Port, LeftDangle


@dataclass(frozen=True)
class SingleControlModMultiply(Bloq):
    x_bitsize: int
    mul_constant: int

    def short_name(self) -> str:
        return 'SCMM'

    @cached_property
    def registers(self) -> Registers:
        return Registers(
            [
                Register('control', 1),
                ApplyFRegister(
                    'x', self.x_bitsize, 'x_out', in_text='x', out_text=f'{self.mul_constant}*x'
                ),
            ]
        )


@dataclass(frozen=True)
class ModMultiply(Bloq):
    exponent_bitsize: int
    x_bitsize: int
    mul_constant: int
    mod_N: int

    @cached_property
    def registers(self) -> Registers:
        return Registers.build(exponent=self.exponent_bitsize, x=self.x_bitsize)

    def better_decompose(self, bb: BloqBuilder, exponent: Port, x: Port) -> CompositeBloq:
        ctls = bb.split(exponent, self.exponent_bitsize)
        out_ctls = []

        for j in range(self.exponent_bitsize - 1, 0 - 1, -1):
            c = self.mul_constant**2**j % self.mod_N
            ctl, x = bb.add(
                SingleControlModMultiply(x_bitsize=self.x_bitsize, mul_constant=c),
                control=ctls[j],
                x=x,
            )
            out_ctls.append(ctl)

        ret_dict = {'exponent': bb.join(out_ctls), 'x': x}
        return bb.finalize(**ret_dict)

    def decompose(self) -> CompositeBloq:
        # TODO: context manager
        # TODO: parent class
        port_dict = {reg.name: Port(LeftDangle, reg.name) for reg in self.registers}
        bb = BloqBuilder(self.registers)
        return self.better_decompose(bb=bb, **port_dict)
