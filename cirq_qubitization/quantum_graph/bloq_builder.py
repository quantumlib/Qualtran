from typing import Sequence, List, Tuple, Set, Dict

from cirq_qubitization.gate_with_registers import Registers, Register
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloq
from cirq_qubitization.quantum_graph.fancy_registers import ApplyFRegister
from cirq_qubitization.quantum_graph.quantum_graph import (
    Wiring,
    Soquet,
    LeftDangle,
    RightDangle,
    BloqInstance,
)
from cirq_qubitization.quantum_graph.util_bloqs import Split, Join


def reg_out_name(reg: Register):
    # TODO: ???
    if isinstance(reg, ApplyFRegister):
        return reg.out_name
    return reg.name


class BloqBuilder:
    def __init__(self, parent_reg: Registers):
        # Our builder builds a sequence of Wires
        self._wires: List[Wiring] = []

        # Initialize our BloqInstance counter
        self._i = 0

        # To name: this is a list of ports aka half-wires. We enforce linearity.
        self._used: Set[Soquet] = set()

        # Keep a reference to this for some reason
        self._parent_reg = parent_reg

    def initial_soquets(self) -> Dict[str, Soquet]:
        return {reg.name: Soquet(LeftDangle, reg.name) for reg in self._parent_reg}

    def _new_binst(self, bloq: Bloq):
        inst = BloqInstance(bloq, self._i)
        self._i += 1
        return inst

    def split(self, fr_port: Soquet, n: int) -> Tuple[Soquet, ...]:
        """Add a `Split` bloq to the compute graph."""
        binst = self._new_binst(Split(n))
        splitname = binst.bloq.registers[0].name

        self._wires.append(Wiring(fr_port, Soquet(binst, splitname)))
        new_ports = tuple(Soquet(binst, f'{splitname}{i}') for i in range(n))
        return new_ports

    def join(self, xs: Sequence[Soquet]) -> Soquet:
        binst = self._new_binst(Join(len(xs)))
        joinname = binst.bloq.registers[0].name

        for i, x in enumerate(xs):
            self._wires.append(Wiring(x, Soquet(binst, f'{joinname}{i}')))

        return Soquet(binst, joinname)

    def add(self, bloq: Bloq, **reg_map: Soquet) -> Tuple[Soquet, ...]:
        # TODO: rename to call(?)
        inst = self._new_binst(bloq)

        for to_name, fr_port in reg_map.items():
            if fr_port in self._used:
                # TODO: check this in split() and join()
                raise TypeError(f"{fr_port} re-used!")
            self._used.add(fr_port)

            # TODO: construct out ports for returning while we're making wires
            self._wires.append(Wiring(fr_port, Soquet(inst, to_name)))

        return tuple(Soquet(inst, reg_out_name(reg)) for reg in bloq.registers)

    def finalize(self, **final_ports: Soquet) -> CompositeBloq:
        # TODO: rename "return"

        # TODO: use parent_registers to iterate? or at least check that all the final_names
        #       are correct?
        for final_name, fr_port in final_ports.items():
            self._wires.append(Wiring(fr_port, Soquet(RightDangle, final_name)))

        # TODO: remove things from used
        # TODO: assert used() is empty

        return CompositeBloq(wires=self._wires, registers=self._parent_reg)
