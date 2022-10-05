from typing import Sequence, List, Tuple, Set, Dict

from cirq_qubitization.gate_with_registers import Registers, Register
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloq
from cirq_qubitization.quantum_graph.fancy_registers import ApplyFRegister
from cirq_qubitization.quantum_graph.quantum_graph import (
    Wire,
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
        self._wires: List[Wire] = []

        # Initialize our BloqInstance counter
        self._i = 0

        # TODO: enforce linearity
        self._used: Set[Soquet] = set()

        self._parent_reg = parent_reg

    def initial_soquets(self) -> Dict[str, Soquet]:
        return {reg.name: Soquet(LeftDangle, reg.name) for reg in self._parent_reg}

    def _new_binst(self, bloq: Bloq):
        # TODO: bloqinstance has reference to parent bloq to make equality work?
        inst = BloqInstance(bloq, self._i)
        self._i += 1
        return inst

    def split(self, in_soq: Soquet, n: int) -> Tuple[Soquet, ...]:
        """Add a `Split` bloq to the compute graph."""
        binst = self._new_binst(Split(n))
        splitname = binst.bloq.registers[0].name

        self._wires.append(Wire(in_soq, Soquet(binst, splitname)))
        out_soqs = tuple(Soquet(binst, f'{splitname}{i}') for i in range(n))
        return out_soqs

    def join(self, in_soqs: Sequence[Soquet]) -> Soquet:
        binst = self._new_binst(Join(len(in_soqs)))
        joinname = binst.bloq.registers[0].name

        for i, in_soq in enumerate(in_soqs):
            self._wires.append(Wire(in_soq, Soquet(binst, f'{joinname}{i}')))

        return Soquet(binst, joinname)

    def add(self, bloq: Bloq, **soq_map: Soquet) -> Tuple[Soquet, ...]:
        # TODO: rename method?
        binst = self._new_binst(bloq)

        # TODO: use registers
        for out_name, in_soq in soq_map.items():
            if in_soq in self._used:
                # TODO: check this in split() and join()
                raise TypeError(f"{in_soq} re-used!")
            self._used.add(in_soq)  # TODO: factor out checking method.

            out_soq = Soquet(binst, out_name)
            self._wires.append(Wire(in_soq, out_soq))

        return tuple(Soquet(binst, reg_out_name(reg)) for reg in bloq.registers)

    def finalize(self, **out_soqs: Soquet) -> CompositeBloq:
        # TODO: rename method?

        # TODO: use parent_registers to iterate? or at least check that all the final_names
        #       are correct?
        for out_name, in_soq in out_soqs.items():
            self._wires.append(Wire(in_soq, Soquet(RightDangle, out_name)))

        # TODO: remove things from used
        # TODO: assert used() is empty

        return CompositeBloq(wires=self._wires, registers=self._parent_reg)
