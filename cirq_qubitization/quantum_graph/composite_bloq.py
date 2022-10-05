from typing import Sequence, Tuple, Set, List, Dict, Optional

import networkx as nx
import cirq
from cirq_qubitization.gate_with_registers import Registers, Register
from cirq_qubitization.quantum_graph.bloq import Bloq, NoCirqEquivalent
from cirq_qubitization.quantum_graph.fancy_registers import (
    ApplyFRegister,
    SplitRegister,
    JoinRegister,
)
from cirq_qubitization.quantum_graph.quantum_graph import (
    Wire,
    Soquet,
    LeftDangle,
    RightDangle,
    BloqInstance,
    DanglingT,
)
from cirq_qubitization.quantum_graph.util_bloqs import Split, Join


class CompositeBloq(Bloq):
    def __init__(self, wires: Sequence[Wire], registers: Registers):
        self._wires = tuple(wires)
        self._registers = registers

    @property
    def registers(self) -> Registers:
        return self._registers

    @property
    def wires(self) -> Tuple[Wire, ...]:
        return self._wires

    def to_cirq_circuit(self, **quregs: Sequence[cirq.Qid]):
        return cbloq_to_cirq_circuit(quregs, self.wires)


def _process_binst(
    binst: BloqInstance, soqmap: Dict[Soquet, Sequence[cirq.Qid]], binst_graph: nx.DiGraph
) -> Optional[cirq.Operation]:
    """Helper function used in `cbloq_to_cirq_circuit`.

    Args:
        binst: The current BloqInstance to process
        soqmap: The current mapping between soquets and qubits that *is updated by this function*.
            At input, the mapping should contain values for all of binst's soquets. Afterwards,
            it should contain values for all of binst's successors' soquets.
        binst_graph: Used for finding binst's successors to update soqmap.

    Returns:
        an operation if there is a corresponding one in Cirq. Some bookkeeping Bloqs will not
        correspond to Cirq operations.
    """
    if not isinstance(binst, DanglingT):
        # TODO: dealing with dangling is annoying.

        # First: fix internal connections, i.e. differing in- and out- names
        for reg in binst.bloq.registers:
            if isinstance(reg, SplitRegister):
                for i in range(reg.bitsize):
                    soqmap[Soquet(binst, f'{reg.name}{i}')] = soqmap[Soquet(binst, reg.name)][
                        i : i + 1
                    ]
            elif isinstance(reg, JoinRegister):
                _qq = []
                for i in range(reg.bitsize):
                    _qq.extend(soqmap[Soquet(binst, f'{reg.name}{i}')])
                soqmap[Soquet(binst, reg.name)] = _qq
            elif isinstance(reg, ApplyFRegister):
                soqmap[Soquet(binst, reg.out_name)] = soqmap[Soquet(binst, reg.name)]
            else:
                pass

        # Then add it using the current mapping of soqmap to regnames
        bloq = binst.bloq
        quregs = {reg.name: soqmap[Soquet(binst, reg.name)] for reg in bloq.registers}
        try:
            op = bloq.on_registers(**quregs)
        except NoCirqEquivalent:
            op = None
    else:
        op = None

    # Finally: track name updates for successors
    for suc in binst_graph.successors(binst):
        reg_conns = binst_graph.edges[binst, suc]['conns']
        for in_regname, out_regname in reg_conns:
            soqmap[Soquet(suc, out_regname)] = soqmap[Soquet(binst, in_regname)]

    return op


def cbloq_to_cirq_circuit(
    quregs: Dict[str, Sequence[cirq.Qid]], wires: Sequence[Wire]
) -> cirq.Circuit:
    # Make a graph where we just connect binsts but note in the edges what the mappings are.
    binst_graph = nx.DiGraph()
    for wire in wires:
        # TODO: append to conns instead of replace for two connections between the same two binsts
        binst_graph.add_edge(
            wire.left.binst, wire.right.binst, conns=[(wire.left.reg_name, wire.right.reg_name)]
        )

    # A running map of soquet to qubits
    soqmap = {Soquet(LeftDangle, reg_name): qubits for reg_name, qubits in quregs.items()}

    moments = []
    for i, binsts in enumerate(nx.topological_generations(binst_graph)):
        mom = []
        for binst in binsts:
            op = _process_binst(binst, soqmap, binst_graph)
            if op:
                mom.append(op)
        if mom:
            moments.append(cirq.Moment(mom))

    return cirq.Circuit(moments)


def reg_out_name(reg: Register):
    # TODO: ???
    if isinstance(reg, ApplyFRegister):
        return reg.out_name
    return reg.name


class CompositeBloqBuilder:
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
