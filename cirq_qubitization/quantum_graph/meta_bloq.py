import random
from functools import cached_property
from typing import Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union

import attrs
import cirq
from attrs import frozen

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import (
    _binst_to_cxns,
    CompositeBloq,
    CompositeBloqBuilder,
)
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from cirq_qubitization.quantum_graph.quantum_graph import (
    BloqInstance,
    Connection,
    DanglingT,
    LeftDangle,
    RightDangle,
    Soquet,
)


@frozen
class Atom(Bloq):
    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters.build(stuff=1)


class TestSerialBloq(Bloq):
    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters.build(stuff=1)

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', stuff: 'SoquetT'
    ) -> Dict[str, 'Soquet']:

        for i in range(3):
            (stuff,) = bb.add(Atom(), stuff=stuff)
        return {'stuff': stuff}


@frozen
class TestParallelBloq(Bloq):
    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters.build(stuff=3)

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', stuff: 'SoquetT'
    ) -> Dict[str, 'Soquet']:
        stuff = bb.split(stuff)
        for i in range(len(stuff)):
            stuff[i] = bb.add(Atom(), stuff=stuff[i])[0]

        return {'stuff': bb.join(stuff)}


@frozen
class ModRequest:
    bloq: Optional[Bloq] = None
    cxn_map: Dict[Union[None, Connection], Connection] = attrs.field(factory=dict)
    add_cxns: List[Connection] = attrs.field(factory=list)


@frozen
class _MainIterStep:
    binst: BloqInstance
    pred_cxns: Tuple[Connection]
    succ_cxns: Tuple[Connection]


class _Beginning:
    pass


class _Added:
    pass


_IterStepT = Union[_Beginning, _Added, _MainIterStep]
_Beginning = _Beginning()
_Added = _Added()


class BloqModder:
    def __init__(self, cbloq: CompositeBloq):
        self.cbloq = cbloq
        self.iter = cbloq.iter_bloqnections()
        self.this_step: _IterStepT = _Beginning

        self.cxns = []
        self.mapping_to_new = {}
        self.i = 1_000

    def add_register(self, reg: FancyRegister):
        return Soquet(LeftDangle, reg)

    def update(self, old_binst: BloqInstance, new_bloq: Bloq, **soqs: Soquet) -> Tuple[Soquet]:
        new_soqs: List[Soquet] = []
        new_cxns: List[Connection] = []
        for name, soq in soqs.items():
            # TODO: thru registers only?
            new_soq = Soquet(old_binst, reg=new_bloq.registers.get_right(name), idx=soq.idx)
            cxn = Connection(soq, new_soq)
            new_cxns.append(cxn)
            new_soqs.append(new_soq)
        self.mod(ModRequest(bloq=new_bloq, add_cxns=new_cxns))
        return tuple(new_soqs)

    def __next__(self):
        if isinstance(self.this_step, _MainIterStep):
            self.mod(ModRequest())

        binst, preds, succs = next(self.iter)
        self.this_step = _MainIterStep(binst, tuple(preds), tuple(succs))
        return binst, preds, succs

    def __iter__(self):
        if self.this_step is not _Beginning:
            raise ValueError("BloqModder iteration has already started somewhere else.")
        return self

    def map_soq(self, soq: Soquet) -> Soquet:
        if isinstance(soq.binst, DanglingT):
            return soq
        return attrs.evolve(soq, binst=self.mapping_to_new[soq.binst])

    def mod(self, mod: ModRequest):
        if not isinstance(self.this_step, _MainIterStep):
            raise ValueError(
                "You must retrieve the current binst before requesting a modification."
            )

        # part 1: bloq transform
        binst = self.this_step.binst
        if isinstance(binst, DanglingT):
            pass
        else:
            new_bloq = mod.bloq if mod.bloq is not None else binst.bloq
            new_binst = BloqInstance(new_bloq, self.i)
            self.i += 1
            self.mapping_to_new[binst] = new_binst

        # part 2: connection transform
        for cxn in self.this_step.pred_cxns:
            if cxn in mod.cxn_map:
                cxn = mod.cxn_map[cxn]
            self.cxns.append(Connection(self.map_soq(cxn.left), self.map_soq(cxn.right)))

        # part 3: additional connections
        for cxn in mod.add_cxns:
            self.cxns.append(Connection(self.map_soq(cxn.left), self.map_soq(cxn.right)))

        # part 4: remove connections?

        # bookkeep
        self.this_step = _Added

    def finalize(self, registers: FancyRegisters, **soqs: Soquet):
        if self.this_step is not _Added:
            raise ValueError("Finalize called but there are outstanding steps.")

        pred_cxns, _ = _binst_to_cxns(RightDangle, self.cbloq._binst_graph)
        for cxn in pred_cxns:
            self.cxns.append(Connection(self.map_soq(cxn.left), self.map_soq(cxn.right)))

        for name, soq in soqs.items():
            cxn = Connection(soq, Soquet(RightDangle, registers.get_right(name)))
            self.cxns.append(Connection(self.map_soq(cxn.left), self.map_soq(cxn.right)))

        return CompositeBloq(cxns=self.cxns, registers=registers)


class ControlledBloq(Bloq):
    def __init__(self, subbloq: Bloq):
        self._subbloq = subbloq

    def pretty_name(self) -> str:
        return f'C[{self._subbloq.pretty_name()}]'

    def short_name(self) -> str:
        return f'C[{self._subbloq.short_name()}]'

    def __repr__(self):
        return f'C[{repr(self._subbloq)}]'

    def __str__(self):
        return f'C[{str(self._subbloq)}]'

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters(
            [FancyRegister(name="control", bitsize=1)] + list(self._subbloq.registers)
        )

    def decompose_bloq(self) -> 'CompositeBloq':
        if not isinstance(self._subbloq, CompositeBloq):
            return ControlledBloq(self._subbloq.decompose_bloq()).decompose_bloq()

        modder = BloqModder(self._subbloq)

        # TODO: how should "add register" interact with myself.
        my_ctl_reg = self.registers[0]
        ctrl_soq = modder.add_register(FancyRegister('control', 1))

        for binst, pred_cxns, succ_cxns in modder:
            new_bloq = ControlledBloq(binst.bloq)
            (ctrl_soq,) = modder.update(binst, new_bloq, control=ctrl_soq)

        return modder.finalize(self.registers, control=ctrl_soq)
