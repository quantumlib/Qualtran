import abc
from functools import cached_property
from typing import Dict, List

import attrs
from attrs import frozen

from qualtran import Bloq, Register, Signature, SoquetT
from qualtran.bloqs.util_bloqs import Partition


class Prepare(Bloq, metaclass=abc.ABCMeta):
    """Interface for prepare operations.

    This interface specifies which registers are used as selection registers.
    """

    @property
    @abc.abstractmethod
    def selection_registers(self) -> List[Register]:
        ...

    @property
    @abc.abstractmethod
    def junk_registers(self) -> List[Register]:
        ...

    @cached_property
    def signature(self) -> Signature:
        return Signature([*self.selection_registers, *self.junk_registers])

    @abc.abstractmethod
    def dagger(self) -> 'Bloq':
        ...


@frozen
class DummyPrepare(Prepare):
    """An annonymous prepare operation used for testing and demonstration.

    The registers are a charicature of a second-quantized chemistry hamiltonian selection.

    Args:
        n: bitsize of the `p` and `q` registers.
        adjoint: Whether this is the adjoint preparation.

    Registers:
     - p: A `n` bitsize register styled after a chemistry orbital index.
     - q: A `n` bitsize register styled after a chemistry orbital index.
     - spin: A bit styled after a chemistry spin index.
    """

    n: int = 32
    adjoint: bool = False

    @property
    def selection_registers(self) -> List[Register]:
        return list(Signature.build(p=self.n, q=self.n, spin=1))

    @property
    def junk_registers(self) -> List[Register]:
        return []

    def dagger(self) -> 'Bloq':
        return attrs.evolve(self, adjoint=not self.adjoint)


@frozen
class BlackBoxPrepare(Bloq):
    """Provide a black-box interface to `Prepare` bloqs.

    This wrapper uses `Partition` to combine descriptive selection
    registers into one register named "selection".

    Args:
        prepare: The bloq following the `Prepare` interface to wrap.
        adjoint: Whether this is the adjoint preparation.
    """

    prepare: Prepare
    adjoint: bool = False

    @cached_property
    def bitsize(self):
        return sum(reg.total_bits() for reg in self.prepare.signature)

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('selection', self.bitsize)])

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', selection: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        regs = tuple(self.prepare.signature)
        partition = Partition(n=self.bitsize, regs=regs)

        sel_parts = bb.add(partition, x=selection)
        sel_parts = bb.add(self.prepare, **{reg.name: sp for reg, sp in zip(regs, sel_parts)})
        (selection,) = bb.add(
            partition.dagger(), **{reg.name: sp for reg, sp in zip(regs, sel_parts)}
        )
        return {'selection': selection}

    def dagger(self) -> 'BlackBoxPrepare':
        return attrs.evolve(self, adjoint=not self.adjoint)

    def short_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f'Prep{dag}'
