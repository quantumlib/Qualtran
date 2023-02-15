from functools import cached_property
from typing import Dict, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np
import quimb.tensor as qtn
from attrs import field, frozen

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from cirq_qubitization.quantum_graph.quantum_graph import BloqInstance, Soquet

if TYPE_CHECKING:
    import cirq


@frozen
class Split(Bloq):
    """Split a bitsize `n` register into a length-`n` array-register.

    Args:
        n: The bitsize of the left register.
    """

    n: int

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters(
            [
                FancyRegister(name='split', bitsize=self.n, wireshape=tuple(), side=Side.LEFT),
                FancyRegister(name='split', bitsize=1, wireshape=(self.n,), side=Side.RIGHT),
            ]
        )

    def on_registers(
        self, **qubit_regs: Union['cirq.Qid', Sequence['cirq.Qid']]
    ) -> 'cirq.GateOperation':
        return None


@frozen
class Join(Bloq):
    """Join a length-`n` array-register into one register of bitsize `n`.

    Args:
        n: The bitsize of the right register.
    """

    n: int

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters(
            [
                FancyRegister('join', bitsize=1, wireshape=(self.n,), side=Side.LEFT),
                FancyRegister('join', bitsize=self.n, wireshape=tuple(), side=Side.RIGHT),
            ]
        )

    def on_registers(
        self, **qubit_regs: Union['cirq.Qid', Sequence['cirq.Qid']]
    ) -> 'cirq.GateOperation':
        return None

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        binst: BloqInstance,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        assert sorted(incoming.keys()) == ['join']
        in_soqs = incoming['join']
        assert in_soqs.shape == (self.n,)

        assert sorted(outgoing.keys()) == ['join']
        out_soq = outgoing['join']
        assert isinstance(out_soq, Soquet)

        tn.add(
            qtn.Tensor(
                data=np.eye(2**self.n, 2**self.n).reshape((2,) * self.n + (2**self.n,)),
                inds=in_soqs.tolist() + [out_soq],
                tags=['Join', binst],
            )
        )


@frozen
class Allocate(Bloq):
    """Allocate an `n` bit register.

    Args:
          n: the bitsize of the allocated register.
    """

    n: int

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters([FancyRegister('alloc', bitsize=self.n, side=Side.RIGHT)])


@frozen
class Free(Bloq):
    """Free (i.e. de-allocate) an `n` bit register.

    Args:
        n: the bitsize of the register to be freed.
    """

    n: int

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters([FancyRegister('free', bitsize=self.n, side=Side.LEFT)])


@frozen
class Partition(Bloq):
    """Partition one register into a number of new registers of uneven size.

    Args:
        sizes: Create one right register per size of the given bitsize.
    """

    sizes: Tuple[int, ...] = field(converter=tuple)

    @cached_property
    def registers(self) -> FancyRegisters:
        tot = sum(self.sizes)
        return FancyRegisters(
            [FancyRegister(name='x', bitsize=tot, wireshape=tuple(), side=Side.LEFT)]
            + [
                FancyRegister(name=f'y{i}', bitsize=n, wireshape=tuple(), side=Side.RIGHT)
                for i, n in enumerate(self.sizes)
            ]
        )

    def on_registers(
        self, **qubit_regs: Union['cirq.Qid', Sequence['cirq.Qid']]
    ) -> 'cirq.GateOperation':
        return None


@frozen
class Unpartition(Bloq):
    """Take a number of registers of uneven size and combine into one register.

    Args:
        sizes: Input one left regsiter per size of the given bitsize. The output register
            will be of bitsize `sum(sizes)`.
    """

    sizes: Tuple[int, ...]

    @cached_property
    def registers(self) -> FancyRegisters:
        tot = sum(self.sizes)
        return FancyRegisters(
            [
                FancyRegister(name=f'y{i}', bitsize=n, wireshape=tuple(), side=Side.LEFT)
                for i, n in enumerate(self.sizes)
            ]
            + [FancyRegister(name='x', bitsize=tot, wireshape=tuple(), side=Side.RIGHT)]
        )

    def on_registers(
        self, **qubit_regs: Union['cirq.Qid', Sequence['cirq.Qid']]
    ) -> 'cirq.GateOperation':
        return None
