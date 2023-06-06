from functools import cached_property
from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np
import quimb.tensor as qtn
from attrs import frozen

from cirq_qubitization import TComplexity
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.classical_sim import bits_to_ints, ints_to_bits
from cirq_qubitization.quantum_graph.composite_bloq import SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from cirq_qubitization.quantum_graph.quantum_graph import BloqInstance

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from cirq_qubitization.quantum_graph.cirq_conversion import CirqQuregT
    from cirq_qubitization.quantum_graph.classical_sim import ClassicalValT


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

    def as_cirq_op(
        self, qubit_manager, split: 'CirqQuregT'
    ) -> Tuple[None, Dict[str, 'CirqQuregT']]:
        return None, {'split': split.reshape((self.n, 1))}

    def t_complexity(self) -> 'TComplexity':
        return TComplexity()

    def on_classical_vals(self, split: int) -> Dict[str, 'ClassicalValT']:
        return {'split': ints_to_bits(np.array([split]), self.n)[0]}

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        binst: BloqInstance,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        tn.add(
            qtn.Tensor(
                data=np.eye(2**self.n, 2**self.n).reshape((2,) * self.n + (2**self.n,)),
                inds=outgoing['split'].tolist() + [incoming['split']],
                tags=['Split', binst],
            )
        )


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

    def as_cirq_op(self, qubit_manager, join: 'CirqQuregT') -> Tuple[None, Dict[str, 'CirqQuregT']]:
        return None, {'join': join.reshape(self.n)}

    def t_complexity(self) -> 'TComplexity':
        return TComplexity()

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        binst: BloqInstance,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        tn.add(
            qtn.Tensor(
                data=np.eye(2**self.n, 2**self.n).reshape((2,) * self.n + (2**self.n,)),
                inds=incoming['join'].tolist() + [outgoing['join']],
                tags=['Join', binst],
            )
        )

    def on_classical_vals(self, join: 'NDArray[np.uint8]') -> Dict[str, int]:
        return {'join': bits_to_ints(join)[0]}


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

    def on_classical_vals(self) -> Dict[str, int]:
        return {'alloc': 0}


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
