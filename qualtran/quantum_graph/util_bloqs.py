"""Bloqs for virtual operations and register reshaping."""

from functools import cached_property
from typing import Dict, Tuple, TYPE_CHECKING, Union

import numpy as np
import quimb.tensor as qtn
from attrs import frozen
from cirq_ft import TComplexity
from sympy import Expr

from qualtran.quantum_graph.bloq import Bloq
from qualtran.quantum_graph.classical_sim import bits_to_ints, ints_to_bits
from qualtran.quantum_graph.composite_bloq import SoquetT
from qualtran.quantum_graph.quantum_graph import BloqInstance, Soquet
from qualtran.quantum_graph.registers import Register, Side, Signature

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from qualtran.quantum_graph.cirq_conversion import CirqQuregT
    from qualtran.quantum_graph.classical_sim import ClassicalValT
    from qualtran.quantum_graph.musical_score import WireSymbol


@frozen
class Split(Bloq):
    """Split a bitsize `n` register into a length-`n` array-register.

    Attributes:
        n: The bitsize of the left register.
    """

    n: int

    @cached_property
    def registers(self) -> Signature:
        return Signature(
            [
                Register(name='split', bitsize=self.n, shape=tuple(), side=Side.LEFT),
                Register(name='split', bitsize=1, shape=(self.n,), side=Side.RIGHT),
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

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        from qualtran.quantum_graph.musical_score import directional_text_box

        if soq.reg.shape:
            text = f'[{", ".join(str(i) for i in soq.idx)}]'
            return directional_text_box(text, side=soq.reg.side)
        return directional_text_box(' ', side=soq.reg.side)


@frozen
class Join(Bloq):
    """Join a length-`n` array-register into one register of bitsize `n`.

    Attributes:
        n: The bitsize of the right register.
    """

    n: int

    @cached_property
    def registers(self) -> Signature:
        return Signature(
            [
                Register('join', bitsize=1, shape=(self.n,), side=Side.LEFT),
                Register('join', bitsize=self.n, shape=tuple(), side=Side.RIGHT),
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

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        from qualtran.quantum_graph.musical_score import directional_text_box

        if soq.reg.shape:
            text = f'[{", ".join(str(i) for i in soq.idx)}]'
            return directional_text_box(text, side=soq.reg.side)
        return directional_text_box(' ', side=soq.reg.side)


@frozen
class Allocate(Bloq):
    """Allocate an `n` bit register.

    Attributes:
          n: the bitsize of the allocated register.
    """

    n: int

    @cached_property
    def registers(self) -> Signature:
        return Signature([Register('alloc', bitsize=self.n, side=Side.RIGHT)])

    def on_classical_vals(self) -> Dict[str, int]:
        return {'alloc': 0}


@frozen
class Free(Bloq):
    """Free (i.e. de-allocate) an `n` bit register.

    Attributes:
        n: the bitsize of the register to be freed.
    """

    n: int

    @cached_property
    def registers(self) -> Signature:
        return Signature([Register('free', bitsize=self.n, side=Side.LEFT)])

    def on_classical_vals(self, free: int) -> Dict[str, 'ClassicalValT']:
        if free != 0:
            raise ValueError(f"Tried to free a non-zero register: {free}.")
        return {}


@frozen
class ArbitraryClifford(Bloq):
    """A bloq representing an arbitrary `n`-qubit clifford operation.

    In the surface code architecture, clifford operations are generally considered
    cheaper than non-clifford gates. Each clifford also has roughly the same cost independent
    of what particular operation it is doing.

    You can use this to bloq to represent an arbitrary clifford operation e.g. in bloq_counts
    resource estimates where the details are unimportant for the resource estimation task
    at hand.
    """

    n: Union[int, Expr]

    @cached_property
    def registers(self) -> Signature:
        return Signature([Register('x', bitsize=self.n)])
