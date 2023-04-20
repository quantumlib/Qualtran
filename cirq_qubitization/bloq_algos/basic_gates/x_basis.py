from functools import cached_property
from typing import Dict, TYPE_CHECKING

import numpy as np
import quimb.tensor as qtn
from attrs import frozen
from numpy.typing import NDArray

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.classical_sim import ClassicalValT
from cirq_qubitization.quantum_graph.composite_bloq import SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side

if TYPE_CHECKING:
    import cirq

_PLUS = np.ones(2, dtype=np.complex128) / np.sqrt(2)
_MINUS = np.array([1, -1], dtype=np.complex128) / np.sqrt(2)
_PAULIX = np.array([[0, 1], [1, 0]], dtype=np.complex128)


@frozen
class _XVector(Bloq):
    """The |+> or |-> state or effect.

    Please use the explicitly named subclasses instead of the boolean arguments.

    Args:
        bit: False chooses |+>, True chooses |->
        state: True means this is a state with right registers; False means this is an
            effect with left registers.
        n: bitsize of the vector.

    """

    bit: bool
    state: bool = True
    n: int = 1

    def __attrs_post_init__(self):
        if self.n != 1:
            raise NotImplementedError("Come back later.")

    def pretty_name(self) -> str:
        s = self.short_name()
        return f'|{s}>' if self.state else f'<{s}|'

    def short_name(self) -> str:
        return '-' if self.bit else '+'

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters(
            [FancyRegister('q', bitsize=1, side=Side.RIGHT if self.state else Side.LEFT)]
        )

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        binst,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):
        side = outgoing if self.state else incoming
        tn.add(
            qtn.Tensor(
                data=_MINUS if self.bit else _PLUS,
                inds=(side['q'],),
                tags=[self.short_name(), binst],
            )
        )


def _hide_base_fields(cls, fields):
    # for use in attrs `field_trasnformer`.
    return [
        field.evolve(repr=False) if field.name in ['bit', 'state'] else field for field in fields
    ]


@frozen(init=False, field_transformer=_hide_base_fields)
class PlusState(_XVector):
    """The state |+>"""

    def __init__(self, n: int = 1):
        self.__attrs_init__(bit=False, state=True, n=n)


@frozen(init=False, field_transformer=_hide_base_fields)
class PlusEffect(_XVector):
    """The effect <+|"""

    def __init__(self, n: int = 1):
        self.__attrs_init__(bit=False, state=False, n=n)


@frozen(init=False, field_transformer=_hide_base_fields)
class MinusState(_XVector):
    """The state |->"""

    def __init__(self, n: int = 1):
        self.__attrs_init__(bit=True, state=True, n=n)


@frozen(init=False, field_transformer=_hide_base_fields)
class MinusEffect(_XVector):
    """The effect <-|"""

    def __init__(self, n: int = 1):
        self.__attrs_init__(bit=True, state=False, n=n)


@frozen
class XGate(Bloq):
    """The Pauli X gate.

    This causes a bit flip: X|0> = |1> and vice-versa.
    """

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters.build(q=1)

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        binst,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):
        tn.add(
            qtn.Tensor(
                data=_PAULIX, inds=(outgoing['q'], incoming['q']), tags=[self.short_name(), binst]
            )
        )

    def on_classical_vals(self, q: int) -> Dict[str, 'ClassicalValT']:
        return {'q': (q + 1) % 2}

    def as_cirq_op(self, cirq_quregs: Dict[str, 'NDArray[cirq.Qid]']) -> 'cirq.Operation':
        import cirq

        (q,) = cirq_quregs['q']
        return cirq.X(q)
