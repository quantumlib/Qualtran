from functools import cached_property
from typing import Dict, TYPE_CHECKING

import numpy as np
import quimb.tensor as qtn
from attrs import frozen
from numpy.typing import NDArray

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side

if TYPE_CHECKING:
    import cirq

_ZERO = np.array([1, 0], dtype=np.complex128)
_ONE = np.array([0, 1], dtype=np.complex128)
_PAULIZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)


@frozen
class _ZVector(Bloq):
    """The |0> or |1> state or effect.

    Please use the explicitly named subclasses instead of the boolean arguments.

    Args:
        bit: False chooses |0>, True chooses |1>
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
        return '1' if self.bit else '0'

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
                data=_ONE if self.bit else _ZERO, inds=(side['q'],), tags=[self.short_name(), binst]
            )
        )

    def apply_classical(self, **vals: int) -> Dict[str, int]:
        """Return or consume 1 or 0 depending on `self.state` and `self.bit`.

        If `self.state`, we return a bit in the `q` register. Otherwise,
        we assert that the inputted `q` register is the correct bit.
        """
        if self.state:
            if vals:
                raise ValueError(f"Extra values to `apply_classical`: {vals}")

            if self.bit:
                return {'q': 1}
            else:
                return {'q': 0}

        q = vals.pop('q')
        if vals:
            raise ValueError(f"Extra values to `apply_classical`: {vals}")
        if self.bit:
            assert q == 1, q
        else:
            assert q == 0, q
        return {}


def _hide_base_fields(cls, fields):
    # for use in attrs `field_trasnformer`.
    return [
        field.evolve(repr=False) if field.name in ['bit', 'state'] else field for field in fields
    ]


@frozen(init=False, field_transformer=_hide_base_fields)
class ZeroState(_ZVector):
    """The state |0>"""

    def __init__(self, n: int = 1):
        self.__attrs_init__(bit=False, state=True, n=n)


@frozen(init=False, field_transformer=_hide_base_fields)
class ZeroEffect(_ZVector):
    """The effect <0|"""

    def __init__(self, n: int = 1):
        self.__attrs_init__(bit=False, state=False, n=n)


@frozen(init=False, field_transformer=_hide_base_fields)
class OneState(_ZVector):
    """The state |1>"""

    def __init__(self, n: int = 1):
        self.__attrs_init__(bit=True, state=True, n=n)


@frozen(init=False, field_transformer=_hide_base_fields)
class OneEffect(_ZVector):
    """The effect <1|"""

    def __init__(self, n: int = 1):
        self.__attrs_init__(bit=True, state=False, n=n)


@frozen
class ZGate(Bloq):
    """The Z gate.

    This causes a phase flip: Z|+> = |-> and vice-versa.
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
                data=_PAULIZ, inds=(outgoing['q'], incoming['q']), tags=[self.short_name(), binst]
            )
        )

    def on_registers(self, q: 'NDArray[cirq.Qid]') -> 'cirq.OP_TREE':
        import cirq

        (q,) = q
        return cirq.Z(q)
