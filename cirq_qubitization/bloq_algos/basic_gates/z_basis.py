from functools import cached_property
from typing import Any, Dict, Tuple, TYPE_CHECKING

import attrs
import numpy as np
import quimb.tensor as qtn
from attrs import frozen

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.classical_sim import ints_to_bits
from cirq_qubitization.quantum_graph.composite_bloq import SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side

if TYPE_CHECKING:
    import cirq

    from cirq_qubitization.quantum_graph.cirq_conversion import CirqQuregT
    from cirq_qubitization.quantum_graph.classical_sim import ClassicalValT

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

    def on_classical_vals(self, **vals: int) -> Dict[str, int]:
        """Return or consume 1 or 0 depending on `self.state` and `self.bit`.

        If `self.state`, we return a bit in the `q` register. Otherwise,
        we assert that the inputted `q` register is the correct bit.
        """
        bit_int = 1 if self.bit else 0  # guard against bad `self.bit` types.
        if self.state:
            assert not vals, vals
            return {'q': bit_int}

        q = vals.pop('q')
        assert not vals, vals
        assert q == bit_int, q
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

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', q: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq

        (q,) = q
        return cirq.Z(q), {'q': [q]}


@frozen
class _IntVector(Bloq):
    """Represent a classical non-negative integer vector (state or effect).

    Args:
        val: the classical value
        bitsize: The bitsize of the register
        state: True if this is a state; an effect otherwise.

    Registers:
        val: The register of size `bitsize` which initializes the value `val`.
    """

    val: int = attrs.field()
    bitsize: int
    state: bool

    @val.validator
    def check(self, attribute, val):
        if val < 0:
            raise ValueError("`val` must be positive")

        if val >= 2**self.bitsize:
            raise ValueError(f"`val` is too big for bitsize {self.bitsize}")

    @cached_property
    def registers(self) -> FancyRegisters:
        side = Side.RIGHT if self.state else Side.LEFT
        return FancyRegisters([FancyRegister('val', bitsize=self.bitsize, side=side)])

    def build_composite_bloq(self, bb: 'CompositeBloqBuilder') -> Dict[str, 'SoquetT']:
        states = [ZeroState(), OneState()]
        xs = []
        for bit in ints_to_bits(np.array([self.val]), w=self.bitsize)[0]:
            (x,) = bb.add(states[bit])
            xs.append(x)
        xs = np.array(xs)

        return {'val': bb.join(xs)}

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        tag: Any,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):
        data = np.zeros(2**self.bitsize).reshape((2,) * self.bitsize)
        bitstring = ints_to_bits(np.array([self.val]), w=self.bitsize)[0]
        data[tuple(bitstring)] = 1
        data = data.reshape(-1)

        if self.state:
            inds = (outgoing['val'],)
        else:
            inds = (incoming['val'],)

        tn.add(qtn.Tensor(data=data, inds=inds, tags=[self.short_name(), tag]))

    def on_classical_vals(self, **vals: 'ClassicalValT') -> Dict[str, int]:
        if self.state:
            assert not vals
            return {'val': self.val}

        assert vals['val'] == self.val, vals['val']

    def short_name(self) -> str:
        return f'{self.val}'

    def pretty_name(self) -> str:
        s = self.short_name()
        return f'|{s}>' if self.state else f'<{s}|'


@frozen(init=False, field_transformer=_hide_base_fields)
class IntState(_IntVector):
    """The state |val> for non-negative integer val

    Args:
        val: the classical value
        bitsize: The bitsize of the register

    Registers:
        val: The register of size `bitsize` which initializes the value `val`.
    """

    def __init__(self, val: int, bitsize: int):
        self.__attrs_init__(val=val, bitsize=bitsize, state=True)


@frozen(init=False, field_transformer=_hide_base_fields)
class IntEffect(_IntVector):
    """The effect <val| for non-negative integer val

    Args:
        val: the classical value
        bitsize: The bitsize of the register

    Registers:
        val: The register of size `bitsize` which de-allocates the value `val`.
    """

    def __init__(self, val: int, bitsize: int):
        self.__attrs_init__(val=val, bitsize=bitsize, state=False)
