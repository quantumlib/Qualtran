#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import abc
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from typing import Optional, TYPE_CHECKING, Set, Union

import numpy as np
import sympy
from attrs import frozen

from qualtran import (
    AddControlledT,
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    CBit,
    CompositeBloq,
    ConnectionT,
    CtrlSpec,
    DecomposeTypeError,
    QBit,
    QUInt,
    QVar,
    Register,
    Side,
    Signature,
    SoquetT,
)
from qualtran.drawing import Circle, directional_text_box, Text, TextBox, WireSymbol
from qualtran.symbolics import SymbolicInt

if TYPE_CHECKING:
    import cirq
    import quimb.tensor as qtn
    from pennylane.operation import Operation
    from pennylane.wires import Wires

    from qualtran.cirq_interop import CirqQuregT
    from qualtran.resource_counting import BloqCountDictT, BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValRetT, ClassicalValT

_ZERO = np.array([1, 0], dtype=np.complex128)
_ONE = np.array([0, 1], dtype=np.complex128)
_PAULIZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)


class _ZVector(Bloq, metaclass=abc.ABCMeta):
    """The |0> or |1> state or effect.

    Please use the explicitly named subclasses instead of the boolean arguments.

    Args:
        bit: False chooses |0>, True chooses |1>
        state: True means this is a state with right registers; False means this is an
            effect with left registers.
        n: bitsize of the vector.

    """

    @property
    @abc.abstractmethod
    def bit(self) -> bool: ...

    @property
    @abc.abstractmethod
    def state(self) -> bool: ...

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('q', QBit(), side=Side.RIGHT if self.state else Side.LEFT)])

    def decompose_bloq(self) -> CompositeBloq:
        raise DecomposeTypeError(f"{self} is atomic")

    def my_tensors(
        self, incoming: dict[str, 'ConnectionT'], outgoing: dict[str, 'ConnectionT']
    ) -> list['qtn.Tensor']:
        import quimb.tensor as qtn

        side = outgoing if self.state else incoming
        return [
            qtn.Tensor(data=_ONE if self.bit else _ZERO, inds=[(side['q'], 0)], tags=[str(self)])
        ]

    def on_classical_vals(self, *, q: int | None = None) -> dict[str, int]:
        """Return or consume 1 or 0 depending on `self.state` and `self.bit`.

        If `self.state`, we return a bit in the `q` register. Otherwise,
        we assert that the inputted `q` register is the correct bit.
        """
        bit_int = 1 if self.bit else 0  # guard against bad `self.bit` types.
        if self.state:
            assert q is None
            return {'q': bit_int}

        assert q == bit_int, q
        return {}

    def as_cirq_op(
        self,
        qubit_manager: 'cirq.QubitManager',
        **cirq_quregs: 'CirqQuregT',  # type: ignore[type-var]
    ) -> tuple[Optional['cirq.Operation'], dict[str, 'CirqQuregT']]:  # type: ignore[type-var]
        if not self.state:
            raise ValueError(f"There is no Cirq equivalent for {self}")

        import cirq

        (q,) = qubit_manager.qalloc(1)

        if self.bit:
            op = cirq.X(q)
        else:
            op = None
        return op, {'q': np.array([q])}

    def __str__(self) -> str:
        s = '1' if self.bit else '0'
        return f'|{s}>' if self.state else f'<{s}|'

    def wire_symbol(
        self, reg: Optional['Register'], idx: tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text('')
        s = '1' if self.bit else '0'
        return directional_text_box(s, side=reg.side)


@frozen
class ZeroState(_ZVector):
    """The state |0>"""

    @property
    def bit(self) -> bool:
        return False

    @property
    def state(self) -> bool:
        return True

    def adjoint(self) -> 'Bloq':
        return ZeroEffect()

    def __str__(self):
        return 'ZeroState'


@bloq_example
def _zero_state() -> ZeroState:
    zero_state = ZeroState()
    return zero_state


_ZERO_STATE_DOC = BloqDocSpec(bloq_cls=ZeroState, examples=[_zero_state])


@frozen
class ZeroEffect(_ZVector):
    """The effect <0|"""

    @property
    def bit(self) -> bool:
        return False

    @property
    def state(self) -> bool:
        return False

    def adjoint(self) -> 'Bloq':
        return ZeroState()

    def __str__(self):
        return 'ZeroEffect'


@bloq_example
def _zero_effect() -> ZeroEffect:
    zero_effect = ZeroEffect()
    return zero_effect


_ZERO_EFFECT_DOC = BloqDocSpec(bloq_cls=ZeroEffect, examples=[_zero_effect])


@frozen
class OneState(_ZVector):
    """The state |1>"""

    @property
    def bit(self) -> bool:
        return True

    @property
    def state(self) -> bool:
        return True

    def adjoint(self) -> 'Bloq':
        return OneEffect()

    def __str__(self):
        return 'OneState'


@bloq_example
def _one_state() -> OneState:
    one_state = OneState()
    return one_state


_ONE_STATE_DOC = BloqDocSpec(bloq_cls=OneState, examples=[_one_state])


@frozen
class OneEffect(_ZVector):
    """The effect <1|"""

    @property
    def bit(self) -> bool:
        return True

    @property
    def state(self) -> bool:
        return False

    def adjoint(self) -> 'Bloq':
        return OneState()

    def __str__(self):
        return 'OneEffect'


@bloq_example
def _one_effect() -> OneEffect:
    one_effect = OneEffect()
    return one_effect


_ONE_EFFECT_DOC = BloqDocSpec(bloq_cls=OneEffect, examples=[_one_effect])


@frozen
class ZGate(Bloq):
    """The Z gate.

    This causes a phase flip: Z|+> = |-> and vice-versa.
    """

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(q=1)

    def adjoint(self) -> 'Bloq':
        return self

    @classmethod
    def qcall(cls, q: 'QVar'):
        return q.bb.add(cls(), q=q)

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    def my_tensors(
        self, incoming: dict[str, 'ConnectionT'], outgoing: dict[str, 'ConnectionT']
    ) -> list['qtn.Tensor']:
        import quimb.tensor as qtn

        return [
            qtn.Tensor(
                data=_PAULIZ, inds=[(outgoing['q'], 0), (incoming['q'], 0)], tags=[str(self)]
            )
        ]

    def on_classical_vals(self, **vals: 'ClassicalValT') -> dict[str, 'ClassicalValT']:
        # Diagonal, but causes phases: see `basis_state_phase`
        return vals

    def basis_state_phase(self, q: int) -> Optional[complex]:
        if q == 1:
            return -1
        return 1

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> tuple['Bloq', 'AddControlledT']:
        if ctrl_spec != CtrlSpec():
            # Delegate to the general superclass behavior
            return super().get_ctrl_system(ctrl_spec=ctrl_spec)

        bloq = CZ()

        def add_controlled(
            bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: dict[str, 'SoquetT']
        ) -> tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
            (ctrl_soq,) = ctrl_soqs
            ctrl_soq, q2 = bb.add(bloq, q1=ctrl_soq, q2=in_soqs['q'])
            return (ctrl_soq,), (q2,)

        return bloq, add_controlled

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', q: 'CirqQuregT'
    ) -> tuple['cirq.Operation', dict[str, 'CirqQuregT']]:
        import cirq

        (q,) = q
        return cirq.Z(q), {'q': np.asarray([q])}

    def as_pl_op(self, wires: 'Wires') -> 'Operation':
        import pennylane as qml

        return qml.Z(wires=wires)

    def wire_symbol(
        self, reg: Optional['Register'], idx: tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text('')

        return TextBox('Z')


@bloq_example
def _zgate() -> ZGate:
    zgate = ZGate()
    return zgate


_Z_GATE_DOC = BloqDocSpec(bloq_cls=ZGate, examples=[_zgate], call_graph_example=None)


@frozen
class CZ(Bloq):
    """Two-qubit controlled-Z gate.

    Registers:
        q1: One-bit control register.
        q2: One-bit target register.
    """

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(q1=1, q2=1)

    @classmethod
    def qcall(cls, q1: "QVar", q2: "QVar"):
        bb = q1.bb
        return bb.add(cls(), q1=q1, q2=q2)

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    def adjoint(self) -> 'Bloq':
        return self

    def my_tensors(
        self, incoming: dict[str, 'ConnectionT'], outgoing: dict[str, 'ConnectionT']
    ) -> list['qtn.Tensor']:
        import quimb.tensor as qtn

        unitary = np.diag(np.array([1, 1, 1, -1], dtype=np.complex128)).reshape((2, 2, 2, 2))
        inds = [(outgoing['q1'], 0), (outgoing['q2'], 0), (incoming['q1'], 0), (incoming['q2'], 0)]

        return [qtn.Tensor(data=unitary, inds=inds, tags=[str(self)])]

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', q1: 'CirqQuregT', q2: 'CirqQuregT'
    ) -> tuple['cirq.Operation', dict[str, 'CirqQuregT']]:
        import cirq

        (q1,) = q1
        (q2,) = q2
        return cirq.CZ(q1, q2), {'q1': np.array([q1]), 'q2': np.array([q2])}

    def as_pl_op(self, wires: 'Wires') -> 'Operation':
        import pennylane as qml

        return qml.CZ(wires=wires)

    def wire_symbol(self, reg: Optional[Register], idx: tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        if reg.name == 'q1' or reg.name == 'q2':
            return Circle()
        raise ValueError(f'Unknown wire symbol register name: {reg.name}')

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> tuple['Bloq', 'AddControlledT']:
        from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs

        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=1, bloq_with_ctrl=self, ctrl_reg_name='q1'
        )

    def on_classical_vals(self, **vals: 'ClassicalValT') -> dict[str, 'ClassicalValT']:
        # Diagonal, but causes phases: see `basis_state_phase`
        return vals

    def basis_state_phase(self, q1: int, q2: int) -> Optional[complex]:
        if q1 == 1 and q2 == 1:
            return -1
        return 1


@bloq_example
def _cz() -> CZ:
    cz = CZ()
    return cz


_CZ_DOC = BloqDocSpec(bloq_cls=CZ, examples=[_cz], call_graph_example=None)


@frozen
class MeasureZ(Bloq):
    """Measure a qubit in the Z basis.

    Registers:
        q [LEFT]: The qubit to measure.
        c [RIGHT]: The classical measurement result.
    """

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [Register('q', QBit(), side=Side.LEFT), Register('c', CBit(), side=Side.RIGHT)]
        )

    def on_classical_vals(self, q: int) -> Mapping[str, 'ClassicalValRetT']:
        return {'c': q}

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic.")

    @classmethod
    def qcall(cls, q: 'QVar') -> 'QVar':
        return q.bb.add(cls(), q=q)

    def my_tensors(
        self, incoming: dict[str, 'ConnectionT'], outgoing: dict[str, 'ConnectionT']
    ) -> list['qtn.Tensor']:
        import quimb.tensor as qtn

        from qualtran.simulation.tensor import DiscardInd

        copy = np.zeros((2, 2, 2), dtype=np.complex128)
        copy[0, 0, 0] = 1
        copy[1, 1, 1] = 1
        # Tie together q, c, and meas_result with the copy tensor; throw out one of the legs.
        meas_result = qtn.rand_uuid('meas_result')
        t = qtn.Tensor(
            data=copy,
            inds=[(incoming['q'], 0), (outgoing['c'], 0), (meas_result, 0)],
            tags=[str(self)],
        )
        return [t, DiscardInd((meas_result, 0))]

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', **cirq_quregs: 'CirqQuregT'  # type: ignore[type-var]
    ) -> tuple[Optional['cirq.Operation'], dict[str, 'CirqQuregT']]:  # type: ignore[type-var]

        import cirq

        (q,) = cirq_quregs['q']
        return cirq.measure(q), {}


@bloq_example
def _meas_z() -> MeasureZ:
    meas_z = MeasureZ()
    return meas_z


_MEASURE_Z_DOC = BloqDocSpec(bloq_cls=MeasureZ, examples=[_meas_z])


@frozen
class IntState(Bloq):
    """The state |val> for non-negative (unsigned) integer val

    Please prefer QUIntState for new code, which makes it clear that the resultant
    datatype is a `QUInt` unsigned integer.

    Args:
        val: the classical value
        bitsize: The bitsize of the register

    Registers:
        val: The register of size `bitsize` which initializes the value `val`.
    """

    val: SymbolicInt
    bitsize: SymbolicInt

    def __attrs_post_init__(self):
        import warnings

        warnings.warn(
            "IntState will be deprecated. Use QUIntState instead.",
            PendingDeprecationWarning,
            stacklevel=2,
        )

    @cached_property
    def _impl(self):
        from qualtran.bloqs.basic_gates.qconst import _QConst

        return _QConst(val=self.val, qdtype=QUInt(self.bitsize), state=True)

    @property
    def signature(self) -> 'Signature':
        return self._impl.signature

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> dict[str, 'SoquetT']:
        return self._impl.build_composite_bloq(bb, **soqs)

    def on_classical_vals(
        self, **vals: Union['sympy.Symbol', 'ClassicalValT']
    ) -> Mapping[str, 'ClassicalValRetT']:
        return self._impl.on_classical_vals(**vals)

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return self._impl.build_call_graph(ssa)

    def wire_symbol(
        self, reg: Optional['Register'], idx: tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        return self._impl.wire_symbol(reg, idx)

    def __str__(self) -> str:
        return f'|{self.val}>'


@bloq_example
def _int_state() -> IntState:
    int_state = IntState(55, bitsize=8)
    return int_state


_INT_STATE_DOC = BloqDocSpec(bloq_cls=IntState, examples=[_int_state])


@frozen
class IntEffect(Bloq):
    """The effect <val| for non-negative integer val

    Please prefer QUIntEffect for new code, which makes it clear that the incoming
    datatype is a `QUInt` unsigned integer.

    Args:
        val: the classical value
        bitsize: The bitsize of the register

    Registers:
        val: The register of size `bitsize` which de-allocates the value `val`.
    """

    val: SymbolicInt
    bitsize: SymbolicInt

    def __attrs_post_init__(self):
        import warnings

        warnings.warn(
            "IntEffect is deprecated. Use QUIntEffect instead.",
            PendingDeprecationWarning,
            stacklevel=2,
        )

    @cached_property
    def _impl(self):
        from qualtran.bloqs.basic_gates.qconst import _QConst

        return _QConst(val=self.val, qdtype=QUInt(self.bitsize), state=False)

    @property
    def signature(self) -> 'Signature':
        return self._impl.signature

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> dict[str, 'SoquetT']:
        return self._impl.build_composite_bloq(bb, **soqs)

    def on_classical_vals(
        self, **vals: Union['sympy.Symbol', 'ClassicalValT']
    ) -> Mapping[str, 'ClassicalValRetT']:
        return self._impl.on_classical_vals(**vals)

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return self._impl.build_call_graph(ssa)

    def wire_symbol(
        self, reg: Optional['Register'], idx: tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        return self._impl.wire_symbol(reg, idx)

    def __str__(self) -> str:
        return f'<{self.val}|'


@bloq_example
def _int_effect() -> IntEffect:
    int_effect = IntEffect(55, bitsize=8)
    return int_effect


_INT_EFFECT_DOC = BloqDocSpec(bloq_cls=IntEffect, examples=[_int_effect])
