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
from functools import cached_property
from typing import Any, Dict, Iterable, List, Protocol, Sequence, Set, Tuple, TYPE_CHECKING, Union

import attrs
import numpy as np
from numpy.typing import NDArray

from .bloq import Bloq
from .data_types import QBit, QDType
from .registers import Register, Side, Signature

if TYPE_CHECKING:
    from qualtran import BloqBuilder, CompositeBloq, Soquet, SoquetT
    from qualtran.drawing import WireSymbol
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


def _cvs_convert(
    cvs: Union[int, Sequence[int], Sequence[Sequence[int]]]
) -> Tuple[NDArray[int], ...]:
    if isinstance(cvs, (int, np.integer)):
        return (np.array(cvs),)
    if isinstance(cvs, np.ndarray):
        return (cvs,)
    if all(isinstance(cv, (int, np.integer)) for cv in cvs):
        return (np.asarray(cvs),)
    return tuple(np.asarray(cv) for cv in cvs)


@attrs.frozen(eq=False)
class CtrlSpec:
    """A specification for how to control a bloq.

    This class can be used by controlled bloqs to specify the condition under which the bloq
    is active.

    In the simplest form, a controlled gate is active when the control input is one qubit of data,
    and it's in the |1> state. Otherwise, the gate is not performed. This corresponds to the
    following two equivalent CtrlSpecs:

        CtrlSpec()
        CtrlSpec(qdtypes=QBit(), cvs=1)

    This class supports additional control specifications:
     1. 'negative' controls where the bloq is active if the input is |0>.
     2. integer-equality controls where a QInt input must match an integer control value.
     3. ndarrays of control values, where the bloq is active if **all** inputs are active.
     4. Multiple control registers, control values for each of which can be specified
        using 1-3 above.

    For example:
    1. `CtrlSpec(qdtypes=QUInt(4), cvs=0b0101)`:
            Ctrl for a single register, of type `QUInt(4)` and shape `()`, is active when the
            soquet of the input register takes value 5.
    2. `CtrlSpec(cvs=[0, 1, 1, 0])`:
            Ctrl for a single register, of type `QBit()` and shape `(4,)`, is active when soquets
            of input register take values `[0, 1, 1, 0]`.
    3. `CtrlSpec(qdtypes=[QBit(), QBit()], cvs=[[0, 1], [1, 0]]).is_active([0, 1], [1, 0])`:
            Ctrl for 2 registers, each of type `QBit()` and shape `(2,)`, is active when the
            soquet for each register takes values `[0, 1]` and  `[1, 0]` respectively.

    This `CtrlSpec` class represents control values as a single `AND` clause, that applies
    simultaneously to all control registers. The functions The methods `activation_function_dtypes`
    and `is_active` are defined for future extensibility; such that one could define a ctrl spec
    that can represent more general control values as a union of `AND` clauses where the control
    is active if any of the `AND` clause is satisfied.

    Args:
        qdtypes: The quantum data type of the control input.
        cvs: The control value(s). If more than one value is provided, they must all be
            compatible with `qdtype` and the bloq is implied to be active if **all** inputs
            are active.
    """

    qdtypes: Tuple[QDType, ...] = attrs.field(
        default=QBit(), converter=lambda qt: (qt,) if isinstance(qt, QDType) else tuple(qt)
    )
    cvs: Tuple[NDArray[int], ...] = attrs.field(default=1, converter=_cvs_convert)

    def __attrs_post_init__(self):
        assert len(self.qdtypes) == len(self.cvs)

    @cached_property
    def num_ctrl_reg(self) -> int:
        return len(self.qdtypes)

    @cached_property
    def shapes(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(cv.shape for cv in self.cvs)

    def activation_function_dtypes(self) -> Sequence[Tuple[QDType, Tuple[int, ...]]]:
        """The data types that serve as input to the 'activation function'.

        The activation function takes in (quantum) inputs of these types and shapes and determines
        whether the bloq should be active. This method is useful for setting up appropriate
        control registers for a ControlledBloq.

        Returns:
            A sequence of (type, shape) tuples analogous to the arguments to `Register`.
        """
        return [(qdtype, cv.shape) for qdtype, cv in zip(self.qdtypes, self.cvs)]

    def is_active(self, *vals: 'ClassicalValT') -> bool:
        """A classical implementation of the 'activation function'.

        The activation function takes in (quantum) data and determines whether
        the bloq should be active. This method captures the same behavior on specific classical
        values representing computational basis states.

        This implementation evaluates to `True` if all the values match `self.cvs`.

        Args:
            *vals: The classical values (that fit within the types given by
                `activation_function_dtypes`) on which we evaluate whether the spec is active.

        Returns:
            True if the specific input values evaluate to `True` for this CtrlSpec.
        """
        if len(vals) != self.num_ctrl_reg:
            raise ValueError(f"Incorrect number of inputs for {self}: {len(vals)}.")

        for val, cv in zip(vals, self.cvs):
            if isinstance(val, (int, np.integer)):
                val = np.array(val)
            else:
                val = np.asarray(val)
            if val.shape != cv.shape:
                raise ValueError(f"Incorrect input shape for {self}: {val.shape} != {cv.shape}.")
            if np.any(val != cv):
                return False
        return True

    def wire_symbol(self, i: int, soq: 'Soquet') -> 'WireSymbol':
        # Return a circle for bits; a box otherwise.
        from qualtran.drawing import Circle, TextBox

        if soq.reg.bitsize == 1:
            cv = self.cvs[i][soq.idx]
            return Circle(filled=(cv == 1))

        cv = self.cvs[i][soq.idx]
        return TextBox(f'{cv}')

    @cached_property
    def _cvs_tuple(self) -> Tuple[int, ...]:
        return tuple(cv for cvs in self.cvs for cv in tuple(cvs.reshape(-1)))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CtrlSpec):
            return False

        return (
            other.qdtypes == self.qdtypes
            and other.shapes == self.shapes
            and other._cvs_tuple == self._cvs_tuple
        )

    def __hash__(self):
        return hash((self.qdtypes, self.shapes, self._cvs_tuple))


class AddControlledT(Protocol):
    """The signature for the `add_controlled` callback part of `ctrl_system`.

    See `Bloq.get_ctrl_system` for details.

    Args:
        bb: A bloq builder to use for adding.
        ctrl_soqs: The soquets that represent the control lines. These must be compatible with
            the ControlSpec; specifically with the control registers implied
            by `activation_function_dtypes`.
        in_soqs: The soquets that plug in to the normal, uncontrolled bloq.

    Returns:
        ctrl_soqs: The output control soquets.
        out_soqs: The output soquets from the uncontrolled bloq.
    """

    def __call__(
        self, bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: Dict[str, 'SoquetT']
    ) -> Tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
        ...


def _get_nice_ctrl_reg_names(reg_names: List[str], n: int) -> Tuple[str, ...]:
    """Get `n` names for the ctrl registers that don't overlap with (existing) `reg_names`."""
    if n == 1 and 'ctrl' not in reg_names:
        # Special case for nicer register name if we just have one control register
        # and can safely name it 'ctrl'.
        return ('ctrl',)

    if 'ctrl' in reg_names:
        i = 1
    else:
        i = 0
    names = []
    while len(names) < n:
        while True:
            i += 1
            candidate = f'ctrl{i}'
            if candidate not in reg_names:
                names.append(candidate)
                break
    return tuple(names)


@attrs.frozen
class Controlled(Bloq):
    """A controlled version of `subbloq`.

    This meta-bloq is part of the 'controlled' protocol. As a default fallback,
    we wrap any bloq without a custom controlled version in this meta-bloq.

    Users should likely not use this class directly. Prefer using `bloq.controlled(ctrl_spec)`,
    which may return a tailored Bloq that is controlled in the desired way.

    Args:
        subbloq: The bloq we are controlling.
        ctrl_spec: The specification for how to control the bloq.
    """

    subbloq: 'Bloq'
    ctrl_spec: 'CtrlSpec'

    @classmethod
    def make_ctrl_system(cls, bloq: 'Bloq', ctrl_spec: 'CtrlSpec') -> Tuple[Bloq, AddControlledT]:
        """A factory method for creating both the Controlled and the adder function.

        See `Bloq.get_ctrl_system`.
        """
        cb = cls(subbloq=bloq, ctrl_spec=ctrl_spec)
        ctrl_reg_names = cb.ctrl_reg_names

        def add_controlled(
            bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: Dict[str, 'SoquetT']
        ) -> Tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
            in_soqs |= dict(zip(ctrl_reg_names, ctrl_soqs))
            new_out_d = bb.add_d(cb, **in_soqs)

            # We need to pluck out the `ctrl_soq` from the new out_soqs
            # dictionary, so we can correspond it to `old_out_soqs`.
            ctrl_soqs = tuple(new_out_d.pop(creg_name) for creg_name in ctrl_reg_names)
            return ctrl_soqs, new_out_d.values()

        return cb, add_controlled

    @cached_property
    def ctrl_reg_names(self) -> Sequence[str]:
        """The name of the control registers.

        This is generated on-the-fly to avoid conflicts with existing register
        names. Users should not rely on the absolute value of this property staying constant.
        """
        reg_names = [reg.name for reg in self.subbloq.signature]
        n = len(self.ctrl_spec.activation_function_dtypes())
        return _get_nice_ctrl_reg_names(reg_names, n)

    @cached_property
    def signature(self) -> 'Signature':
        # Prepend register(s) corresponding to `ctrl_spec`.
        ctrl_regs = tuple(
            Register(name=self.ctrl_reg_names[i], dtype=qdtype, shape=shape, side=Side.THRU)
            for i, (qdtype, shape) in enumerate(self.ctrl_spec.activation_function_dtypes())
        )
        return Signature(ctrl_regs + tuple(self.subbloq.signature))

    def decompose_bloq(self) -> 'CompositeBloq':
        # Use subbloq's decomposition but wire up the additional ctrl_soqs.
        from qualtran import BloqBuilder, CompositeBloq

        if isinstance(self.subbloq, CompositeBloq):
            cbloq = self.subbloq
        else:
            cbloq = self.subbloq.decompose_bloq()

        bb, initial_soqs = BloqBuilder.from_signature(self.signature)
        ctrl_soqs = [initial_soqs[creg_name] for creg_name in self.ctrl_reg_names]

        soq_map: List[Tuple[SoquetT, SoquetT]] = []
        for binst, in_soqs, old_out_soqs in cbloq.iter_bloqsoqs():
            in_soqs = bb.map_soqs(in_soqs, soq_map)
            new_bloq, adder = binst.bloq.get_ctrl_system(self.ctrl_spec)
            ctrl_soqs, new_out_soqs = adder(bb, ctrl_soqs=ctrl_soqs, in_soqs=in_soqs)
            soq_map.extend(zip(old_out_soqs, new_out_soqs))

        fsoqs = bb.map_soqs(cbloq.final_soqs(), soq_map)
        fsoqs |= dict(zip(self.ctrl_reg_names, ctrl_soqs))
        return bb.finalize(**fsoqs)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {
            (bloq.controlled(self.ctrl_spec), n)
            for bloq, n in self.subbloq.build_call_graph(ssa=ssa)
        }

    def on_classical_vals(self, **vals: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        ctrl_vals = [vals[reg_name] for reg_name in self.ctrl_reg_names]
        other_vals = {reg.name: vals[reg.name] for reg in self.subbloq.signature}
        if self.ctrl_spec.is_active(*ctrl_vals):
            rets = self.subbloq.on_classical_vals(**other_vals)
            rets |= {
                reg_name: ctrl_val for reg_name, ctrl_val in zip(self.ctrl_reg_names, ctrl_vals)
            }
            return rets

        return vals

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        if soq.reg.name not in self.ctrl_reg_names:
            # Delegate to subbloq
            return self.subbloq.wire_symbol(soq)

        # Otherwise, it's part of the control register.
        i = self.ctrl_reg_names.index(soq.reg.name)
        return self.ctrl_spec.wire_symbol(i, soq)

    def pretty_name(self) -> str:
        return f'C[{self.subbloq.pretty_name()}]'

    def short_name(self) -> str:
        return f'C[{self.subbloq.short_name()}]'

    def __str__(self) -> str:
        return f'C[{self.subbloq}]'
