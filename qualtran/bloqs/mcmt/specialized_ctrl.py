#  Copyright 2024 Google LLC
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
import enum
from functools import cached_property
from typing import Callable, cast, Iterable, Optional, Sequence, TYPE_CHECKING

import attrs
import numpy as np

from qualtran import Adjoint, Bloq, BloqBuilder, CompositeBloq, QBit, Register, Signature
from qualtran.bloqs.bookkeeping import AutoPartition

if TYPE_CHECKING:
    from qualtran import AddControlledT, CtrlSpec, SoquetT
    from qualtran._infra.controlled import ControlBit
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@attrs.frozen
class _MultiControlledFromSinglyControlled(Bloq):
    """Helper bloq implementing a multi-controlled-U given access to controlled-U.

    This is for internal use only. For reducing multiple controls to a single control,
    see :class:`qualtran.bloqs.mcmt.ControlledViaAnd` and
    :meth:`qualtran.bloqs.mcmt.specialized_ctrl.get_ctrl_system_1bit_cv`.

    This bloq is used as an intermediate bloq by `get_ctrl_system_1bit_cv` in the
    controlled-controlled-bloq case. To cleanly support further controlling this bloq,
    the `cvs` attribute accepts a tuple (of at least two controls), and defers to
    `ControlledViaAnd` whenever possible, and only extends the `cvs` in the edge cases.
    """

    cvs: tuple[int, ...]
    ctrl_bloq: Bloq
    ctrl_reg_name: str

    def __attrs_post_init__(self):
        assert len(self.cvs) >= 2, f"{self} must have at least 2 controls, got {self.cvs=}"

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [Register(self.ctrl_reg_name, dtype=QBit(), shape=(len(self.cvs),))]
            + [reg for reg in self.ctrl_bloq.signature if reg.name != self.ctrl_reg_name]
        )

    @cached_property
    def _and_bloq(self) -> Bloq:
        from qualtran.bloqs.mcmt import And, MultiAnd

        if len(self.cvs) == 2:
            return And(*self.cvs)
        else:
            return MultiAnd(self.cvs)

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> dict[str, 'SoquetT']:
        and_soqs = bb.add_d(self._and_bloq, ctrl=soqs.pop(self.ctrl_reg_name))

        soqs |= {self.ctrl_reg_name: and_soqs.pop('target')}
        soqs = bb.add_d(self.ctrl_bloq, **soqs)
        and_soqs |= {'target': soqs.pop(self.ctrl_reg_name)}

        soqs |= {self.ctrl_reg_name: bb.add(self._and_bloq.adjoint(), **and_soqs)}

        return soqs

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {self._and_bloq: 1, self.ctrl_bloq: 1, self._and_bloq.adjoint(): 1}

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> tuple['Bloq', 'AddControlledT']:
        if ctrl_spec.num_qubits != 1:
            return super().get_ctrl_system(ctrl_spec=ctrl_spec)

        ctrl_bloq = attrs.evolve(self, cvs=(ctrl_spec.get_single_ctrl_val(),) + self.cvs)

        def _adder(bb, ctrl_soqs, in_soqs):
            in_soqs[self.ctrl_reg_name] = np.concatenate(ctrl_soqs, in_soqs[self.ctrl_reg_name])
            ctrls, *out_soqs = bb.add_t(ctrl_bloq, **in_soqs)
            return ctrls[:1], [*ctrls[1:], *out_soqs]

        return ctrl_bloq, _adder

    def __str__(self):
        return f'C[{len(self.cvs)-1}][{self.ctrl_bloq}]'


def _get_ctrl_system_1bit_cv(
    bloq: 'Bloq',
    ctrl_spec: 'CtrlSpec',
    *,
    current_ctrl_bit: Optional['ControlBit'],
    get_ctrl_bloq_and_ctrl_reg_name: Callable[['ControlBit'], Optional[tuple['Bloq', str]]],
) -> tuple['Bloq', 'AddControlledT']:
    """Internal method to build the control system for a bloq using single-qubit controlled variants.

    Uses the provided specialized implementation when a singly-controlled variant of the bloq is
    requested. When controlled by multiple qubits, the controls are reduced to a single qubit
    and the singly-controlled bloq is used.

    The user can provide specializations for the bloq controlled by `1` and (optionally) by `0`.
    The specialization for control bit `1` must be provided.
    In case a specialization for a control bit `0` is not provided, the default fallback is used
    instead, which wraps the bloq using the `Controlled` metabloq.

    Args:
        bloq: The current bloq.
        ctrl_spec: The control specification
        current_ctrl_bit: The control bit of the current bloq, one of `0, 1, None`.
        get_ctrl_bloq_and_ctrl_reg_name: A callable that accepts a control bit (`0` or `1`),
            and returns the controlled variant of this bloq and the name of the control register.
            If the callable returns `None`, then the default fallback is used.
    """
    from qualtran import make_ctrl_system_with_correct_metabloq, Soquet

    def _get_default_fallback():
        return make_ctrl_system_with_correct_metabloq(bloq=bloq, ctrl_spec=ctrl_spec)

    if ctrl_spec.num_qubits != 1:
        return _get_default_fallback()

    ctrl_bit = ctrl_spec.get_single_ctrl_val()

    if current_ctrl_bit is None:
        # the easy case: use the controlled bloq
        ctrl_bloq_and_ctrl_reg_name = get_ctrl_bloq_and_ctrl_reg_name(ctrl_bit)
        if ctrl_bloq_and_ctrl_reg_name is None:
            assert ctrl_bit != 1, "invalid usage: controlled-by-1 variant must be provided"
            return _get_default_fallback()

        ctrl_bloq, ctrl_reg_name = ctrl_bloq_and_ctrl_reg_name

        def _adder(
            bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: dict[str, 'SoquetT']
        ) -> tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
            (ctrl,) = ctrl_soqs
            in_soqs |= {ctrl_reg_name: ctrl}

            out_soqs = bb.add_d(ctrl_bloq, **in_soqs)

            ctrl = out_soqs.pop(ctrl_reg_name)
            return [ctrl], out_soqs.values()

    else:
        # the difficult case: must combine the two controls into one
        ctrl_1_bloq_and_reg_name = get_ctrl_bloq_and_ctrl_reg_name(1)
        assert (
            ctrl_1_bloq_and_reg_name is not None
        ), "invalid usage: controlled-by-1 variant must be provided"
        ctrl_1_bloq, ctrl_reg_name = ctrl_1_bloq_and_reg_name

        ctrl_bloq = _MultiControlledFromSinglyControlled(
            cvs=(ctrl_bit, current_ctrl_bit), ctrl_bloq=ctrl_1_bloq, ctrl_reg_name=ctrl_reg_name
        )

        def _adder(
            bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: dict[str, 'SoquetT']
        ) -> tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
            # extract the two control bits
            (ctrl0,) = ctrl_soqs
            ctrl1 = in_soqs.pop(ctrl_reg_name)

            ctrl0 = cast(Soquet, ctrl0)
            ctrl1 = cast(Soquet, ctrl1)

            # add the singly controlled bloq
            in_soqs |= {ctrl_reg_name: np.array([ctrl0, ctrl1])}
            ctrls, *out_soqs = bb.add_t(ctrl_bloq, **in_soqs)
            assert isinstance(ctrls, np.ndarray)
            ctrl0, ctrl1 = ctrls

            return [ctrl0], [ctrl1, *out_soqs]

    def _unwrap(b):
        if isinstance(b, AutoPartition):
            return _unwrap(b.bloq)
        return b

    return _unwrap(ctrl_bloq), _adder


def get_ctrl_system_1bit_cv(
    bloq: 'Bloq',
    ctrl_spec: 'CtrlSpec',
    *,
    current_ctrl_bit: Optional['ControlBit'],
    get_ctrl_bloq_and_ctrl_reg_name: Callable[['ControlBit'], tuple['Bloq', str]],
) -> tuple['Bloq', 'AddControlledT']:
    """Build the control system for a bloq with specialized single-qubit controlled variants.

    Uses the provided specialized implementation when a singly-controlled variant of the bloq is
    requested. When controlled by multiple qubits, the controls are reduced to a single qubit
    and the singly-controlled bloq is used.

    The user must provide two specializations for the bloq: controlled by `1` and by `0`.

    When only one specialization (controlled by `1`) is known, use
    :meth:`get_ctrl_system_1bit_cv_from_bloqs` instead.

    Args:
        bloq: The current bloq.
        ctrl_spec: The control specification
        current_ctrl_bit: The control bit of the current bloq, one of `0, 1, None`.
        get_ctrl_bloq_and_ctrl_reg_name: A callable that accepts a control bit (`0` or `1`),
            and returns the controlled variant of this bloq and the name of the control register.
    """
    return _get_ctrl_system_1bit_cv(
        bloq,
        ctrl_spec,
        current_ctrl_bit=current_ctrl_bit,
        get_ctrl_bloq_and_ctrl_reg_name=get_ctrl_bloq_and_ctrl_reg_name,
    )


def get_ctrl_system_1bit_cv_from_bloqs(
    bloq: 'Bloq',
    ctrl_spec: 'CtrlSpec',
    *,
    current_ctrl_bit: Optional['ControlBit'],
    bloq_with_ctrl: 'Bloq',
    ctrl_reg_name: 'str',
) -> tuple['Bloq', 'AddControlledT']:
    """Helper to construct the control system given a singly-controlled variant of a bloq.

    Uses the provided specialized implementation when a singly-controlled (by `1`) variant of
    the bloq is requested. When controlled by multiple qubits, the controls are reduced to a
    single qubit and the singly-controlled bloq is used.

    When specializations for both cases - controlled by `1` and by `0` - are known, use
    :meth:`get_ctrl_system_1bit_cv` instead.

    Args:
        bloq: The current bloq.
        ctrl_spec: The control specification
        current_ctrl_bit: The control bit of the current bloq, one of `0, 1, None`.
        bloq_with_ctrl: The variant of this bloq controlled by a single qubit in the `1` basis state.
        ctrl_reg_name: The name of the control register for the controlled bloq variant(s).
    """

    def get_ctrl_bloq_and_ctrl_reg_name(cv: 'ControlBit') -> Optional[tuple['Bloq', str]]:
        if cv == 1:
            return bloq_with_ctrl, ctrl_reg_name
        else:
            return None

    return _get_ctrl_system_1bit_cv(
        bloq,
        ctrl_spec,
        current_ctrl_bit=current_ctrl_bit,
        get_ctrl_bloq_and_ctrl_reg_name=get_ctrl_bloq_and_ctrl_reg_name,
    )


class SpecializeOnCtrlBit(enum.Flag):
    """Control-specs to propagate to the subbloq.

    See `AdjointWithSpecializedCtrl` for usage.

    Currently only allows pushing a single-qubit-control.
    """

    NONE = enum.auto()
    ZERO = enum.auto()
    ONE = enum.auto()
    BOTH = ZERO | ONE


@attrs.frozen()
class AdjointWithSpecializedCtrl(Adjoint):
    """Adjoint of a bloq with a specialized control implementation.

    If the subbloq has a specialized control implementation, then calling
    `Adjoint(subbloq).controlled()` propagates the controls to the subbloq.
    This only propagates single-qubit `CtrlSpec`s, all others use the default:
    reduced to single-qubit control using the `ControlledViaAnd` bloq.

    By default in Qualtran, `Controlled(bloq).adjoint()` returns `Controlled(bloq.adjoint())`.
    But `Adjoint(bloq).controlled()` does not propagate the controls, therefore returns
    `Controlled(Adjoint(bloq))`.
    This bloq helps override that behaviour for single-qubit controlled versions.

    For example, if a bloq has a specialized implementation for the controlled-by-1 case:

    ```py
    class BloqWithSpecializedCtrl(Bloq):
        ...

        def adjoint(self):
            return AdjointWithSpecializedCtrl(self, SpecializeOnCtrlBit.ONE)
    ```

    See `get_ctrl_system_1bit_cv` on one way to provide specialized controlled implementations
    for bloqs. If a bloq uses the above and does not have a trivial `adjoint` implementation,
    it is recommended to override the `adjoint` method as show above.

    Caution:
        Use this bloq _only_ when a specialized control implementation is guaranteed,
        i.e. `subbloq.controlled()` should not return `Controlled(...)`.
        Otherwise, it could lead to an infinite recursion.

    Args:
        subbloq: The bloq to wrap.
        specialize_on_ctrl: Values of the control bit to propagate the control into the subbloq.
            Can be `SpecializeOnCtrlBit.ONE` for `1` only, `SpecializeOnCtrlBit.ZERO` for `0` only,
            or `SpecializeOnCtrlBit.BOTH` for both `0` and `1`.
    """

    specialize_on_ctrl: SpecializeOnCtrlBit = SpecializeOnCtrlBit.NONE

    def _specialize_control(self, ctrl_spec: 'CtrlSpec') -> bool:
        """if True, push the control to the subbloq"""
        if ctrl_spec.num_qubits != 1:
            return False

        cv = ctrl_spec.get_single_ctrl_val()
        cv_flag = SpecializeOnCtrlBit.ONE if cv == 1 else SpecializeOnCtrlBit.ZERO
        return cv_flag in self.specialize_on_ctrl

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> tuple['Bloq', 'AddControlledT']:
        from qualtran._infra.controlled import _get_nice_ctrl_reg_names

        if not self._specialize_control(ctrl_spec):
            # no specialized controlled version available, fallback to default
            return super().get_ctrl_system(ctrl_spec)

        # get the builder for the controlled version of subbloq
        ctrl_subbloq, ctrl_subbloq_adder = self.subbloq.get_ctrl_system(ctrl_spec)
        ctrl_bloq = attrs.evolve(self, subbloq=ctrl_subbloq)
        (ctrl_reg_name,) = _get_nice_ctrl_reg_names([reg.name for reg in self.subbloq.signature], 1)

        # build a composite bloq using the control-adder
        def _get_adj_cbloq() -> 'CompositeBloq':
            bb, initial_soqs = BloqBuilder.from_signature(
                self.subbloq.signature, add_registers_allowed=True
            )
            ctrl = bb.add_register(ctrl_reg_name, 1)
            bb.add_register_allowed = False

            (ctrl,), out_soqs_t = ctrl_subbloq_adder(bb, [ctrl], initial_soqs)

            out_soqs = dict(zip([reg.name for reg in self.subbloq.signature.rights()], out_soqs_t))
            out_soqs |= {ctrl_reg_name: ctrl}

            cbloq = bb.finalize(**out_soqs)
            return cbloq.adjoint()

        adj_cbloq = _get_adj_cbloq()

        def _adder(
            bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: dict[str, 'SoquetT']
        ) -> tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
            (ctrl,) = ctrl_soqs
            in_soqs |= {ctrl_reg_name: ctrl}
            soqs = bb.add_from(adj_cbloq, **in_soqs)

            # locate the correct control soquet
            soqs = list(soqs)
            ctrl_soq = None
            for soq, reg in zip(soqs, adj_cbloq.signature.rights()):
                if reg.name == ctrl_reg_name:
                    ctrl_soq = soq
                    soqs.remove(soq)
                    break
            assert ctrl_soq is not None, "ctrl_soq must be present in output soqs"

            return [ctrl_soq], soqs

        return ctrl_bloq, _adder
