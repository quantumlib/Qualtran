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
from functools import cached_property
from typing import Optional, TYPE_CHECKING, Union

from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    BQUInt,
    CtrlSpec,
    DecomposeTypeError,
    QAny,
    QBit,
    QInt,
    QIntOnesComp,
    QMontgomeryUInt,
    QUInt,
    Register,
    Side,
    Signature,
)
from qualtran.bloqs.bookkeeping.partition import Partition
from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd
from qualtran.drawing import directional_text_box, Text, WireSymbol
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.symbolics import HasLength, is_symbolic, SymbolicInt

if TYPE_CHECKING:
    from qualtran import BloqBuilder, SoquetT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@frozen
class CtrlSpecAnd(Bloq):
    """Computes a single qubit which is `1` iff the CtrlSpec of And clauses is satisfied.

    This reduces an arbitrary 'And' clause control spec to a single qubit, which can be used
    to then control a bloq. Therefore, a bloq author is only required to implement a
    single-controlled version of their bloq, and can be generalized to arbitrary controls.

    The control registers are passed through as-is. If the same control bit is required for
    multiple bloqs, the user can use the `target` qubit of this bloq multiple times, and only
    uncompute at the very end. For more custom strategies and trade-offs, see Ref. [1].

    Note:
        This only applies to CtrlSpec being a logical AND of all registers, and each register
        being equal to a constant. See documentation for :class:`CtrlSpec` for more details.

    Args:
        ctrl_spec: The control specification.

    Registers:
        ctrl_i: The control register for the i-th ctrl dtype in the `ctrl_spec`.
        junk [right]: `ctrl_spec.num_qubits - 2` qubits that can be cleaned up by the inverse.
                      Only present if the above size is non-zero.
        target [right]: The output bit storing the result of the `ctrl_spec`.

    References:
        [Unqomp: synthesizing uncomputation in Quantum circuits](https://dl.acm.org/doi/10.1145/3453483.3454040)
        Paradis et. al. 2021.
    """

    ctrl_spec: CtrlSpec

    def __attrs_post_init__(self):
        if not is_symbolic(self.n_ctrl_qubits) and self.n_ctrl_qubits <= 1:
            raise ValueError(f"Expected at least 2 controls, got {self.n_ctrl_qubits}")
        for qdtype in self.ctrl_spec.qdtypes:
            if not isinstance(qdtype, (QBit, QInt, QUInt, BQUInt, QIntOnesComp, QMontgomeryUInt)):
                raise NotImplementedError("CtrlSpecAnd currently only supports integer types")

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                *self.control_registers,
                *self.junk_registers(),
                Register('target', QBit(), side=Side.RIGHT),
            ]
        )

    @cached_property
    def control_registers(self) -> tuple[Register, ...]:
        return tuple(
            Register(f'ctrl_{i}', dtype=dtype, shape=shape)
            for i, (dtype, shape) in enumerate(self.ctrl_spec.activation_function_dtypes())
        )

    def junk_registers(self) -> tuple[Register, ...]:
        if not is_symbolic(self.n_ctrl_qubits) and self.n_ctrl_qubits == 2:
            return ()

        return (Register('junk', QAny(self.n_ctrl_qubits - 2), side=Side.RIGHT),)

    @cached_property
    def n_ctrl_qubits(self) -> SymbolicInt:
        return self.ctrl_spec.num_qubits

    @cached_property
    def _ctrl_partition_bloq(self) -> Partition:
        return Partition(self.ctrl_spec.num_qubits, self.control_registers)

    @property
    def _flat_cvs(self) -> Union[tuple[int, ...], HasLength]:
        if is_symbolic(self.ctrl_spec):
            return HasLength(self.n_ctrl_qubits)

        flat_cvs: list[int] = []
        for reg, cv in zip(self.control_registers, self.ctrl_spec.cvs):
            flat_cvs.extend(reg.dtype.to_bits_array(cv.ravel()).ravel())
        return tuple(flat_cvs)

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> dict[str, 'SoquetT']:
        if is_symbolic(self.n_ctrl_qubits):
            raise DecomposeTypeError(
                f"Cannot decompose {self} with symbolic number of controls {self.n_ctrl_qubits}"
            )

        # unpartition (merge) all the control registers
        ctrl_qubits = bb.add(self._ctrl_partition_bloq.adjoint(), **soqs)
        ctrl_qubits = bb.split(ctrl_qubits)

        # Compute the single control qubit `target`
        if self.n_ctrl_qubits == 2:
            assert isinstance(self._flat_cvs, tuple)
            cv1, cv2 = self._flat_cvs
            ctrl_qubits, target = bb.add(And(cv1, cv2), ctrl=ctrl_qubits)
            junk = None
        else:
            ctrl_qubits, junk, target = bb.add(MultiAnd(self._flat_cvs), ctrl=ctrl_qubits)

        # partition the control qubits back into the control registers
        ctrl_qubits = bb.join(ctrl_qubits)
        soqs = bb.add_d(self._ctrl_partition_bloq, x=ctrl_qubits)

        if junk is not None:
            soqs['junk'] = bb.join(junk)
        soqs['target'] = target

        return soqs

    def wire_symbol(self, reg: Optional[Register], idx: tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('Ctrl')

        if reg.name[:5] == 'ctrl_':
            i = int(reg.name[5:])
            return self.ctrl_spec.wire_symbol(i, reg, idx)

        if reg.name == 'target':
            return directional_text_box('C', side=reg.side)

        # junk
        if len(idx) > 0:
            pretty_text = f'{reg.name}[{", ".join(str(i) for i in idx)}]'
        else:
            pretty_text = reg.name
        return directional_text_box(text=pretty_text, side=reg.side)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        if not is_symbolic(self.n_ctrl_qubits) and self.n_ctrl_qubits == 2:
            assert isinstance(self._flat_cvs, tuple)
            cv1, cv2 = self._flat_cvs
            return {And(cv1, cv2): 1}

        return {MultiAnd(self._flat_cvs): 1}


@bloq_example(generalizer=ignore_split_join)
def _ctrl_on_int() -> CtrlSpecAnd:
    from qualtran import CtrlSpec, QUInt

    ctrl_on_int = CtrlSpecAnd(CtrlSpec(qdtypes=QUInt(4), cvs=[0b0101]))
    return ctrl_on_int


@bloq_example(generalizer=ignore_split_join)
def _ctrl_on_bits() -> CtrlSpecAnd:
    from qualtran import CtrlSpec, QBit

    ctrl_on_bits = CtrlSpecAnd(CtrlSpec(qdtypes=QBit(), cvs=[0, 1, 0, 1]))
    return ctrl_on_bits


@bloq_example(generalizer=ignore_split_join)
def _ctrl_on_nd_bits() -> CtrlSpecAnd:
    import numpy as np

    from qualtran import CtrlSpec, QBit

    ctrl_on_nd_bits = CtrlSpecAnd(CtrlSpec(qdtypes=QBit(), cvs=np.array([[0, 1], [1, 0]])))
    return ctrl_on_nd_bits


@bloq_example(generalizer=ignore_split_join)
def _ctrl_on_multiple_values() -> CtrlSpecAnd:
    from qualtran import CtrlSpec, QInt, QUInt

    ctrl_on_multiple_values = CtrlSpecAnd(
        CtrlSpec(qdtypes=(QUInt(4), QInt(4)), cvs=([0b0101, 0b1100], [2, -2]))
    )
    return ctrl_on_multiple_values


_CTRLSPEC_AND_DOC = BloqDocSpec(
    bloq_cls=CtrlSpecAnd,
    examples=(_ctrl_on_int, _ctrl_on_bits, _ctrl_on_nd_bits, _ctrl_on_multiple_values),
)
