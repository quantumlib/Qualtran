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
from collections import defaultdict
from typing import Dict

import attrs
import cirq
import cirq_ft

from qualtran.bloqs.and_bloq import And
from qualtran.bloqs.arithmetic import (
    EqualsAConstant,
    GreaterThan,
    GreaterThanConstant,
    LessThanConstant,
    LessThanEqual,
    ToContiguousIndex,
)
from qualtran.bloqs.basic_gates import Rx, Ry, Rz, TGate
from qualtran.bloqs.multi_control_multi_target_pauli import MultiControlPauli
from qualtran.bloqs.swap_network import CSwapApprox, SwapWithZero
from qualtran.bloqs.util_bloqs import Allocate, ArbitraryClifford, Free, Join, Split
from qualtran.cirq_interop import CirqGateAsBloq
from qualtran.resource_counting import get_bloq_counts_graph, SympySymbolAllocator

single_qubit_clifford = (
    cirq.ops.common_gates.HPowGate,
    cirq.ops.common_gates.XPowGate,
    cirq.ops.common_gates.YPowGate,
    cirq.ops.common_gates.ZPowGate,
    type(cirq.ops.common_gates.S),
)
two_qubit_clifford = (cirq.ops.common_gates.CZPowGate, cirq.ops.common_gates.CXPowGate)
rotation_bloqs = (Rx, Ry, Rz)
bloq_comparators = (
    EqualsAConstant,
    GreaterThanConstant,
    GreaterThan,
    LessThanConstant,
    LessThanEqual,
)

ssa = SympySymbolAllocator()
phi_sym = ssa.new_symbol('phi')
and_cv0 = ssa.new_symbol('cv0')
and_cv1 = ssa.new_symbol('cv1')
mcp_cv0 = ssa.new_symbol('cv3')


def custom_qroam_repr(self) -> str:
    target_repr = repr(self._target_bitsizes)
    return f"cirq_ft.SelectSwapQROM(target_bitsizes={target_repr}, block_size={self.block_size})"


def custom_qrom_repr(self) -> str:
    target_repr = repr(self.target_bitsizes)
    selection_repr = repr(self.selection_bitsizes)
    return f"cirq_ft.QROM(selection_bitsizes={selection_repr}, target_bitsizes={target_repr})"


def generalize(bloq):
    """Genereralizer for THC prepare bloqs."""
    if isinstance(bloq, (Allocate, Free, Split, Join)):
        return None
    if isinstance(bloq, (Rx, Ry, Rz)):
        return attrs.evolve(bloq, angle=phi_sym)
    if isinstance(bloq, And):
        return attrs.evolve(bloq, cv1=and_cv0, cv2=and_cv1)
    if isinstance(bloq, CirqGateAsBloq):
        if isinstance(bloq.gate, cirq_ft.algos.And) and (len(bloq.gate.cv) == 2):
            return And(cv1=and_cv0, cv2=and_cv1, adjoint=bloq.gate.adjoint)
        if isinstance(bloq.gate, cirq_ft.algos.SwapWithZeroGate):
            return SwapWithZero(
                bloq.gate.selection_bitsize, bloq.gate.target_bitsize, bloq.gate.n_target_registers
            )
        # How to check type of cirg_gate**-1
        if (
            isinstance(bloq.gate, cirq.ops.raw_types._InverseCompositeGate)
            and 'SwapWithZero' in bloq.pretty_name()
        ):
            return SwapWithZero(
                (bloq.gate**-1).selection_bitsize,
                (bloq.gate**-1).target_bitsize,
                (bloq.gate**-1).n_target_registers,
            )
        if isinstance(bloq.gate, cirq_ft.algos.SelectSwapQROM):
            bloq.gate.__class__.__repr__ = custom_qroam_repr
        if isinstance(bloq.gate, cirq_ft.algos.QROM):
            bloq.gate.__class__.__repr__ = custom_qrom_repr
        if isinstance(bloq.gate, single_qubit_clifford):
            return ArbitraryClifford(n=1)
        if isinstance(bloq.gate, two_qubit_clifford):
            return ArbitraryClifford(n=2)
        if (
            isinstance(bloq.gate, cirq.ControlledGate)
            and isinstance(bloq.gate._sub_gate, single_qubit_clifford)
            and bloq.gate.num_controls() == 1
        ):
            return ArbitraryClifford(n=2)
    return bloq


def bin_bloq_counts(bloq) -> Dict[str, int]:
    """Classify bloq counts.

    It's helpful to classify bloqs by their type (comparators, reflections, swaps, ...)

    Args:
        bloq: the bloq to classify.

    Returns
    -------
        classified_bloqs : Dataclass containing bloq counts for different types of bloqs.
    """
    classified_bloqs = defaultdict(int)
    for num_calls, bloq in bloq.bloq_counts():
        if isinstance(bloq, (Split, Join, Allocate, Free)):
            continue
        num_t = get_bloq_counts_graph(bloq, generalizer=generalize)[1].get(TGate())
        if num_t is not None:
            if isinstance(bloq, bloq_comparators):
                classified_bloqs['comparator'] += num_calls * num_t
            elif isinstance(bloq, MultiControlPauli):
                if isinstance(bloq.target_gate, cirq.ops.common_gates.ZPowGate):
                    classified_bloqs['reflections'] += num_calls * num_t
            elif isinstance(bloq, CirqGateAsBloq):
                if isinstance(bloq.gate, (cirq_ft.SelectSwapQROM, cirq_ft.QROM)):
                    classified_bloqs['qrom'] += num_calls * num_t
            elif isinstance(bloq, CSwapApprox):
                classified_bloqs['controlled_swaps'] += num_calls * num_t
            elif isinstance(bloq, rotation_bloqs):
                classified_bloqs['rotation'] += num_calls * num_t
            elif isinstance(bloq, ToContiguousIndex):
                classified_bloqs['contiguous_register'] += num_calls * num_t
            else:
                classified_bloqs['other'] += num_calls * num_t
    return classified_bloqs
