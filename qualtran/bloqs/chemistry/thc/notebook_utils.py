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

from typing import Optional

import cirq

from qualtran import Bloq
from qualtran.bloqs.bookkeeping import ArbitraryClifford
from qualtran.bloqs.data_loading.qrom import QROM
from qualtran.bloqs.data_loading.select_swap_qrom import SelectSwapQROM
from qualtran.cirq_interop import CirqGateAsBloq
from qualtran.resource_counting import SympySymbolAllocator
from qualtran.resource_counting.generalizers import (
    generalize_cvs,
    generalize_rotation_angle,
    ignore_alloc_free,
    ignore_split_join,
)

# this code will be cleaned up post cirq_ft alignment:
# https://github.com/quantumlib/Qualtran/issues/425
single_qubit_clifford = (
    cirq.ops.common_gates.HPowGate,
    cirq.ops.common_gates.XPowGate,
    cirq.ops.common_gates.YPowGate,
    cirq.ops.common_gates.ZPowGate,
    type(cirq.ops.common_gates.S),
)
two_qubit_clifford = (cirq.ops.common_gates.CZPowGate, cirq.ops.common_gates.CXPowGate)

ssa = SympySymbolAllocator()
phi_sym = ssa.new_symbol('phi')
and_cv0 = ssa.new_symbol('cv0')
and_cv1 = ssa.new_symbol('cv1')
mcp_cv0 = ssa.new_symbol('cv3')


def custom_qroam_repr(self) -> str:
    target_repr = repr(self.target_bitsizes)
    return f"SelectSwapQROM(target_bitsizes={target_repr}, block_sizes={self.block_sizes})"


# TODO: better way of customizing label
SelectSwapQROM.__repr__ = custom_qroam_repr  # type: ignore[assignment]


def custom_qrom_repr(self) -> str:
    target_repr = repr(self.target_bitsizes)
    selection_repr = repr(self.selection_bitsizes)
    return f"QROM(selection_bitsizes={selection_repr}, target_bitsizes={target_repr})"


# TODO: better way of customizing label
QROM.__repr__ = custom_qrom_repr  # type: ignore[assignment]


def custom_generalizations(bloq: Bloq) -> Optional[Bloq]:
    if isinstance(bloq, CirqGateAsBloq):
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


GENERALIZERS = [
    ignore_split_join,
    ignore_alloc_free,
    generalize_rotation_angle,
    generalize_cvs,
    custom_generalizations,
]
