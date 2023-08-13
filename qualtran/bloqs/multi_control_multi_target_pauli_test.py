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


import cirq
from cirq_ft import MultiControlPauli as CirqMultiControlPauli
from cirq_ft.infra import t_complexity

import qualtran.testing as qlt_testing
from qualtran.bloqs.basic_gates import XGate, ZGate
from qualtran.bloqs.multi_control_multi_target_pauli import MultiControlPauli


def test_multi_control_pauli_decomp():
    mcp = MultiControlPauli([0, 1, 1, 1, 0], XGate())
    qlt_testing.assert_valid_bloq_decomposition(mcp)


def test_tcomplexity():
    mcp = MultiControlPauli([0, 1, 1, 0], ZGate())
    cbloq = mcp.decompose_bloq()
    cirq_mcp = CirqMultiControlPauli([0, 1, 1, 0], cirq.Z)
    assert cbloq.t_complexity() == t_complexity(cirq_mcp)
