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

import numpy as np

from qualtran._infra.gate_with_registers import split_qubits, total_bits
from qualtran.bloqs.basic_gates import CNOT, TGate
from qualtran.bloqs.data_loading.qroam import _qroam_small, QROAM
from qualtran.cirq_interop.bit_tools import iter_bits
from qualtran.cirq_interop.t_complexity_protocol import t_complexity
from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim, GateHelper
from qualtran.resource_counting.generalizers import cirq_to_bloqs
from qualtran.testing import (
    assert_valid_bloq_decomposition,
    assert_wire_symbols_match_expected,
    execute_notebook,
)


def test_qroam_small(bloq_autotester):
    bloq_autotester(_qroam_small)


# def test_qroam_classical():
#     rs = np.random.RandomState()
#     data = rs.randint(0, 2**3, size=10)
#     sel_size = int(np.ceil(np.log2(10)))
#     qrom = QROM([data], (sel_size,), target_bitsizes=(3,))
#     cbloq = qrom.decompose_bloq()
#     for i in range(len(data)):
#         i_out, data_out = qrom.call_classically(selection=i, target0_=0)
#         assert i_out == i
#         assert data_out == data[i]

#         decomp_ret = cbloq.call_classically(selection=i, target0_=0)
#         assert decomp_ret == (i_out, data_out)


# def test_qroam_classical_nonzero_target():
#     rs = np.random.RandomState()
#     data = rs.randint(0, 2**3, size=10)
#     sel_size = int(np.ceil(np.log2(10)))
#     qrom = QROM([data], (sel_size,), target_bitsizes=(3,))
#     cbloq = qrom.decompose_bloq()
#     for i in range(len(data)):
#         target_in = int('111', 2)
#         i_out, data_out = qrom.call_classically(selection=i, target0_=target_in)
#         assert i_out == i
#         assert data_out == data[i] ^ target_in

#         decomp_ret = cbloq.call_classically(selection=i, target0_=target_in)
#         assert decomp_ret == (i_out, data_out)


# def test_qroam_1d_multitarget_classical():
#     rs = np.random.RandomState()
#     n = 10
#     data_sets = [rs.randint(0, 2**3, size=n) for _ in range(3)]
#     sel_size = int(np.ceil(np.log2(10)))
#     qrom = QROM(data_sets, (sel_size,), target_bitsizes=(3, 3, 3))
#     cbloq = qrom.decompose_bloq()
#     for i in range(n):
#         init = {f'target{i}_': 0 for i in range(3)}
#         i_out, *data_out = qrom.call_classically(selection=i, **init)
#         assert i_out == i
#         assert data_out == [data[i] for data in data_sets]

#         decomp_i_out, *decomp_data_out = cbloq.call_classically(selection=i, **init)
#         assert decomp_i_out == i_out
#         assert len(data_out) == len(decomp_data_out)
#         for do, decomp_do in zip(data_out, decomp_data_out):
#             np.testing.assert_array_equal(do, decomp_do)


# def test_qroam_wire_symbols():
#     qrom = QROM.build([3, 3, 3, 3])
#     assert_wire_symbols_match_expected(qrom, ['In', 'data_a'])

#     qrom = QROM.build([3, 3, 3, 3], [2, 2, 2, 2])
#     assert_wire_symbols_match_expected(qrom, ['In', 'data_a', 'data_b'])

#     qrom = QROM.build([[3, 3], [3, 3]], [[2, 2], [2, 2]], [[1, 1], [2, 2]])
#     assert_wire_symbols_match_expected(qrom, ['In_i', 'In_j', 'data_a', 'data_b', 'data_c'])

#     qrom = QROM.build(np.arange(27).reshape(3, 3, 3))
#     assert_wire_symbols_match_expected(qrom, ['In_i', 'In_j', 'In_k', 'data_a'])
