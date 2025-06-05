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

import numpy as np
import pytest

from qualtran import QAny, Register, Signature
from qualtran.bloqs.basic_gates import Hadamard
from qualtran.bloqs.block_encoding.phase import _phase_block_encoding, Phase
from qualtran.bloqs.block_encoding.product_test import TestBlockEncoding, TestPrepareOracle
from qualtran.bloqs.block_encoding.unitary import Unitary
from qualtran.testing import execute_notebook


def test_phase(bloq_autotester):
    bloq_autotester(_phase_block_encoding)


def test_phase_signature():
    assert _phase_block_encoding().signature == Signature([Register("system", QAny(1))])


def test_phase_params():
    bloq = _phase_block_encoding()
    assert bloq.alpha == 1
    assert bloq.epsilon == 0
    assert bloq.ancilla_bitsize == 0
    assert bloq.resource_bitsize == 0


def test_phase_tensors():
    bloq = _phase_block_encoding()
    from_gate = np.exp(0.25j * np.pi) * Hadamard().tensor_contract()
    from_tensors = bloq.tensor_contract()
    np.testing.assert_allclose(from_gate, from_tensors)


def test_negate():
    bloq = Phase(Unitary(Hadamard()), phi=1, eps=0)
    from_gate = -Hadamard().tensor_contract()
    from_tensors = bloq.tensor_contract()
    np.testing.assert_allclose(from_gate, from_tensors)


def test_phase_signal_state():
    bloq = Phase(TestBlockEncoding(), phi=1, eps=0)
    assert isinstance(bloq.signal_state.prepare, TestPrepareOracle)


@pytest.mark.notebook
def test_notebook():
    execute_notebook('phase')
