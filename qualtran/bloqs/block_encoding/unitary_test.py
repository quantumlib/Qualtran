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

from typing import cast

import numpy as np
import pytest

from qualtran import BloqBuilder, QAny, Register, Signature, Soquet
from qualtran.bloqs.basic_gates import IntState, TGate, ZeroEffect, ZeroState
from qualtran.bloqs.block_encoding.unitary import (
    _unitary_block_encoding,
    _unitary_block_encoding_properties,
    Unitary,
)
from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
from qualtran.testing import execute_notebook


def test_unitary(bloq_autotester):
    bloq_autotester(_unitary_block_encoding)
    bloq_autotester(_unitary_block_encoding_properties)


def test_unitary_signature():
    assert _unitary_block_encoding().signature == Signature([Register("system", QAny(1))])

    assert _unitary_block_encoding_properties().signature == Signature(
        [Register("system", QAny(1)), Register("ancilla", QAny(2)), Register("resource", QAny(1))]
    )

    with pytest.raises(ValueError):
        _ = Unitary(IntState(55, bitsize=8))


def test_unitary_params():
    bloq = _unitary_block_encoding()
    assert bloq.alpha == 1
    assert bloq.epsilon == 0
    assert bloq.ancilla_bitsize == 0
    assert bloq.resource_bitsize == 0

    bloq = _unitary_block_encoding_properties()
    assert bloq.system_bitsize == 1
    assert bloq.alpha == 0.5
    assert bloq.epsilon == 0.01
    assert bloq.ancilla_bitsize == 2
    assert bloq.resource_bitsize == 1


def test_unitary_tensors():
    from_gate = TGate().tensor_contract()
    from_tensors = _unitary_block_encoding().tensor_contract()
    np.testing.assert_allclose(from_gate, from_tensors)


def test_unitary_override_tensors():
    bb = BloqBuilder()
    system = bb.add_register("system", 1)
    ancilla = bb.join(np.array([bb.add(ZeroState()), bb.add(ZeroState())]))
    resource = bb.add(ZeroState())
    system, ancilla, resource = bb.add_t(
        _unitary_block_encoding_properties(),
        system=system,
        ancilla=ancilla,
        resource=cast(Soquet, resource),
    )
    for q in bb.split(cast(Soquet, ancilla)):
        bb.add(ZeroEffect(), q=q)
    bb.add(ZeroEffect(), q=resource)
    bloq = bb.finalize(system=system)

    from_gate = TGate().tensor_contract()
    from_tensors = bloq.tensor_contract()
    np.testing.assert_allclose(from_gate, from_tensors)


def test_unitary_signal_state():
    assert isinstance(_unitary_block_encoding().signal_state.prepare, PrepareIdentity)
    _ = _unitary_block_encoding().signal_state.decompose_bloq()


@pytest.mark.notebook
def test_notebook():
    execute_notebook('unitary')
