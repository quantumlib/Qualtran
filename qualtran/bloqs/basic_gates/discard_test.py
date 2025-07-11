#  Copyright 2025 Google LLC
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

from qualtran import BloqBuilder
from qualtran.bloqs.basic_gates import (
    CNOT,
    Discard,
    DiscardQ,
    MeasZ,
    PlusState,
    ZeroEffect,
    ZeroState,
)


def test_discard():
    bb = BloqBuilder()
    q = bb.add(ZeroState())
    c = bb.add(MeasZ(), q=q)
    bb.add(Discard(), c=c)
    cbloq = bb.finalize()

    # We're allowed to discard classical bits in the classical simulator
    ret = cbloq.call_classically()
    assert ret == ()

    k = cbloq.tensor_contract(superoperator=True)
    np.testing.assert_allclose(k, 1.0, atol=1e-8)


def test_discard_vs_project():
    # Using the ZeroState effect un-physically projects us, giving trace of 0.5
    bb = BloqBuilder()
    q = bb.add(PlusState())
    bb.add(ZeroEffect(), q=q)
    cbloq = bb.finalize()
    k = cbloq.tensor_contract(superoperator=True)
    np.testing.assert_allclose(k, 0.5, atol=1e-8)

    # Measure and discard is trace preserving
    bb = BloqBuilder()
    q = bb.add(PlusState())
    c = bb.add(MeasZ(), q=q)
    bb.add(Discard(), c=c)
    cbloq = bb.finalize()
    k = cbloq.tensor_contract(superoperator=True)
    np.testing.assert_allclose(k, 1.0, atol=1e-8)


def test_discardq():
    # Completely dephasing map
    # https://learning.quantum.ibm.com/course/general-formulation-of-quantum-information/quantum-channels#the-completely-dephasing-channel
    bb = BloqBuilder()
    q = bb.add_register('q', 1)
    env = bb.add(ZeroState())
    q, env = bb.add(CNOT(), ctrl=q, target=env)
    bb.add(DiscardQ(), q=env)
    cbloq = bb.finalize(q=q)
    ss = cbloq.tensor_contract(superoperator=True)

    should_be = np.zeros((2, 2, 2, 2))
    should_be[0, 0, 0, 0] = 1
    should_be[1, 1, 1, 1] = 1

    np.testing.assert_allclose(ss, should_be, atol=1e-8)

    # Classical simulator will not let you throw out qubits
    with pytest.raises(NotImplementedError, match=r'.*classical simulation.*'):
        _ = cbloq.call_classically(q=1)
