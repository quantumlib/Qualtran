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

from qualtran.simulation.tensor.tensor_from_classical import tensor_from_classical_sim
from qualtran.symbolics import is_symbolic

from .bloq_finder import get_bloq_examples


def _run_fn(fn):
    res, err = None, None
    try:
        res = fn()
    except Exception as e:  # pylint: disable=broad-exception-caught
        res = None
        err = str(e)
    return res, err


@pytest.mark.parametrize("be", get_bloq_examples(), ids=lambda be: be.name)
@pytest.mark.timeout(2)
def test_classical_consistent_with_tensor(be):
    LIM = 15

    bloq = be.make()

    n = bloq.signature.n_qubits()
    if is_symbolic(n):
        pytest.skip(f'symbolic qubits: {n}')
    if n > LIM:
        pytest.skip(f'too many qubits: {n=} > {LIM}')

    tensor_direct, err = _run_fn(lambda: bloq.tensor_contract())
    if err is not None:
        pytest.skip(f'no tensor: {err}')

    tensor_classical, err = _run_fn(lambda: tensor_from_classical_sim(bloq))
    if err is not None:
        pytest.skip(f'no classical action: {err}')

    np.testing.assert_allclose(tensor_classical, tensor_direct, rtol=1e-5, atol=1e-5)
