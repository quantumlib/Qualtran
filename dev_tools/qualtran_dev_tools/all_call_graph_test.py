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

import networkx as nx
import pytest

from .all_call_graph import get_all_call_graph
from .bloq_finder import get_bloq_examples


@pytest.mark.slow
def test_get_all_call_graph():
    # This test generates a union of the call graphs of every bloq example in the library.
    # This test makes sure that there aren't any bloq examples with broken call graphs.
    bes = get_bloq_examples()
    g = get_all_call_graph(bes)
    res = list(nx.simple_cycles(g))
    assert res == []


@pytest.mark.parametrize("be", get_bloq_examples(), ids=lambda be: be.name)
def test_classical_tensor(be):
    import numpy as np
    from qualtran.simulation.tensor.tensor_from_classical import tensor_from_classical_sim
    from qualtran.symbolics import is_symbolic

    if be.name in ['rsa_pe_small']:
        pytest.skip('skiplist')

    LIM = 9

    bloq = be.make()

    n = bloq.signature.n_qubits()
    if is_symbolic(n):
        pytest.skip(f'symbolic qubits: {n=}')
    if n > LIM:
        pytest.skip(f'too many qubits: {n=}')

    try:
        tensor_direct = bloq.tensor_contract()
    except Exception as e:
        pytest.skip(f'no tensor: {e}')

    try:
        tensor_classical = tensor_from_classical_sim(bloq)
    except NotImplementedError as e:
        pytest.skip(str(e))

    np.testing.assert_allclose(tensor_classical, tensor_direct, rtol=1e-5, atol=1e-5)
