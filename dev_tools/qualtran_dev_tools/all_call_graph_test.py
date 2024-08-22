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
