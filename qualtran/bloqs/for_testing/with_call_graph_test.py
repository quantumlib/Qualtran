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

import attrs

from qualtran import Bloq
from qualtran.bloqs.for_testing import TestAtom, TestBloqWithCallGraph


def test_test_bloq_with_call_graph():
    bwcg = TestBloqWithCallGraph()

    def all_atoms_the_same(b: Bloq) -> Optional[Bloq]:
        if isinstance(b, TestAtom):
            return attrs.evolve(b, tag=None)
        return b

    g, sigma = bwcg.call_graph(generalizer=all_atoms_the_same)
    assert len(sigma) == 3
    assert g.number_of_edges() == (3 + 2 + 2)  # level 1 + level 2 + split/join
