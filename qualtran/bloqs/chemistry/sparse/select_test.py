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

from qualtran.bloqs.chemistry.sparse import SelectSparse
from qualtran.resource_counting import get_bloq_counts_graph


def _make_sparse_select():
    from qualtran.bloqs.chemistry.sparse import SelectSparse

    return SelectSparse(10)


def test_sparse_select():
    sel = SelectSparse(10)


def test_sparse_select_bloq_counts():
    _, sigma = get_bloq_counts_graph(SelectSparse(10))
