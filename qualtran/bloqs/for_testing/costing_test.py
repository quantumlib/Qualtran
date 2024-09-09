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

from qualtran.bloqs.for_testing.costing import make_example_costing_bloqs
from qualtran.resource_counting import format_call_graph_debug_text


def test_costing_bloqs():
    algo = make_example_costing_bloqs()
    g, _ = algo.call_graph()
    assert (
        format_call_graph_debug_text(g)
        == """\
Algo -- 2 -> Func1
Algo -- 1 -> Func2
Func1 -- 10 -> H
Func1 -- 10 -> T
Func1 -- 10 -> T†
Func2 -- 100 -> Toffoli"""
    )
