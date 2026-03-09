#  Copyright 2026 Google LLC
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

import functools

from qualtran.l1._ast_visitor_base import L1VisitorBase
from qualtran.l1._parse import parse_module
from qualtran.l1.nodes import CObjectNode


class SimpleVisitor(L1VisitorBase):
    def __init__(self):
        super().__init__()
        self.visited_cobjects = []

    @functools.singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: CObjectNode):
        self.visited_cobjects.append(node.name)
        return super().visit(node)


def test_ast_visitor_base_simple():
    code = """
qdef MainRoutine
from qualtran.bloqs.SomeClass()
[
    qubit: QBit,
] {
    x = XGate
    q2 = x[q=qubit]
    q3 = x[q=q2]
    return [qubit=q3]
}
"""
    module = parse_module(code)
    visitor = SimpleVisitor()

    result = visitor.visit(module)

    # Check that our custom visitor logic fired
    assert visitor.visited_cobjects == ["QBit", "qualtran.bloqs.SomeClass"]

    # Check that the default dictionary-conversion logic returned the expected structure
    assert result['qdefs'][0]['bloq_key'] == "MainRoutine"
    assert result['qdefs'][0]['cobject_from']['name'] == "qualtran.bloqs.SomeClass"
    assert result['qdefs'][0]['body'][0]['bloq_key'] == "XGate"
