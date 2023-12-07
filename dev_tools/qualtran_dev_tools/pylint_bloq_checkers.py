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

from astroid.nodes import Arguments, ClassDef, FunctionDef
from pylint.checkers import BaseChecker
from pylint.lint import PyLinter


class BloqClassicalSimChecker(BaseChecker):
    name = "bloq-classical"
    msgs = {
        "W0001": (
            "Method `on_classical_vals` in %s should have keyword-only arguments.",
            'bloq-classical-args',
            None,
        ),
        "W0002": (
            "Override `on_classical_vals`, not `call_classically`.",
            'bloq-classical-override',
            None,
        ),
    }

    def __init__(self, linter: Optional[PyLinter] = None):
        super().__init__(linter)
        self._bloq_classes = set()

    def visit_classdef(self, node: ClassDef):
        for ancestor in node.ancestors():
            if ancestor.name == "Bloq":
                self._bloq_classes.add(node)

    def visit_functiondef(self, node: FunctionDef):
        if node.name == "on_classical_vals" and node.parent in self._bloq_classes:
            args: Arguments = node.args
            argnames = [arg.name for arg in args.arguments]
            if not argnames == ['self']:
                self.add_message("bloq-classical-args", node=node, args=node.parent.name)
            if args.vararg:
                self.add_message("bloq-classical-args", node=node, args=node.parent.name)
            if args.kwarg:
                # this one is less serious
                self.add_message("bloq-classical-args", node=node, args=node.parent.name)

        if node.name == 'call_classically' and node.parent in self._bloq_classes:
            self.add_message('bloq-classical-override', node=node)


def register(linter):
    linter.register_checker(BloqClassicalSimChecker(linter))
