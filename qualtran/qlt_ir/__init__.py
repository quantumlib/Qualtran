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
"""QLT IR: Qualtran's textual intermediate representation (`.qlt` files)."""

from ._ast_to_code import QltASTPrinter
from ._ast_to_code_fast import FastQltASTPrinter
from ._ast_visitor_base import QltVisitorBase
from ._eval import eval_cvalue_node, eval_module
from ._examples import get_qlt_ir_examples, QLT_IR_EXAMPLES, QltExample
from ._parse import dump_ast, parse_module, parse_objectstring, QltParser
from ._parse_eval import load_bloq, load_module, load_objectstring
from ._roundtrip import (
    assert_bloq_roundtrips,
    check_artifacts,
    check_bloq_roundtrip,
    compile_bloq_to_qlt_ir,
    RoundtripArtifacts,
    save_bloq_qlt,
    validate_bloq,
)
from ._to_cobject_node import dump_objectstring, to_cobject_node
from ._to_qlt_ir import dump_qlt_ir, dump_root_qlt_ir, QltModuleBuilder, signature_to_qlt_ir_entries
from ._vm import StandardQualtranArchitectureAgnosticVirtualMachine
from .nodes import QltASTNode, QltModule, QltNodes
