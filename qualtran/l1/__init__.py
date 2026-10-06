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
"""Deprecated alias for `qualtran.qlt_ir`.

"Qualtran L1" was the working name for what is now called QLT IR. This module
re-exports the public API of `qualtran.qlt_ir` under the old names and will be
removed in a future release. Import from `qualtran.qlt_ir` instead.
"""

import warnings

from qualtran.qlt_ir import assert_bloq_roundtrips, check_artifacts, check_bloq_roundtrip
from qualtran.qlt_ir import compile_bloq_to_qlt_ir as compile_bloq_to_l1
from qualtran.qlt_ir import dump_ast, dump_objectstring
from qualtran.qlt_ir import dump_qlt_ir as dump_l1
from qualtran.qlt_ir import dump_root_qlt_ir as dump_root_l1
from qualtran.qlt_ir import eval_cvalue_node, eval_module
from qualtran.qlt_ir import get_qlt_ir_examples as get_l1_examples
from qualtran.qlt_ir import (
    load_bloq,
    load_module,
    load_objectstring,
    parse_module,
    parse_objectstring,
)
from qualtran.qlt_ir import QLT_IR_EXAMPLES as L1_EXAMPLES
from qualtran.qlt_ir import QltASTPrinter as L1ASTPrinter
from qualtran.qlt_ir import QltExample as L1Example
from qualtran.qlt_ir import QltModuleBuilder as L1ModuleBuilder
from qualtran.qlt_ir import QltVisitorBase as L1VisitorBase
from qualtran.qlt_ir import RoundtripArtifacts, save_bloq_qlt
from qualtran.qlt_ir import signature_to_qlt_ir_entries as signature_to_l1_entries
from qualtran.qlt_ir import (
    StandardQualtranArchitectureAgnosticVirtualMachine,
    to_cobject_node,
    validate_bloq,
)

warnings.warn(
    "`qualtran.l1` has been renamed to `qualtran.qlt_ir`; "
    "`qualtran.l1` will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    'assert_bloq_roundtrips',
    'check_artifacts',
    'check_bloq_roundtrip',
    'compile_bloq_to_l1',
    'dump_ast',
    'dump_l1',
    'dump_objectstring',
    'dump_root_l1',
    'eval_cvalue_node',
    'eval_module',
    'get_l1_examples',
    'L1_EXAMPLES',
    'L1ASTPrinter',
    'L1Example',
    'L1ModuleBuilder',
    'L1VisitorBase',
    'load_bloq',
    'load_module',
    'load_objectstring',
    'parse_module',
    'parse_objectstring',
    'RoundtripArtifacts',
    'save_bloq_qlt',
    'signature_to_l1_entries',
    'StandardQualtranArchitectureAgnosticVirtualMachine',
    'to_cobject_node',
    'validate_bloq',
]
