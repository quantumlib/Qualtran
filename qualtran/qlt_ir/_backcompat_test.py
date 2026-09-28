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
"""Tests for the deprecated `qualtran.l1` shim over `qualtran.qlt_ir`."""

import importlib
import types
import warnings

import pytest

import qualtran.qlt_ir as qlt_ir
import qualtran.qlt_ir.nodes as qlt_ir_nodes

with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    import qualtran.l1 as l1
    import qualtran.l1.nodes as l1_nodes


def test_import_warns():
    with pytest.warns(DeprecationWarning, match='renamed to `qualtran.qlt_ir`'):
        importlib.reload(l1)


@pytest.mark.parametrize(
    ('old_name', 'new_name'),
    [
        ('L1ASTPrinter', 'QltASTPrinter'),
        ('L1VisitorBase', 'QltVisitorBase'),
        ('L1Example', 'QltExample'),
        ('L1_EXAMPLES', 'QLT_IR_EXAMPLES'),
        ('get_l1_examples', 'get_qlt_ir_examples'),
        ('compile_bloq_to_l1', 'compile_bloq_to_qlt_ir'),
        ('dump_l1', 'dump_qlt_ir'),
        ('dump_root_l1', 'dump_root_qlt_ir'),
        ('L1ModuleBuilder', 'QltModuleBuilder'),
        ('signature_to_l1_entries', 'signature_to_qlt_ir_entries'),
    ],
)
def test_renamed_symbols_are_aliases(old_name, new_name):
    assert getattr(l1, old_name) is getattr(qlt_ir, new_name)


@pytest.mark.parametrize(
    'name',
    [
        'assert_bloq_roundtrips',
        'check_artifacts',
        'check_bloq_roundtrip',
        'dump_ast',
        'dump_objectstring',
        'eval_cvalue_node',
        'eval_module',
        'load_bloq',
        'load_module',
        'load_objectstring',
        'parse_module',
        'parse_objectstring',
        'RoundtripArtifacts',
        'save_bloq_qlt',
        'StandardQualtranArchitectureAgnosticVirtualMachine',
        'to_cobject_node',
        'validate_bloq',
    ],
)
def test_unrenamed_symbols_are_same_objects(name):
    assert getattr(l1, name) is getattr(qlt_ir, name)


def test_all_is_complete():
    public = {
        n
        for n in dir(l1)
        if not n.startswith('_') and not isinstance(getattr(l1, n), types.ModuleType)
    }
    assert set(l1.__all__) == public


def test_nodes_shim():
    assert l1_nodes.L1ASTNode is qlt_ir_nodes.QltASTNode
    assert l1_nodes.L1Module is qlt_ir_nodes.QltModule
    assert l1_nodes.L1Nodes is qlt_ir_nodes.QltNodes
    for name in l1_nodes.__all__:
        if name.startswith('L1'):
            continue
        assert getattr(l1_nodes, name) is getattr(qlt_ir_nodes, name)
