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
"""Deprecated alias for `qualtran.qlt_ir.nodes`. Import from `qualtran.qlt_ir.nodes` instead."""

from qualtran.qlt_ir.nodes import (
    AliasAssignmentNode,
    CArgNode,
    CObjectNode,
    CValueNode,
    LiteralNode,
    LValueNode,
    NestedQArgValue,
    QArgNode,
    QArgValueNode,
    QCallNode,
    QCastNode,
    QDefExternNode,
    QDefImplNode,
    QDefNode,
    QDTypeNode,
)
from qualtran.qlt_ir.nodes import QltASTNode as L1ASTNode
from qualtran.qlt_ir.nodes import QltModule as L1Module
from qualtran.qlt_ir.nodes import QltNodes as L1Nodes
from qualtran.qlt_ir.nodes import QReturnNode, QSignatureEntry, StatementNode, TupleNode

__all__ = [
    'AliasAssignmentNode',
    'CArgNode',
    'CObjectNode',
    'CValueNode',
    'L1ASTNode',
    'L1Module',
    'L1Nodes',
    'LiteralNode',
    'LValueNode',
    'NestedQArgValue',
    'QArgNode',
    'QArgValueNode',
    'QCallNode',
    'QCastNode',
    'QDefExternNode',
    'QDefImplNode',
    'QDefNode',
    'QDTypeNode',
    'QReturnNode',
    'QSignatureEntry',
    'StatementNode',
    'TupleNode',
]
