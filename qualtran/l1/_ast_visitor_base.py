#  Copyright 2025 Google LLC
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

import abc
import functools
from typing import Any, cast, Dict

import attrs

from ._parse import (
    CObjectNode,
    L1ASTNode,
    L1Module,
    QArgNode,
    QArgValueNode,
    QCallNode,
    QDefExternNode,
    QDefImplNode,
    QDTypeNode,
    QReturnNode,
    QSignatureEntry,
    TupleNode,
)


class L1VisitorBase(metaclass=abc.ABCMeta):
    @functools.singledispatchmethod
    def visit(self, node: L1ASTNode) -> Dict[str, Any]:
        record = attrs.asdict(cast(Any, node), recurse=False)
        record = {k: self.visit(v) if isinstance(v, L1ASTNode) else v for k, v in record.items()}
        return record

    @visit.register
    def _(self, node: L1Module):
        record = {'qdefs': [self.visit(qdef) for qdef in node.qdefs]}
        return record

    @visit.register
    def _(self, node: QDefImplNode):
        record: Dict[str, Any] = {'bloq_key': node.bloq_key}
        record['qsignature'] = [self.visit(sig_entry) for sig_entry in node.qsignature]
        record['body'] = [self.visit(stmt) for stmt in node.body]

        if node.cobject_from is not None:
            record['cobject_from'] = self.visit(node.cobject_from)

        return record

    @visit.register
    def _(self, node: QDefExternNode):
        record: Dict[str, Any] = {'bloq_key': node.bloq_key}
        record['qsignature'] = [self.visit(sig_entry) for sig_entry in node.qsignature]
        record['cobject_from'] = self.visit(node.cobject_from)
        return record

    @visit.register
    def _(self, node: QSignatureEntry):
        record: Dict[str, Any] = {'name': node.name}
        if isinstance(node.dtype, QDTypeNode):
            record['dtype'] = self.visit(node.dtype)
        else:
            record['dtypes'] = (
                self.visit(node.dtype[0]) if node.dtype[0] is not None else None,
                self.visit(node.dtype[1]) if node.dtype[1] is not None else None,
            )
        return record

    @visit.register
    def _(self, node: QDTypeNode):
        record: Dict[str, Any] = {'dtype': self.visit(node.dtype)}
        if node.shape is not None:
            record['shape'] = node.shape
        return record

    @visit.register
    def _(self, node: QCallNode):
        record: Dict[str, Any] = {
            'bloq_key': node.bloq_key,
            'lvalues': [lv for lv in node.lvalues],
            'qargs': [self.visit(qa) for qa in node.qargs],
        }
        return record

    @visit.register
    def _(self, node: QReturnNode):
        record: Dict[str, Any] = {'ret_mapping': [self.visit(qa) for qa in node.ret_mapping]}
        return record

    def visit_nested_qarg_value(self, v):
        if isinstance(v, QArgValueNode):
            return self.visit(v)
        return [self.visit_nested_qarg_value(v2) for v2 in v]

    @visit.register
    def _(self, node: QArgNode):
        record: Dict[str, Any] = {'key': node.key}
        if isinstance(node.value, QArgValueNode):
            record['value'] = self.visit(node.value)
        else:
            record['value'] = [self.visit_nested_qarg_value(v) for v in node.value]
        return record

    @visit.register
    def _(self, node: CObjectNode):
        record: Dict[str, Any] = {
            'name': node.name,
            'cargs': [self.visit(carg) for carg in node.cargs],
        }
        return record

    @visit.register
    def _(self, node: TupleNode):
        record: Dict[str, Any] = {'items': [self.visit(i) for i in node.items]}
        return record
