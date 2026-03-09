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
import functools
from typing import Any

import sympy

from ._ast_visitor_base import L1VisitorBase
from .nodes import (
    AliasAssignmentNode,
    CArgNode,
    CObjectNode,
    L1ASTNode,
    L1Module,
    LiteralNode,
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


class L1ASTPrinter(L1VisitorBase):
    """Walk an L1 AST to generate a human-readable representation of the IR."""

    @functools.singledispatchmethod
    def visit(self, node: L1ASTNode) -> Any:
        r = super().visit(node)
        return str(r)

    @visit.register
    def _(self, node: L1Module):
        r = super().visit(node)
        s = '# Qualtran-L1\n# 1.0.0\n\n'
        s += '\n\n'.join(r['qdefs'])
        return s

    @visit.register
    def _(self, node: QDefExternNode):
        r = super().visit(node)
        s = f"extern qdef {r['bloq_key']}\n"
        s += f"from {r['cobject_from']}\n"
        signature = ', '.join(r['qsignature'])
        s += f"[{signature}]"

        return s

    @visit.register
    def _(self, node: QDefImplNode):
        r = super().visit(node)
        s = f"qdef {r['bloq_key']}\n"
        if 'cobject_from' in r:
            s += f"from {r['cobject_from']}\n"
        signature = '\n'.join(f'    {x},' for x in r['qsignature'])
        s += f"[\n{signature}\n] {{\n"

        # retslen = max((len(c.rets) for c in self.calls), default=0)
        # ctorlen = max((len(c.ctor) for c in self.calls), default=0)
        # qargslen = max((len(c.qargs) for c in self.calls), default=0)
        # aliaslen = max((len(a.varname) for a in self.aliases), default=0)
        colwidth = 20

        for stmt in r['body']:
            columnated = ''.join(f'{col:{colwidth}}' for col in stmt)
            s += f"    {columnated}\n"

        s += '}\n'
        return s

    @visit.register
    def _(self, node: QSignatureEntry):
        r = super().visit(node)
        if 'dtype' in r:
            dtype = r['dtype']
        else:
            dts = r['dtypes']
            dt0 = dts[0] if dts[0] is not None else '|'
            dt1 = dts[1] if dts[1] is not None else '|'
            dtype = f"{dt0} -> {dt1}"
        s = f"{r['name']}: {dtype}"
        return s

    @visit.register
    def _(self, node: QDTypeNode):
        r = super().visit(node)

        if 'shape' in r:
            if not all(isinstance(x, (int, sympy.Expr)) for x in r['shape']):
                raise ValueError(f"Invalid shape in QDTypeNode {node}")
            shape_str = ', '.join(repr(x) for x in r['shape'])
            return f"{r['dtype']}[{shape_str}]"

        return f"{r['dtype']}"

    @visit.register
    def _(self, node: AliasAssignmentNode):
        r = super().visit(node)
        return f"{node.alias}", f" = {node.bloq_key}"

    @visit.register
    def _(self, node: QCallNode):
        r = super().visit(node)

        if r['lvalues']:
            rets = ', '.join(r['lvalues'])
        else:
            rets = '|'
        qargs = ', '.join(r['qargs'])
        return (rets, f" = {r['bloq_key']}", f"[{qargs}]")

    @visit.register
    def _(self, node: QReturnNode):
        r = super().visit(node)
        ret_qargs = ', '.join(r['ret_mapping'])
        return ("", "   return", f"[{ret_qargs}]")

    def nested_val_str(self, v):
        if not isinstance(v, list):
            return v

        nvs = ', '.join(self.nested_val_str(v2) for v2 in v)
        return f"[{nvs}]"

    @visit.register
    def _(self, node: QArgNode):
        r = super().visit(node)
        value = self.nested_val_str(r['value'])
        return f"{r['key']}={value}"

    @visit.register
    def _(self, node: QArgValueNode):
        if node.idx:
            idxstr = ', '.join(str(x) for x in node.idx)
            return f'{node.name}[{idxstr}]'
        return f'{node.name}'

    @visit.register
    def _(self, node: CObjectNode):
        r = super().visit(node)
        if r['cargs']:
            carg_str = ', '.join(c for c in r['cargs'])
            return f"{r['name']}({carg_str})"
        return r['name']

    @visit.register
    def _(self, node: CArgNode):
        r = super().visit(node)
        if r['key']:
            return f"{r['key']}={r['value']}"
        return r['value']

    @visit.register
    def _(self, node: LiteralNode):
        return repr(node.value)

    @visit.register
    def _(self, node: TupleNode):
        r = super().visit(node)
        if not len(r['items']):
            return '()'

        items_str = ', '.join(r['items'])
        return f"({items_str})"
