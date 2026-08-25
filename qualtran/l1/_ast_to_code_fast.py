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
"""Direct recursive AST-to-code formatting for Qualtran-L1.

This module converts Qualtran-L1 AST nodes directly into their textual code
representation using direct function calls rather than singledispatch/visitor
metaprogramming.
"""

import numbers
from typing import Any, cast, Tuple

from .nodes import (
    AliasAssignmentNode,
    CArgNode,
    CObjectNode,
    CValueNode,
    L1ASTNode,
    L1Module,
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
    QReturnNode,
    QSignatureEntry,
    StatementNode,
    TupleNode,
)


def format_cvalue(node: CValueNode) -> str:
    """Format a classical value node."""
    t = type(node)
    if t is CObjectNode:
        return format_cobject(cast(CObjectNode, node))
    elif t is LiteralNode:
        return format_literal(cast(LiteralNode, node))
    elif t is TupleNode:
        return format_tuple(cast(TupleNode, node))
    return node.canonical_str()


def format_literal(node: LiteralNode) -> str:
    """Format a literal value node."""
    return repr(node.value)


def format_tuple(node: TupleNode) -> str:
    """Format a tuple of classical values."""
    if not node.items:
        return '()'
    items_str = ', '.join(format_cvalue(i) for i in node.items)
    return f"({items_str})"


def format_carg(node: CArgNode) -> str:
    """Format a classical argument node."""
    val_str = format_cvalue(node.value)
    if node.key:
        return f"{node.key}={val_str}"
    return val_str


def format_cobject(node: CObjectNode) -> str:
    """Format a classical object node."""
    if node.cargs:
        carg_str = ', '.join(format_carg(c) for c in node.cargs)
        return f"{node.name}({carg_str})"
    return node.name


def format_qdtype(node: QDTypeNode) -> str:
    """Format a quantum data type node."""
    dtype_str = format_cobject(node.dtype)
    if node.shape is not None:
        if not all(isinstance(x, numbers.Integral) for x in node.shape):
            raise ValueError(f"Invalid shape in QDTypeNode {node}")
        shape_str = ', '.join(str(int(x)) for x in node.shape)
        return f"{dtype_str}[{shape_str}]"
    return dtype_str


def format_signature_entry(node: QSignatureEntry) -> str:
    """Format a quantum signature entry node."""
    if isinstance(node.dtype, QDTypeNode):
        dtype_str = format_qdtype(node.dtype)
    else:
        dt0 = format_qdtype(node.dtype[0]) if node.dtype[0] is not None else '|'
        dt1 = format_qdtype(node.dtype[1]) if node.dtype[1] is not None else '|'
        dtype_str = f"{dt0} -> {dt1}"
    s = f"{node.name}: {dtype_str}"
    if node.annotation is not None:
        s += f" @ {node.annotation.canonical_str()}"
    return s


def format_lvalue(node: LValueNode) -> str:
    """Format an L-value node."""
    if node.annotation is not None:
        return f"{node.name} @ {node.annotation.canonical_str()}"
    return node.name


def format_qarg_value(node: QArgValueNode) -> str:
    """Format a quantum argument value node."""
    if node.idx:
        idxstr = ', '.join(str(x) for x in node.idx)
        return f'{node.name}[{idxstr}]'
    return f'{node.name}'


def format_nested_qarg_value(v: NestedQArgValue) -> str:
    """Format nested quantum argument values (arrays/lists of soquets)."""
    if isinstance(v, (list, tuple)):
        nvs = ', '.join(format_nested_qarg_value(v2) for v2 in v)
        return f"[{nvs}]"
    elif isinstance(v, QArgValueNode):
        return format_qarg_value(v)
    elif isinstance(v, str):
        return v
    return str(v)


def format_qarg(node: QArgNode) -> str:
    """Format a quantum argument keyword pair node."""
    value_str = format_nested_qarg_value(node.value)
    s = f"{node.key}={value_str}"
    if node.annotation is not None:
        s += f" @ {node.annotation.canonical_str()}"
    return s


def format_alias_assignment(node: AliasAssignmentNode) -> Tuple[str, str]:
    """Format an alias assignment statement node."""
    return f"{node.alias}", f" = {node.bloq_key}"


def format_qcall(node: QCallNode) -> Tuple[str, str, str]:
    """Format a quantum call statement node."""
    if node.lvalues:
        rets = ', '.join(format_lvalue(lv) for lv in node.lvalues)
    else:
        rets = '|'
    qargs = ', '.join(format_qarg(qa) for qa in node.qargs)
    s = f" = {node.bloq_key}"
    if node.annotation is not None:
        s += f" @ {node.annotation.canonical_str()}"
    return (rets, s, f"[{qargs}]")


def format_qreturn(node: QReturnNode) -> Tuple[str, str, str]:
    """Format a return statement node."""
    ret_qargs = ', '.join(format_qarg(qa) for qa in node.ret_mapping)
    return ("", "   return", f"[{ret_qargs}]")


def format_statement(node: StatementNode) -> Tuple[str, ...]:
    """Format a statement node into its column components."""
    t = type(node)
    if t is QCallNode:
        return format_qcall(cast(QCallNode, node))
    elif t is AliasAssignmentNode:
        return format_alias_assignment(cast(AliasAssignmentNode, node))
    elif t is QReturnNode:
        return format_qreturn(cast(QReturnNode, node))
    raise TypeError(f"Unknown StatementNode type: {type(node).__name__}")


def format_qdef_impl(node: QDefImplNode) -> str:
    """Format an implemented qdef node."""
    s = f"qdef {node.bloq_key}\n"
    if node.cobject_from is not None:
        s += f"from {format_cobject(node.cobject_from)}\n"
    signature = '\n'.join(f'    {format_signature_entry(x)},' for x in node.qsignature)
    s += f"[\n{signature}\n] {{\n"

    colwidth = 20
    for stmt in node.body:
        stmt_cols = format_statement(stmt)
        columnated = ''.join(f'{col:{colwidth}}' for col in stmt_cols)
        s += f"    {columnated}\n"

    s += '}\n'
    return s


def format_qdef_extern(node: QDefExternNode) -> str:
    """Format an external qdef node."""
    s = f"extern qdef {node.bloq_key}\n"
    if node.cobject_from is not None:
        s += f"from {format_cobject(node.cobject_from)}\n"
    signature = ', '.join(format_signature_entry(sig_entry) for sig_entry in node.qsignature)
    s += f"[{signature}]"
    return s


def format_qcast(node: QCastNode) -> str:
    """Format a qcast node."""
    s = f"qcast {node.bloq_key}\n"
    signature = ', '.join(format_signature_entry(sig_entry) for sig_entry in node.qsignature)
    s += f"[{signature}]"
    return s


def format_qdef(node: QDefNode) -> str:
    """Format any qdef node."""
    t = type(node)
    if t is QDefImplNode:
        return format_qdef_impl(cast(QDefImplNode, node))
    elif t is QDefExternNode:
        return format_qdef_extern(cast(QDefExternNode, node))
    elif t is QCastNode:
        return format_qcast(cast(QCastNode, node))
    raise TypeError(f"Unknown QDefNode type: {type(node).__name__}")


def format_module(node: L1Module) -> str:
    """Format an entire L1Module node into L1 source code."""
    s = '# Qualtran-L1\n# 1.0.0\n\n'
    s += '\n\n'.join(format_qdef(qdef) for qdef in node.qdefs)
    return s


def format_node(node: L1ASTNode) -> Any:
    """Format any L1 AST node."""
    t = type(node)
    if t is L1Module:
        return format_module(cast(L1Module, node))
    elif t is QDefImplNode:
        return format_qdef_impl(cast(QDefImplNode, node))
    elif t is QDefExternNode:
        return format_qdef_extern(cast(QDefExternNode, node))
    elif t is QCastNode:
        return format_qcast(cast(QCastNode, node))
    elif t is QSignatureEntry:
        return format_signature_entry(cast(QSignatureEntry, node))
    elif t is QDTypeNode:
        return format_qdtype(cast(QDTypeNode, node))
    elif t is LValueNode:
        return format_lvalue(cast(LValueNode, node))
    elif t is AliasAssignmentNode:
        return format_alias_assignment(cast(AliasAssignmentNode, node))
    elif t is QCallNode:
        return format_qcall(cast(QCallNode, node))
    elif t is QReturnNode:
        return format_qreturn(cast(QReturnNode, node))
    elif t is QArgNode:
        return format_qarg(cast(QArgNode, node))
    elif t is QArgValueNode:
        return format_qarg_value(cast(QArgValueNode, node))
    elif t is CObjectNode:
        return format_cobject(cast(CObjectNode, node))
    elif t is CArgNode:
        return format_carg(cast(CArgNode, node))
    elif t is LiteralNode:
        return format_literal(cast(LiteralNode, node))
    elif t is TupleNode:
        return format_tuple(cast(TupleNode, node))
    raise TypeError(f"Unknown AST node type: {type(node).__name__}")


class FastL1ASTPrinter:
    """Walk an L1 AST using direct recursive functions to generate IR text."""

    def visit(self, node: L1ASTNode) -> Any:
        return format_node(node)
