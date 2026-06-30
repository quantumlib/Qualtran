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

from functools import lru_cache
from typing import cast, Dict, Optional, Tuple, Type

import qualtran as qlt
import qualtran.dtype as qdt

from . import nodes as qualtran_l1_nodes
from .nodes import CArgNode, CObjectNode, L1Nodes, LiteralNode, QDTypeNode


@lru_cache
def get_builtin_qdtype_mapping() -> Dict[str, Type['qdt.QCDType']]:
    """Datatypes that are available without namespacing and with `safe=True`."""
    from qualtran.dtype import (
        BQUInt,
        CBit,
        QAny,
        QBit,
        QFxp,
        QInt,
        QIntOnesComp,
        QIntSignMag,
        QMontgomeryUInt,
        QUInt,
    )

    return {
        k.__name__: cast(Type['qdt.QCDType'], k)
        for k in [
            BQUInt,
            QAny,
            QBit,
            QInt,
            QUInt,
            QFxp,
            QMontgomeryUInt,
            CBit,
            QIntOnesComp,
            QIntSignMag,
        ]
    }


@lru_cache
def get_builtin_qdtypes() -> Tuple[Type['qdt.QCDType'], ...]:
    return tuple(get_builtin_qdtype_mapping().values())


def to_qdtype_node(
    dtype: 'qlt.QCDType',
    *,
    nodes: L1Nodes = qualtran_l1_nodes,
) -> CObjectNode:
    """Convert a QCDType object to its equivalent AST node.

    This includes special casing for 'builtin' datatypes. Otherwise, it uses
    the generic `object_to_cobject_node` for the data type object, which may not load if
    `safe=True`.

    Args:
        dtype: The data type to serialize.
        nodes: The module providing the AST node constructors.
    """
    if dtype == qdt.QBit():
        dtype_node = nodes.CObjectNode('QBit', [])
    elif dtype == qdt.CBit():
        dtype_node = nodes.CObjectNode('CBit', [])
    elif isinstance(dtype, (qdt.QAny, qdt.QInt, qdt.QUInt, qdt.QIntSignMag, qdt.QIntOnesComp)):
        dtype_node = nodes.CObjectNode(
            dtype.__class__.__name__,
            cargs=[nodes.CArgNode(None, nodes.LiteralNode(cast(int, dtype.bitsize)))],
        )
    elif isinstance(dtype, qdt.BQUInt):
        dtype_node = nodes.CObjectNode(
            'BQUInt',
            cargs=[
                nodes.CArgNode(None, nodes.LiteralNode(cast(int, dtype.bitsize))),
                nodes.CArgNode(None, nodes.LiteralNode(cast(int, dtype.iteration_length))),
            ],
        )
    elif isinstance(dtype, qdt.QFxp):
        dtype_node = nodes.CObjectNode(
            'QFxp',
            cargs=[
                nodes.CArgNode(None, nodes.LiteralNode(cast(int, dtype.bitsize))),
                nodes.CArgNode(None, nodes.LiteralNode(cast(int, dtype.num_frac))),
                nodes.CArgNode('signed', nodes.LiteralNode(dtype.signed)),
            ],
        )

    else:
        from ._to_cobject_node import to_cobject_node

        cval_node = to_cobject_node(dtype, nodes=nodes)
        assert isinstance(cval_node, nodes.CObjectNode)
        dtype_node = cval_node

    return dtype_node


def reg_to_qdtype_node(
    reg: 'qlt.Register',
    *,
    nodes: L1Nodes = qualtran_l1_nodes,
) -> QDTypeNode:
    """Extract the shaped dtype from a register and return it as an AST node.

    This includes special casing for 'builtin' datatypes. Otherwise, it uses
    the generic `object_to_cobject_node` for the data type object, which may not load if
    `safe=True`.
    """
    dtype_node = to_qdtype_node(reg.dtype, nodes=nodes)
    shape = reg.shape if reg._shape else None
    return nodes.QDTypeNode(dtype=dtype_node, shape=shape)
