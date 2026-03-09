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
from typing import cast, Dict, Tuple, Type

import qualtran as qlt
import qualtran.dtype as qdt

from .nodes import CArgNode, CObjectNode, LiteralNode, QDTypeNode


@lru_cache
def get_builtin_qdtype_mapping() -> Dict[str, Type['qdt.QCDType']]:
    """Datatypes that are available without namespacing and with `safe=True`."""
    from qualtran.dtype import BQUInt, CBit, QAny, QBit, QFxp, QInt, QMontgomeryUInt, QUInt

    return {
        k.__name__: cast(Type['qdt.QCDType'], k)
        for k in [BQUInt, QAny, QBit, QInt, QUInt, QFxp, QMontgomeryUInt, CBit]
    }


@lru_cache
def get_builtin_qdtypes() -> Tuple[Type['qdt.QCDType'], ...]:
    return tuple(get_builtin_qdtype_mapping().values())


def reg_to_qdtype_node(reg: 'qlt.Register') -> QDTypeNode:
    """Extract the shaped dtype from a register and return it as an AST node.

    This includes special casing for 'builtin' datatypes. Otherwise, it uses
    the generic `object_to_cobject_node` for the data type object, which may not load if
    `safe=True`.
    """
    dtype = reg.dtype
    if dtype == qdt.QBit():
        dtype_node = CObjectNode('QBit', [])
    elif dtype == qdt.CBit():
        dtype_node = CObjectNode('CBit', [])
    elif isinstance(dtype, (qdt.QAny, qdt.QInt, qdt.QUInt)):
        dtype_node = CObjectNode(
            dtype.__class__.__name__, cargs=[CArgNode(None, LiteralNode(cast(int, dtype.bitsize)))]
        )
    elif isinstance(dtype, qdt.BQUInt):
        dtype_node = CObjectNode(
            'BQUInt',
            cargs=[
                CArgNode(None, LiteralNode(cast(int, dtype.bitsize))),
                CArgNode(None, LiteralNode(cast(int, dtype.iteration_length))),
            ],
        )
    elif isinstance(dtype, qdt.QFxp):
        dtype_node = CObjectNode(
            'QFxp',
            cargs=[
                CArgNode(None, LiteralNode(cast(int, dtype.bitsize))),
                CArgNode(None, LiteralNode(cast(int, dtype.num_frac))),
                CArgNode('signed', LiteralNode(dtype.signed)),
            ],
        )

    else:
        from ._to_cobject_node import to_cobject_node

        cval_node = to_cobject_node(dtype)
        assert isinstance(cval_node, CObjectNode)
        dtype_node = cval_node

    shape = reg.shape_symbolic if reg.shape_symbolic else None
    return QDTypeNode(dtype=dtype_node, shape=shape)  # type: ignore[arg-type]
