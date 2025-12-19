from functools import lru_cache
from typing import Dict, Type

import qualtran as qlt
import qualtran.dtype as qdt

from ._cargs import object_to_cobject_node
from .nodes import CArgNode, CObjectNode, LiteralNode


@lru_cache
def get_builtin_qdtypes() -> Dict[str, Type[qdt.QCDType]]:
    from qualtran import BQUInt, CBit, QAny, QBit, QFxp, QInt, QMontgomeryUInt, QUInt

    dtypes = {k.__name__: k for k in [BQUInt, QAny, QBit, QInt, QUInt, QFxp, QMontgomeryUInt, CBit]}
    return dtypes
