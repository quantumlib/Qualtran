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
from typing import cast, Dict, Tuple, Type, TYPE_CHECKING

if TYPE_CHECKING:
    import qualtran.dtype as qdt


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
