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
import logging
from typing import Dict

import qualtran as qlt
from qualtran.l1._eval import BloqKey, eval_module
from qualtran.l1._parse import CObjectNode, parse_module

from ._eval import eval_cvalue_node
from ._parse import parse_objectstring

logger = logging.getLogger(__name__)


def load_objectstring(objectstring: str, *, safe: bool = True) -> object:
    cobject_node = parse_objectstring(objectstring)
    return eval_cvalue_node(cobject_node, safe=safe)


def load_module(l1_code: str, *, safe: bool = True) -> Dict[BloqKey, 'qlt.Bloq']:
    m = parse_module(l1_code)
    return eval_module(m, safe=safe)


def load_bloq(bloq_str: str, *, safe: bool = True) -> 'qlt.Bloq':
    if not isinstance(x := load_objectstring(bloq_str, safe=safe), qlt.Bloq):
        raise TypeError(f"{bloq_str} evaluated to {x!r}, which is not a `qualtran.Bloq`.")
    return x
