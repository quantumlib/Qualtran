#  Copyright 2024 Google LLC
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

"""Generate the library-wide call graph from all bloq examples."""
import warnings
from typing import Iterable

import networkx as nx

from qualtran import Bloq, BloqExample, DecomposeNotImplementedError, DecomposeTypeError
from qualtran.bloqs.basic_gates import Rx, Ry, Rz, TGate, Toffoli, XPowGate, YPowGate, ZPowGate
from qualtran.resource_counting import SympySymbolAllocator
from qualtran.resource_counting._call_graph import _build_call_graph
from qualtran.resource_counting._generalization import _make_composite_generalizer
from qualtran.resource_counting.generalizers import (
    generalize_cvs,
    generalize_rotation_angle,
    ignore_alloc_free,
    ignore_cliffords,
    ignore_split_join,
)


def get_all_call_graph(bes: Iterable[BloqExample]):
    """Create a call graph that is the union of all of the bloqs in the list of bloq examples.

    This applies some standard generalizers, and will stop at a larger-than-default set
    of leaf bloqs in accordance with https://github.com/quantumlib/Qualtran/issues/873
    """
    generalize = _make_composite_generalizer(
        ignore_split_join,
        generalize_cvs,
        generalize_rotation_angle,
        ignore_alloc_free,
        ignore_cliffords,
    )

    def keep(b: Bloq) -> bool:
        if b == Toffoli():
            return True

        if b == TGate():
            return True

        if isinstance(b, (Rx, Ry, Rz, XPowGate, YPowGate, ZPowGate)):
            return True

        try:
            _ = b.build_call_graph(SympySymbolAllocator())
        except DecomposeTypeError:
            return True
        except DecomposeNotImplementedError:
            warnings.warn(f"{b} lacks a call graph.")
            return True

        return False

    g = nx.DiGraph()
    ssa = SympySymbolAllocator()

    for be in bes:
        bloq = be.make()
        _build_call_graph(
            bloq=bloq, generalizer=generalize, ssa=ssa, keep=keep, max_depth=None, g=g, depth=0
        )

    return g
