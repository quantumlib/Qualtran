#  Copyright 2023 Google LLC
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
from typing import ForwardRef, Set, Type

from qualtran_dev_tools.bloq_finder import get_bloq_classes

from qualtran import Bloq


def _call_graph(bc: Type[Bloq]):
    """Check that a bloq class overrides the right call graph methods.

    - Override `build_call_graph` with canonical type annotations.
    - Don't override `call_graph` or `bloq_counts`.
    """
    call_graph = getattr(bc, 'call_graph')
    if call_graph.__qualname__ != 'Bloq.call_graph':
        print(f'{bc}.call_graph should not be overridden.')
        raise ValueError(str(bc))

    bloq_counts = getattr(bc, 'bloq_counts')
    if bloq_counts.__qualname__ != 'Bloq.bloq_counts':
        print(f'{bc}.bloq_counts should not be overriden.')

    bcg = getattr(bc, 'build_call_graph')
    annot = bcg.__annotations__
    if set(annot.keys()) != {'ssa', 'return'}:
        print(
            f'{bc}.build_call_graph should have one argument named `ssa` '
            f'and a return type annotation'
        )
    if annot['ssa'] != 'SympySymbolAllocator':
        print(f"{bc}.build_call_graph `ssa: 'SympySymbolAllocator'`")
    if annot['return'] != Set[ForwardRef('BloqCountT')]:
        print(f"{bc}.build_call_graph -> 'BloqCountT'")


def report_call_graph_methods():
    bcs = get_bloq_classes()
    for bc in bcs:
        _call_graph(bc)


def main():
    report_call_graph_methods()


if __name__ == '__main__':
    report_call_graph_methods()
