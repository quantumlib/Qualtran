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

"""Classes for drawing bloqs with FlameGraph."""
import functools
import tempfile
import pathlib
import subprocess

from typing import List, Optional, Sequence, Union

import IPython.display
import networkx as nx

from qualtran import Bloq
from qualtran.bloqs.basic_gates import TGate
from qualtran.resource_counting.bloq_counts import _compute_sigma
from qualtran.resource_counting.t_counts_from_sigma import t_counts_from_sigma


def _pretty_name(bloq: Bloq) -> str:
    from qualtran.serialization.bloq import _iter_fields

    ret = bloq.pretty_name()
    if bloq.pretty_name.__qualname__.startswith('Bloq.'):
        for field in _iter_fields(bloq):
            ret += f'[{getattr(bloq, field.name)}]'
    return ret


@functools.lru_cache(maxsize=None)
def _t_counts_for_bloq(bloq: Bloq, graph: nx.DiGraph) -> int:
    sigma = _compute_sigma(bloq, graph)
    return t_counts_from_sigma(sigma)


def _keep_if_small(bloq: Bloq) -> bool:
    from qualtran.bloqs.basic_gates import CSwap, Toffoli
    from qualtran.bloqs.mcmt.and_bloq import And

    if isinstance(bloq, (And, Toffoli, CSwap)):
        return True


def _populate_flame_graph_data(
    bloq: Bloq, graph: nx.DiGraph, graph_t: nx.DiGraph, prefix: List[str]
) -> List[str]:
    callees = [x for x in list(graph.successors(bloq)) if _t_counts_for_bloq(x, graph_t) > 0]
    total_t_counts = _t_counts_for_bloq(bloq, graph_t)
    prefix.append(_pretty_name(bloq) + f'(T:{total_t_counts})')
    data = []
    if len(callees) == 0 or (len(callees) == 1 and callees[0] in [TGate(), TGate(is_adjoint=True)]):
        data += [';'.join(prefix) + '\t' + str(total_t_counts)]
    else:
        succ_t_counts = 0
        for succ in callees:
            curr_data = _populate_flame_graph_data(succ, graph, graph_t, prefix)
            succ_t_counts += (
                _t_counts_for_bloq(succ, graph_t) * graph.get_edge_data(bloq, succ)['n']
            )
            data += curr_data * graph.get_edge_data(bloq, succ)['n']
        # assert total_t_counts == succ_t_counts, f'{bloq=}, {total_t_counts=}, {succ_t_counts=}'
    prefix.pop()
    return data


def get_flame_graph_data(
    *bloqs: Bloq,
    file_path: Union[None, pathlib.Path, str] = None,
    keep: Optional[Sequence['Bloq']] = _keep_if_small,
    **kwargs,
) -> List[str]:
    data = []
    for bloq in bloqs:
        call_graph, _ = bloq.call_graph(keep=keep, **kwargs)
        call_graph_t_counts, _ = bloq.call_graph()
        data += _populate_flame_graph_data(bloq, call_graph, call_graph_t_counts, prefix=[])
    if file_path:
        with open(file_path, 'w') as f:
            f.write('\n'.join(data))
    else:
        return data


def get_flame_graph_svg_data(
    *bloqs: Bloq, file_path: Union[None, pathlib.Path, str] = None, **kwargs
) -> Optional[str]:
    data = get_flame_graph_data(*bloqs, **kwargs)

    data_file = tempfile.TemporaryFile(mode='w')
    data_file_path = tempfile.gettempdir() + f'/{data_file.name}'
    flame_graph_path = pathlib.Path(__file__).resolve().parent.parent / "third_party/flamegraph.pl"

    data_file.write('\n'.join(data))
    svg_data = subprocess.run(
        [flame_graph_path, "--countname", "TCounts", f'{data_file_path}'],
        capture_output=True,
        text=True,
    ).stdout
    data_file.close()

    if file_path:
        with open(file_path, 'w') as f:
            f.write(svg_data)
    else:
        return svg_data


def show_flame_graph(*bloqs: Bloq, **kwargs) -> None:
    svg_data = get_flame_graph_svg_data(*bloqs, **kwargs)
    IPython.display.display(IPython.display.SVG(svg_data))
