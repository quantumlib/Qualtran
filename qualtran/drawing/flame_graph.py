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

"""Classes for drawing bloqs with FlameGraph. This relies on third party flamegraph.pl"""
import functools
import pathlib
import subprocess
import tempfile
from typing import Any, Callable, List, Optional, Union

import networkx as nx
import numpy as np
import sympy

from qualtran import Bloq
from qualtran.resource_counting._call_graph import _compute_sigma
from qualtran.resource_counting.t_counts_from_sigma import t_counts_from_sigma


def _pretty_arg(val: Any) -> str:
    if isinstance(val, (tuple, np.ndarray)):
        return f'{val.shape if isinstance(val, np.ndarray) else len(val)}'
    if isinstance(val, Bloq):
        return _pretty_name(val)
    if isinstance(val, float):
        if np.isclose(val, 0):
            val = 0
        return f'{val:0.2g}'
    return f'{val}'


def _pretty_name(bloq: Bloq) -> str:
    """Hack to get a reasonably concise, reasonably informative description of a bloq.

    This should be removed once we have a better way to get a descriptive and concise
    representation for a bloq. See https://github.com/quantumlib/Qualtran/issues/791
    """

    from qualtran.serialization.bloq import _iter_fields

    ret = bloq.pretty_name()
    if bloq.pretty_name.__qualname__.startswith('Bloq.') and bloq.__str__.__qualname__.startswith(
        'Bloq.'
    ):
        for field in _iter_fields(bloq):
            ret += f'[{_pretty_arg(getattr(bloq, field.name))}]'
    return ret


@functools.lru_cache(maxsize=1024)
def _t_counts_for_bloq(bloq: Bloq, graph: nx.DiGraph) -> Union[int, sympy.Expr]:
    sigma = _compute_sigma(bloq, graph)
    return t_counts_from_sigma(sigma)


def _keep_if_small(bloq: Bloq) -> bool:
    from qualtran.bloqs.basic_gates import Toffoli, TwoBitCSwap
    from qualtran.bloqs.mcmt.and_bloq import And

    if isinstance(bloq, (And, Toffoli, TwoBitCSwap)):
        return True
    return False


def _is_leaf_node(callees: List[Bloq]) -> bool:
    from qualtran.bloqs.basic_gates import TGate

    return len(callees) == 0 or (
        len(callees) == 1 and callees[0] in [TGate(), TGate(is_adjoint=True)]
    )


def _populate_flame_graph_data(
    bloq: Bloq, graph: nx.DiGraph, graph_t: nx.DiGraph, prefix: List[str]
) -> List[str]:
    """Populates data for the flame graph.

    Args:
        bloq: Bloq to get the flame graph data for.
        graph: Callgraph of `bloq` with custom kwargs so users can control the depth / leaf nodes
            for the flame graph. This is the graph we do a DFS ordering on.
        graph_t: Callgraph to use to derive T-complexity of the Bloq. Ideally, we should just be able
            to invoke the `bloq.t_complexity().t` but this hides the T-costs due to rotations, so we
            use a temporary solution to invoke `_t_counts_for_bloq(bloq, graph_t)`. The graph is not
            mutated over the course of DFS and hence can be used as a hash key for better performance.
        prefix: A string that represents the bloqs visited in the path so far. This is mutated as the
            graph is traversed and represents the current path from the root node to the current node
            in the DFS traversal. This is used to populate the flame graph data once we hit leaf nodes
            in `graph`.

    Returns:
        The Flame graph data for the subgraph of `graph` for which `bloq` is a root node.
    """

    callees = [x for x in list(graph.successors(bloq)) if _t_counts_for_bloq(x, graph_t) > 0]
    total_t_counts = _t_counts_for_bloq(bloq, graph_t)
    prefix.append(_pretty_name(bloq) + f'(T:{total_t_counts})')
    data = []
    if _is_leaf_node(callees):
        data += [';'.join(prefix) + '\t' + str(total_t_counts)]
    else:
        succ_t_counts = 0
        for succ in callees:
            curr_data = _populate_flame_graph_data(succ, graph, graph_t, prefix)
            succ_t_counts += (
                _t_counts_for_bloq(succ, graph_t) * graph.get_edge_data(bloq, succ)['n']
            )
            data += curr_data * graph.get_edge_data(bloq, succ)['n']
        # TODO: This assertion should be enabled once, for each bloq, we verify that
        # `assert_equivalent_bloq_example_counts` is True for `TGate`. This is currently not True
        # and is tracked in https://github.com/quantumlib/Qualtran/issues/858
        # assert total_t_counts == succ_t_counts, f'{bloq=}, {total_t_counts=}, {succ_t_counts=}'
    prefix.pop()
    return data


def get_flame_graph_data(
    *bloqs: Bloq,
    file_path: Union[None, pathlib.Path, str] = None,
    keep: Optional[Callable[['Bloq'], bool]] = _keep_if_small,
    **kwargs,
) -> List[str]:
    """Get the flame graph data for visualizing T-costs distribution of a sequence of bloqs.

    For each bloq in the input, this will do a DFS ordering over all edges in the DAG and
    add an entry corresponding to each leaf node in the call graph. The string representation
    added for a leaf node encodes the entire path taken from the root node to the leaf node
    and is repeated a number of times that's equivalent to the weight of that path. Thus, the
    length of the output would be roughly equal to the number of T-gates in the Bloq and can be
    very high. If you want to limit the output size, consider specifying a `keep` predicate where
    the leaf nodes are higher level Bloqs with a larger T-count weight.

    Args:
        bloqs: Bloqs to plot the flame graph for
        file_path: If specified, the output is stored at the file.
        keep: A predicate to determine the leaf nodes in the call graph. The flame graph would use
            these Bloqs as leaf nodes and thus would not contain decompositions for these nodes.
        **kwargs: Additional arguments to be passed to `bloq.call_graph`, like generalizers etc.

    Returns:
        A list of strings, one for each path from root node to the leaf node in the call graph x
        the weight of the path, that can be passed to the `third_party/flame_graph/flame_graph.pl`
        script.
    """
    from qualtran.resource_counting.generalizers import cirq_to_bloqs

    data = []
    for bloq in bloqs:
        call_graph, _ = bloq.call_graph(keep=keep, **kwargs, generalizer=cirq_to_bloqs)
        call_graph_t_counts, _ = bloq.call_graph()
        data += _populate_flame_graph_data(bloq, call_graph, call_graph_t_counts, prefix=[])
    if file_path:
        with open(file_path, 'w') as f:
            f.write('\n'.join(data))
    return data


def get_flame_graph_svg_data(
    *bloqs: Bloq, file_path: Union[None, pathlib.Path, str] = None, **kwargs
) -> Optional[str]:
    """Invokes the `third_party/flamegraph/flamegraph.pl` using data from `get_flame_graph_data`."""

    data = get_flame_graph_data(*bloqs, **kwargs)

    data_file = tempfile.NamedTemporaryFile(mode='w')
    flame_graph_path = (
        pathlib.Path(__file__).resolve().parent.parent / "third_party/flamegraph/flamegraph.pl"
    )

    data_file.write('\n'.join(data))
    data_file.flush()
    svg_data = subprocess.run(
        [flame_graph_path, "--countname", "TCounts", f'{data_file.name}'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    data_file.close()

    if file_path:
        with open(file_path, 'w') as f:
            f.write(svg_data)
    return svg_data
