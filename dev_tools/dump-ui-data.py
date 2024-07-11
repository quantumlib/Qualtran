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
import json
from typing import Any, Dict, TypedDict, List, Tuple

import attrs

from qualtran import Bloq, Soquet
from qualtran.bloqs.factoring.mod_exp import ModExp
from qualtran.drawing import get_musical_score_data
from qualtran.drawing.musical_score import MusicalScoreEncoder, MusicalScoreData
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran_dev_tools.parse_docstrings import get_markdown_docstring_lines


def get_bloq_classname(bloq: Bloq) -> str:
    """The class name of the bloq, can be used as a title."""
    return bloq.__class__.__name__


def get_bloq_docstring(bloq: Bloq) -> str:
    """The bloq docstring formatted as a markdown string.

    Possible schema extensions include: keeping the docstring factored by section.
    """
    return '\n'.join(get_markdown_docstring_lines(bloq.__class__))


def get_bloq_attributes(bloq: Bloq) -> Dict[str, Any]:
    """Get the class attributes (member variables) of a bloq.

    Each bloq is configured with one or more attributes. This returns a dict
    mapping the string attribute name to its arbitrary Python value.

    Limitations: the values can be arbitrary python values which may not be serializable.
    If we're using this for display purposes, the serialization routines will need to learn
    how to nicely serialize the other types of values that may appear.
    """
    return attrs.asdict(bloq)


class CalleeData(TypedDict):
    """Data as part of `get_callees`."""

    bloq_ref: str
    """A reference to another bloq. We'll have to figure out how to do this. Right now,
    it just uses the successor bloq's repr."""

    edge_data: Dict[str, Any]
    """Arbitrary data associated with the caller-callee relationship. Currently,
    it's a dictionary with one entry mapping the key 'n' to the number of times the
    caller calls the callee."""


def get_callees(bloq: Bloq) -> List[CalleeData]:
    """Get a list of successor 'callee' bloqs. See the docs CalleData.

    Limitations: We'll need to figure out how to reference other bloq instances.
    """

    call_graph, _ = bloq.call_graph(max_depth=1, generalizer=ignore_split_join)
    callees = []
    for child_bloq, edge_data in sorted(call_graph.succ[bloq].items(), key=lambda x: str(x[0])):
        callees.append({
            'bloq_ref': repr(child_bloq),  # TODO: how do we link to other bloqs
            'edge_data': edge_data,
        })

    return callees


class SerializableSoquet(TypedDict):
    """The compute graph is a list of directed edges between nodes. This describes the schame
    for each node. Each field is required to uniquely address a particular node."""

    binst: str  # TODO probably want to make this more structured
    """Each bloq instance can have multiple soquets. Connections (edges in the compute
    graph) are defined between soquets, but when drawing the diagram; all the soquets
    for a given bloq instance should be grouped."""

    reg_name: str
    """The name of the register on the bloq instance."""

    reg_side: str
    """Oh boy. So, registers can be LEFT, RIGHT, or THRU registers. In the first two cases,
    it's possible to have two distinct soquets with the same register name that are differentiated
    by which side they're on. If it's a THRU register, the soquet can have both incoming
    and outgoing edges."""

    idx: Tuple[int, ...]
    """Each register can optionally host an n-dimensional array of soquets. This is the particular
    index into the ndarray. Or the empty list if the register is not an ndarray of soquets.
    """


def _make_soquet_serializable(soq: Soquet) -> SerializableSoquet:
    """Serialize each node in the graph processed as part of `get_decomposition_dag`."""
    return {
        'binst': repr(soq.binst),  # TODO: structured data
        'reg_name': soq.reg.name,
        'reg_side': str(soq.reg.side),
        'idx': soq.idx
    }


def get_decomposition_dag(bloq: Bloq) -> List[Tuple[SerializableSoquet, SerializableSoquet]]:
    """Return a list of edges that describes the flow of quantum information in a bloq's decomposition.

    A bloq's decomposition is a graph where the nodes are "soquets". There are usually multiple
    soquets for a given bloq instance that should be grouped when displaying the decomposition
    graph. I.e. they should be grouped according to the `"binst"` key.

    See the documentaiton for `SerializableSoquet` for the schema of the nodes. Each directed edge
    is a pair of nodes, where quantum data flows from the first node to the second one.
    """

    cbloq = bloq.decompose_bloq()
    edges = []
    for cxn in cbloq.connections:
        edges.append((_make_soquet_serializable(cxn.left), _make_soquet_serializable(cxn.right)))

    return {
        'edges': edges
    }


def get_decomposition_musical_score(bloq: Bloq) -> MusicalScoreData:
    """Please navigate to the `MusicalScoreData` dataclass and its affiliates for details."""
    return get_musical_score_data(bloq.decompose_bloq())


def dump_example_data():
    """Dump an example JSON file containing information about an example bloq."""

    # Use an arbitrary bloq as an example.
    modexp_small = ModExp(base=3, mod=15, exp_bitsize=3, x_bitsize=2048)

    # The top level schema
    data = {
        'bloq_classname': get_bloq_classname(modexp_small),
        'bloq_docstring': get_bloq_docstring(modexp_small),
        'bloq_attributes': get_bloq_attributes(modexp_small),
        'callees': get_callees(modexp_small),
        'decomposition_dag': get_decomposition_dag(modexp_small),
        'decomposition_musical_score': get_decomposition_musical_score(modexp_small),

    }

    # Dump to json. Note that we use the `MusicalScoreEncoder` because the musical score
    # data already has a robust(ish) JSON conversion/schema.
    with open('modexp_small.json', 'w') as f:
        json.dump(data, f, indent=2, cls=MusicalScoreEncoder)


if __name__ == '__main__':
    dump_example_data()
