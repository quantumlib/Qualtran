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

import attrs

from qualtran import Bloq, Soquet
from qualtran.bloqs.factoring.mod_exp import ModExp
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran_dev_tools.parse_docstrings import get_markdown_docstring_lines

modexp_small = ModExp(base=3, mod=15, exp_bitsize=3, x_bitsize=2048)


def get_callees(bloq: Bloq):
    call_graph, _ = modexp_small.call_graph(max_depth=1, generalizer=ignore_split_join)
    callees = []
    for child_bloq, edge_data in call_graph.succ[bloq].items():
        callees.append({
            'bloq_ref': repr(child_bloq),  # TODO: how do we link to other bloqs
            'edge_data': edge_data,
        })

    return callees


def serializable_soquet(soq: Soquet):
    return {
        'binst': repr(soq.binst),  # TODO: structured data
        'reg_name': soq.reg.name,
        'reg_side': str(soq.reg.side),
        'idx': soq.idx
    }


def get_decomposition_dag(bloq: Bloq):
    cbloq = bloq.decompose_bloq()
    edges = []
    for cxn in cbloq.connections:
        edges.append((serializable_soquet(cxn.left), serializable_soquet(cxn.right)))

    return {
        'edges': edges
    }


data = {
    'bloq_classname': modexp_small.__class__.__name__,
    'bloq_docstring': '\n'.join(get_markdown_docstring_lines(modexp_small.__class__)),
    'bloq_attributes': attrs.asdict(modexp_small),
    'callees': get_callees(modexp_small),
    'decomposition_dag': get_decomposition_dag(modexp_small),

}

with open('modexp_small.json', 'w') as f:
    json.dump(data, f, indent=2)
