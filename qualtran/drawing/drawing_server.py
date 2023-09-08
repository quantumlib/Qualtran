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

import asyncio
import re
from functools import lru_cache
from typing import Dict, Any, Tuple

import pydot
import tornado

from qualtran import CompositeBloq, Bloq, Signature, Register, Soquet, DanglingT
from qualtran.bloqs.factoring import ModExp
from qualtran.drawing import PrettyGraphDrawer
from qualtran.drawing.graphviz_test import TestParallelBloq


def minimal_soq_to_json(soq: Tuple[Register, Tuple[int, ...]]):
    reg, idx = soq
    label = reg.name
    if len(idx) > 0:
        label = f'{label}[{", ".join(str(i) for i in idx)}]'
    return {
        'label': label,
        'side': str(reg.side),
    }


@lru_cache()
def get_minimal_soqs(signature: Signature):
    minimal_soqs = []
    for reg in signature:
        if reg.shape:
            for idx in reg.all_idxs():
                minimal_soqs.append((reg, idx))
        else:
            minimal_soqs.append((reg, ()))

    return minimal_soqs


@lru_cache()
def get_minimal_soqs_back_map(signature: Signature):
    return {msoq: i for i, msoq in enumerate(get_minimal_soqs(signature))}


def signature_to_json(signature: Signature):
    return [minimal_soq_to_json(soq) for soq in get_minimal_soqs(signature)]


def bloq_to_json(bloq: Bloq):
    return {
        'pretty_name': bloq.pretty_name(),
    }


def get_pos_data(cbloq: CompositeBloq):
    pgd = PrettyGraphDrawer(cbloq)
    graph = pgd.get_graph()
    graph2, = pydot.graph_from_dot_data(graph.create_dot().decode())
    nodes = graph2.get_nodes()
    pos_dict = {node.get_name(): node.get_pos() for node in nodes}

    x_pos = {}
    y_pos = {}
    for binst in cbloq.bloq_instances:
        graphviz_id = pgd.ids[binst]
        strpos = pos_dict[graphviz_id]
        if strpos is None:
            raise ValueError(f"Graphviz didn't provide a pos for {binst}")

        x, y = strpos.rstrip('"').lstrip('"').split(',')
        x_pos[binst.i] = float(x)
        y_pos[binst.i] = float(y)
    return x_pos, y_pos


def soq_to_soqref(soq: Soquet):
    if isinstance(soq.binst, DanglingT):
        return
    msoq = (soq.reg, soq.idx)
    soq_i = get_minimal_soqs_back_map(soq.binst.bloq.signature)[msoq]

    return {
        'binst_i': soq.binst.i,
        'soq_i': soq_i
    }


def cbloq_to_json(cbloq: CompositeBloq) -> Dict[str, Any]:
    xpos, ypos = get_pos_data(cbloq)
    binsts = sorted(cbloq.bloq_instances, key=lambda b: b.i)
    binsts = [
        {
            'i': binst.i,
            'bloq': bloq_to_json(binst.bloq),
            'soqs': signature_to_json(binst.bloq.signature),
            'x': xpos[binst.i],
            'y': ypos[binst.i],
        } for binst in binsts]
    cxns = [(soq_to_soqref(cxn.left), soq_to_soqref(cxn.right)) for cxn in cbloq.connections]
    cxns = [cxn for cxn in cxns if cxn[0] is not None and cxn[1] is not None]

    return {'binsts': binsts, 'cxns': cxns}


ROOT_BLOQS = {
    'ModExp': ModExp(base=8, mod=27, exp_bitsize=3, x_bitsize=2048),
    'TestParallelBloq': TestParallelBloq()
}


def get_root_bloq(name: str) -> Bloq:
    return ROOT_BLOQS[name]


@lru_cache
def get_cbloq(key: str) -> CompositeBloq:
    key_parts = key.split('/')
    if len(key_parts) == 0:
        raise ValueError(f"Bad key: {key}")
    if len(key_parts) == 1:
        return get_root_bloq(key_parts[0]).as_composite_bloq()

    # Recurse
    *parts, last_part = key_parts
    cbloq = get_cbloq('/'.join(parts))

    # Do the final transformation
    if ma := re.match(r'i(\d+)', last_part):
        # decompose by `i`
        i = int(ma.group(1))
        return cbloq.flatten_once(lambda binst: binst.i == i)

    raise ValueError(f"Bad instruction {last_part}")


@lru_cache
def get_cbloq_data(key: str) -> Dict[Any, str]:
    cbloq = get_cbloq(key)
    return cbloq_to_json(cbloq)


class BloqHandler(tornado.web.RequestHandler):
    def get(self, key: str):
        return self.write(get_cbloq_data(key))


def make_app():
    return tornado.web.Application([
        (r"/bloq/(.*)", BloqHandler),
    ])


async def main():
    app = make_app()
    app.listen(8081)
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
