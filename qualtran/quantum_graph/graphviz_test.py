import re
from functools import cached_property
from typing import Dict

import IPython.display
import pytest
from attrs import frozen

from qualtran import Bloq, BloqBuilder, FancyRegisters, Soquet
from qualtran.jupyter_tools import execute_notebook
from qualtran.quantum_graph.graphviz import (
    _assign_ids_to_bloqs_and_soqs,
    GraphDrawer,
    PrettyGraphDrawer,
)


@frozen
class Atom(Bloq):
    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters.build(q=1)


class TestParallelBloq(Bloq):
    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters.build(stuff=3)

    def build_composite_bloq(self, bb: 'BloqBuilder', stuff: 'SoquetT') -> Dict[str, 'Soquet']:

        qs = bb.split(stuff)
        for i in range(3):
            (qs[i],) = bb.add(Atom(), q=qs[i])
        return {'stuff': bb.join(qs)}


def test_assign_ids():
    cbloq = TestParallelBloq().decompose_bloq()
    id_map = _assign_ids_to_bloqs_and_soqs(cbloq.bloq_instances, cbloq.all_soquets)

    ids = sorted(id_map.values())

    # check correct number
    n_binst = 3 + 1 + 1  # Atom, Split, Join
    n_group = n_binst  # Each has one register group in this example
    n_soq = 1 + 1 + 3 + 3 + 3 + 1 + 1  # dangle, split(l), split(r), atoms, join(l), join(r), dangle
    assert len(ids) == n_binst + n_group + n_soq

    # ids are prefix_G123
    prefixes = set()
    for v in ids:
        ma = re.match(r'(\w+)_G(\d+)', v)
        if ma is None:
            prefixes.add(v)
            continue
        prefixes.add(ma.group(1))
    assert sorted(prefixes) == ['Atom', 'Join', 'Split', 'join', 'q', 'split', 'stuff']


@pytest.mark.parametrize('draw_cls', [GraphDrawer, PrettyGraphDrawer])
def test_graphviz(draw_cls):
    bloq = TestParallelBloq().decompose_bloq()
    drawer = draw_cls(bloq)
    graph = drawer.get_graph()
    assert len(graph.get_nodes()) == 1 + 3 + 1  # split, atoms, join
    assert len(graph.get_subgraphs()) == 2  # left, right dangling labels go in a subgraph
    assert len(graph.get_edges()) == 1 + 3 + 3 + 1
    assert len(graph.create_svg()) > 0

    svg_bytes = drawer.get_svg_bytes()
    assert svg_bytes.startswith(b'<?xml')
    assert svg_bytes.decode().startswith('<?xml')

    svg_widget = drawer.get_svg()
    assert isinstance(svg_widget, IPython.display.DisplayObject)


def test_notebook():
    execute_notebook('graphviz')
