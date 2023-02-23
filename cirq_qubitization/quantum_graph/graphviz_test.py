import re
from functools import cached_property
from typing import Dict, List, Sequence, Tuple

import cirq
from attrs import frozen

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from cirq_qubitization.quantum_graph.graphviz import _assign_ids_to_bloqs_and_soqs
from cirq_qubitization.quantum_graph.quantum_graph import Soquet
from cirq_qubitization.quantum_graph.util_bloqs import Join, Partition, Split, Unpartition


@frozen
class Atom(Bloq):
    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters.build(q=1)


class TestParallelBloq(Bloq):
    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters.build(stuff=3)

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', stuff: 'SoquetT'
    ) -> Dict[str, 'Soquet']:

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
