import numpy as np
import pytest

import cirq_qubitization.testing as cq_testing
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder
from cirq_qubitization.quantum_graph.fancy_registers import Side
from cirq_qubitization.quantum_graph.quantum_graph import Soquet
from cirq_qubitization.quantum_graph.util_bloqs import Allocate, Free, Join, Split


@pytest.mark.parametrize('n', [5, 123])
@pytest.mark.parametrize('bloq_cls', [Split, Join])
def test_register_sizes_add_up(bloq_cls, n):
    bloq = bloq_cls(n)
    for name, group_regs in bloq.registers.groups():

        if any(reg.side is Side.THRU for reg in group_regs):
            assert not any(reg.side != Side.THRU for reg in group_regs)
            continue

        lefts = group_regs.lefts()
        left_size = np.product([l.total_bits() for l in lefts])
        rights = group_regs.rights()
        right_size = np.product([r.total_bits() for r in rights])

        assert left_size > 0
        assert left_size == right_size


def test_util_bloqs():
    bb = CompositeBloqBuilder()
    (qs1,) = bb.add(Allocate(10))
    assert isinstance(qs1, Soquet)
    (qs2,) = bb.add(Split(10), split=qs1)
    assert qs2.shape == (10,)
    (qs3,) = bb.add(Join(10), join=qs2)
    assert isinstance(qs3, Soquet)
    no_return = bb.add(Free(10), free=qs3)
    assert no_return is tuple()


def test_notebook():
    cq_testing.execute_notebook('quantum_graph/util_bloqs')
