import numpy as np

from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister
from cirq_qubitization.quantum_graph.quantum_graph import BloqInstance, Soquet
from cirq_qubitization.quantum_graph.util_bloqs import Split


def test_split():
    bloq = Split(n=5)
    binst = BloqInstance(bloq)

    # What follows is a highly-simplified version of CompositeBloqBuilder.add for Split
    left_reg: FancyRegister
    (left_reg,) = bloq.registers.lefts()
    (idx,) = left_reg.wire_idxs()
    assert idx == tuple()

    soq = Soquet(binst, left_reg, idx=idx)
    assert soq.pretty() == 'split'

    right_reg: FancyRegister
    (right_reg,) = bloq.registers.rights()
    out = np.empty(right_reg.wireshape, dtype=object)
    for ri in right_reg.wire_idxs():
        out_soq = Soquet(binst, right_reg, idx=ri)
        out[ri] = out_soq

    assert out.shape == (5,)
    for i, o in enumerate(out):
        assert o.pretty() == f'split[{i}]'
