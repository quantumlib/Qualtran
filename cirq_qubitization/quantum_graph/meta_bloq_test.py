from cirq_qubitization.quantum_graph.meta_bloq import ControlledBloq, TestSerialBloq


def test_controlled():
    bloq = ControlledBloq(subbloq=TestSerialBloq()).decompose_bloq()
    print()
    print(bloq.debug_text())
    print()
