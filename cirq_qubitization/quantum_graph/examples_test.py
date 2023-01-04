from cirq_qubitization.quantum_graph.examples import ModMultiply, SplitJoin


def test_code_runs():
    sj = SplitJoin(3)
    print(sj)
    cbloq = sj.decompose_bloq()
    print(cbloq)

    mm = ModMultiply(exponent_bitsize=3, x_bitsize=3, mul_constant=123, mod_N=5)
    print(mm)
    cbloq = mm.decompose_bloq()
    print(cbloq)
