from cirq_qubitization.bloq_algos.shors.shors import ModExp


def tet_mod_exp():
    ModExp(g=8, exp_bitsize=3, x_bitsize=10, mod_N=50)
