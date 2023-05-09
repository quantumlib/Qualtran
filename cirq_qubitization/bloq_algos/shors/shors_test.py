import numpy as np
import pytest

from cirq_qubitization.bloq_algos.shors.shors import CtrlModMul, CtrlScaleModAdd, ModExp


def test_mod_exp_consistent_classical():
    rs = np.random.RandomState(52)

    bloq = ModExp(base=8, exp_bitsize=3, x_bitsize=10, mod=50)
    ret1 = bloq.call_classically(exponent=3)
    ret2 = bloq.decompose_bloq().call_classically(exponent=3)
    assert ret1 == ret2


@pytest.mark.parametrize('ctrl', [0, 1])
def test_cmm_consistent_classical(ctrl):
    rs = np.random.RandomState(52)

    bloq = CtrlModMul(k=10, mod=13 * 17, bitsize=32)
    regs = dict(ctrl=ctrl, x=11)
    ret1 = bloq.call_classically(**regs)
    ret2 = bloq.decompose_bloq().call_classically(**regs)
    assert ret1 == ret2


def test_ctrl_scale_mod_add_classical():
    bloq = CtrlScaleModAdd(k=10, bitsize=32, mod=13 * 17)

    ctrl, src, trg = bloq.call_classically(ctrl=0, src=123, trg=99)
    assert ctrl == 0
    assert src == 123
    assert trg == 99

    ctrl, src, trg = bloq.call_classically(ctrl=1, src=123, trg=99)
    assert ctrl == 1
    assert src == 123
    assert trg == (99 + 123 * 10) % (13 * 17)
