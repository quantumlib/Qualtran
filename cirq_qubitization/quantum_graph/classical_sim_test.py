from typing import Dict

import cirq
import numpy as np
import pytest
from attrs import frozen
from numpy.typing import NDArray

from cirq_qubitization.bloq_algos.basic_gates import CNOT
from cirq_qubitization.jupyter_tools import execute_notebook
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.classical_sim import (
    _cbloq_call_classically,
    bits_to_ints,
    ints_to_bits,
)
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side


def test_bits_to_int():
    rs = np.random.RandomState(52)
    bitstrings = rs.choice([0, 1], size=(100, 23))

    nums = bits_to_ints(bitstrings)
    assert nums.shape == (100,)

    for num, bs in zip(nums, bitstrings):
        ref_num = cirq.big_endian_bits_to_int(bs.tolist())
        assert num == ref_num

    # check one input bitstring instead of array of input bitstrings.
    (num,) = bits_to_ints([1, 0])
    assert num == 2


def test_int_to_bits():
    rs = np.random.RandomState(52)
    nums = rs.randint(0, 2**23 - 1, size=(100,), dtype=np.uint64)
    bitstrings = ints_to_bits(nums, w=23)
    assert bitstrings.shape == (100, 23)

    for num, bs in zip(nums, bitstrings):
        ref_bs = cirq.big_endian_int_to_bits(int(num), bit_count=23)
        np.testing.assert_array_equal(ref_bs, bs)

    # check one input int
    (bitstring,) = ints_to_bits(2, w=8)
    assert bitstring.tolist() == [0, 0, 0, 0, 0, 0, 1, 0]

    # check bounds
    with pytest.raises(AssertionError):
        ints_to_bits([4, -2], w=8)


@frozen
class ApplyClassicalTest(Bloq):
    @property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters(
            [
                FancyRegister('x', 1, wireshape=(5,)),
                FancyRegister('z', 1, wireshape=(5,), side=Side.RIGHT),
            ]
        )

    def on_classical_vals(self, *, x: NDArray[np.uint8]) -> Dict[str, NDArray[np.uint8]]:
        const = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
        z = np.logical_xor(x, const).astype(np.uint8)
        return {'x': x, 'z': z}


def test_apply_classical():
    bloq = ApplyClassicalTest()
    x, z = bloq.call_classically(x=np.zeros(5, dtype=np.uint8))
    np.testing.assert_array_equal(x, np.zeros(5))
    assert z.dtype == np.uint8
    np.testing.assert_array_equal(z, [1, 0, 1, 0, 1])

    x2, z2 = bloq.on_classical_vals(x=np.ones(5, dtype=np.uint8))
    np.testing.assert_array_equal(x2, np.ones(5))
    np.testing.assert_array_equal(z2, [0, 1, 0, 1, 0])


def test_cnot_assign_dict():
    cbloq = CNOT().as_composite_bloq()
    binst_graph = cbloq._binst_graph
    vals = dict(ctrl=1, target=0)
    out_vals, soq_assign = _cbloq_call_classically(cbloq.registers, vals, binst_graph)
    assert out_vals == {'ctrl': 1, 'target': 1}
    # left-dangle, regs, right-dangle
    assert len(soq_assign) == 2 + 2 + 2
    for soq in cbloq.all_soquets:
        assert soq in soq_assign.keys()


def test_apply_classical_cbloq():
    bb = CompositeBloqBuilder()
    x = bb.add_register(FancyRegister('x', 1, wireshape=(5,)))
    x, y = bb.add(ApplyClassicalTest(), x=x)
    y, z = bb.add(ApplyClassicalTest(), x=y)
    cbloq = bb.finalize(x=x, y=y, z=z)

    xarr = np.zeros(5)
    x, y, z = cbloq.call_classically(x=xarr)
    np.testing.assert_array_equal(x, xarr)
    np.testing.assert_array_equal(y, [1, 0, 1, 0, 1])
    np.testing.assert_array_equal(z, xarr)


def test_notebook():
    execute_notebook('classical_sim')
