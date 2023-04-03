from typing import Dict

import numpy as np
from attrs import frozen
from numpy.typing import NDArray

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.classical_sim import _cbloq_apply_classical, int_to_bits
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side


@frozen
class SetBitsTest(Bloq):
    x: int
    w: int

    @property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters([FancyRegister('bits', self.w, side=Side.RIGHT)])

    def apply_classical(self) -> Dict[str, NDArray[np.uint8]]:
        return {'bits': int_to_bits(self.x, self.w)}


@frozen
class CNOTExample(Bloq):
    @property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters.build(ctrl=1, target=1)

    def apply_classical(
        self, ctrl: NDArray[np.uint8], target: NDArray[np.uint8]
    ) -> Dict[str, NDArray[np.uint8]]:
        target_out = (ctrl + target) % 2
        return {'ctrl': ctrl, 'target': target_out}


@frozen
class ApplyClassicalTest(Bloq):
    @property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters([FancyRegister('x', 5), FancyRegister('z', 5, side=Side.RIGHT)])

    def apply_classical(self, *, x: NDArray[np.uint8]) -> Dict[str, NDArray[np.uint8]]:
        const = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
        z = np.logical_xor(x, const).astype(np.uint8)
        return {'x': x, 'z': z}


def test_apply_classical():
    bloq = ApplyClassicalTest()
    ret = bloq.apply_classical(x=np.zeros(5, dtype=np.uint8))
    np.testing.assert_array_equal(ret['x'], np.zeros(5))
    assert ret['z'].dtype == np.uint8
    np.testing.assert_array_equal(ret['z'], [1, 0, 1, 0, 1])

    ret2 = bloq.apply_classical(x=np.ones(5, dtype=np.uint8))
    np.testing.assert_array_equal(ret2['z'], [0, 1, 0, 1, 0])


def test_cnot_assign_dict():
    cbloq = CNOTExample().as_composite_bloq()

    in_data = {'ctrl': np.array([1], dtype=np.uint8), 'target': np.array([0], dtype=np.uint8)}

    out_vals, soq_assign = _cbloq_apply_classical(cbloq.registers, in_data, cbloq._binst_graph)
    print(soq_assign)


def test_apply_classical_cbloq():
    bb = CompositeBloqBuilder()
    x = bb.add_register('x', 5)
    x, y = bb.add(ApplyClassicalTest(), x=x)
    y, z = bb.add(ApplyClassicalTest(), x=y)
    cbloq = bb.finalize(x=x, y=y, z=z)

    xarr = np.zeros(5, dtype=np.uint8)
    ret = cbloq.apply_classical(x=xarr)
    np.testing.assert_array_equal(ret['x'], xarr)
    np.testing.assert_array_equal(
        ret['y'], [not xarr[0], xarr[1], not xarr[2], xarr[3], not xarr[4]]
    )
    np.testing.assert_array_equal(ret['z'], xarr)
