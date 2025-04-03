#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import itertools
from typing import Dict

import networkx as nx
import numpy as np
import pytest
from attrs import frozen
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    BloqBuilder,
    BQUInt,
    LeftDangle,
    QAny,
    QBit,
    QDType,
    QFxp,
    QGF,
    QInt,
    QIntOnesComp,
    QUInt,
    Register,
    Side,
    Signature,
)
from qualtran.bloqs.basic_gates import CNOT
from qualtran.simulation.classical_sim import (
    add_ints,
    call_cbloq_classically,
    ClassicalSimState,
    do_phased_classical_simulation,
)
from qualtran.testing import execute_notebook


def test_dtype_validation():
    # set up mocks for `_update_assign_from_vals`
    binst = 'MyBinst'  # binst is only used for error messages, so we can mock with a string
    sim = ClassicalSimState(Signature([]), nx.DiGraph(), {})

    # set up different register dtypes
    regs = [
        Register('one_bit_int', QBit()),
        Register('int', QAny(5)),
        Register('bit_arr', QBit(), shape=(5,)),
        Register('int_arr', QAny(32), shape=(5,)),
    ]

    # base case: vals are as-expected.
    vals = {
        'one_bit_int': 1,
        'int': 5,
        'bit_arr': np.array([1, 0, 1, 0, 1], dtype=np.uint8),
        'int_arr': np.arange(5),
    }
    sim._update_assign_from_vals(regs, binst, vals)  # type: ignore[arg-type]

    # bad integer
    vals2 = {**vals, 'one_bit_int': 2}
    with pytest.raises(ValueError, match=r'Bad QBit().*one_bit_int'):
        sim._update_assign_from_vals(regs, binst, vals2)  # type: ignore[arg-type]

    # int is a numpy int
    vals3 = {**vals, 'int': np.arange(5, dtype=np.uint8)[4]}
    sim._update_assign_from_vals(regs, binst, vals3)  # type: ignore[arg-type]

    # wrong shape
    vals4 = {**vals, 'int_arr': np.arange(6)}
    with pytest.raises(ValueError, match=r'Incorrect shape.*Want \(5,\)\.'):
        sim._update_assign_from_vals(regs, binst, vals4)  # type: ignore[arg-type]


@frozen
class ApplyClassicalTest(Bloq):
    @property
    def signature(self) -> 'Signature':
        return Signature(
            [Register('x', QBit(), shape=(5,)), Register('z', QBit(), shape=(5,), side=Side.RIGHT)]
        )

    def on_classical_vals(self, *, x: NDArray[np.uint8]) -> Dict[str, NDArray[np.uint8]]:
        const = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
        z = np.logical_xor(x, const).astype(np.uint8)
        return {'x': x, 'z': z}


class ApplyPhasedClassicalTest(ApplyClassicalTest):
    def basis_state_phase(self, x: NDArray[np.uint8]) -> complex:
        return np.prod(np.exp(x * 1.0j / np.pi))


def test_apply_classical():
    bloq = ApplyClassicalTest()
    x, z = bloq.call_classically(x=np.zeros(5, dtype=np.uint8))
    np.testing.assert_array_equal(x, np.zeros(5))
    assert not isinstance(x, int)
    assert not isinstance(z, int)
    assert x.dtype == np.uint8
    assert z.dtype == np.uint8
    np.testing.assert_array_equal(z, [1, 0, 1, 0, 1])

    x2, z2 = bloq.call_classically(x=np.ones(5, dtype=np.uint8))
    assert not isinstance(x2, int)
    assert not isinstance(z2, int)
    assert x2.dtype == np.uint8
    assert z2.dtype == np.uint8
    np.testing.assert_array_equal(x2, np.ones(5))
    np.testing.assert_array_equal(z2, [0, 1, 0, 1, 0])


def test_apply_phased_classical():
    bloq = ApplyPhasedClassicalTest()
    final_vals, phase = do_phased_classical_simulation(bloq, dict(x=np.ones(5, dtype=np.uint8)))
    np.testing.assert_array_equal(final_vals['x'], np.ones(5))
    np.testing.assert_array_equal(final_vals['z'], [0, 1, 0, 1, 0])
    assert np.abs(phase - np.exp(5.0j / np.pi)) < 1e-8


def test_phased_classical_on_normal_classical():
    final_vals, phase = do_phased_classical_simulation(
        ApplyClassicalTest(), dict(x=np.ones(5, dtype=np.uint8))
    )
    np.testing.assert_array_equal(final_vals['x'], np.ones(5))
    np.testing.assert_array_equal(final_vals['z'], [0, 1, 0, 1, 0])
    assert phase == 1.0


def test_normal_classical_on_phased():
    bloq = ApplyPhasedClassicalTest()
    with pytest.raises(ValueError, match=r'.*`do_phased_classical_simulation`.*'):
        x, z = bloq.call_classically(x=np.zeros(5, dtype=np.uint8))


def test_cnot_assign_dict():
    cbloq = CNOT().as_composite_bloq()
    binst_graph = cbloq._binst_graph  # pylint: disable=protected-access
    vals = dict(ctrl=1, target=0)
    out_vals, soq_assign = call_cbloq_classically(cbloq.signature, vals, binst_graph)
    assert out_vals == {'ctrl': 1, 'target': 1}
    # left-dangle, regs, right-dangle
    assert len(soq_assign) == 2 + 2 + 2
    for soq in cbloq.all_soquets:
        assert soq in soq_assign.keys()


def test_apply_classical_cbloq():
    bb = BloqBuilder()
    x = bb.add_register(Register('x', QBit(), shape=(5,)))
    assert x is not None
    x, y = bb.add(ApplyClassicalTest(), x=x)
    y, z = bb.add(ApplyClassicalTest(), x=y)
    cbloq = bb.finalize(x=x, y=y, z=z)

    xarr = np.zeros(5, dtype=np.intc)
    x, y, z = cbloq.call_classically(x=xarr)
    np.testing.assert_array_equal(x, xarr)
    np.testing.assert_array_equal(y, [1, 0, 1, 0, 1])
    np.testing.assert_array_equal(z, xarr)


def test_step():
    bb = BloqBuilder()
    x = bb.add_register(Register('x', QBit(), shape=(5,)))
    assert x is not None
    x, y = bb.add(ApplyClassicalTest(), x=x)
    y, z = bb.add(ApplyClassicalTest(), x=y)
    cbloq = bb.finalize(x=x, y=y, z=z)

    xarr = np.zeros(5, dtype=np.intc)
    sim = ClassicalSimState.from_cbloq(cbloq, dict(x=xarr))
    sim.step()
    assert sim.last_binst is not None and sim.last_binst is LeftDangle
    sim.step()
    assert sim.last_binst is not None and sim.last_binst.bloq_is(ApplyClassicalTest)
    assert sim.last_binst is not None and sim.last_binst.i == 0

    final_vals = sim.step().finalize()
    np.testing.assert_array_equal(final_vals['x'], xarr)
    np.testing.assert_array_equal(final_vals['y'], [1, 0, 1, 0, 1])
    np.testing.assert_array_equal(final_vals['z'], xarr)


@pytest.mark.parametrize('n_bits', range(1, 5))
def test_add_ints_unsigned(n_bits):
    for x, y in itertools.product(range(1 << n_bits), repeat=2):
        assert add_ints(x, y, num_bits=n_bits, is_signed=False) == (x + y) % (1 << n_bits)


@pytest.mark.parametrize('n_bits', range(2, 5))
def test_add_ints_signed(n_bits: int):
    half_n = 1 << (n_bits - 1)
    # Addition of signed ints `x` and `y` is a cyclic rotation of the interval [-2^(n-1), 2^(n-1)) by `y`.
    interval = [*range(-(2 ** (n_bits - 1)), 2 ** (n_bits - 1))]
    for x, y in itertools.product(range(-(2 ** (n_bits - 1)), 2 ** (n_bits - 1)), repeat=2):
        i = x + half_n  # position of `x` in the interval
        z = interval[(i + y) % len(interval)]  # rotate by `y`
        assert add_ints(x, y, num_bits=n_bits, is_signed=True) == z


@pytest.mark.notebook
def test_notebook():
    execute_notebook('classical_sim')


@frozen
class TestMultiDimensionalReg(Bloq):
    dtype: QDType
    n: int
    dtypes_to_assert: tuple[type, ...] = (int, np.integer)

    @property
    def signature(self):
        return Signature(
            [
                Register('x', self.dtype, shape=(self.n,), side=Side.LEFT),
                Register('y', self.dtype, shape=(self.n,), side=Side.RIGHT),
            ]
        )

    def on_classical_vals(self, x):
        assert all(isinstance(y, self.dtypes_to_assert) for y in x.reshape(-1))
        return {'y': x}


@pytest.mark.parametrize(
    'dtype', [QBit(), QInt(5), QUInt(5), QIntOnesComp(5), BQUInt(5, 20), QFxp(5, 3, signed=True)]
)
def test_multidimensional_classical_sim_for_dtypes(dtype: QDType):
    x = [*dtype.get_classical_domain()]
    bloq = TestMultiDimensionalReg(dtype, len(x))
    np.testing.assert_equal(bloq.call_classically(x=np.array(x))[0], x)


def test_multidimensional_classical_sim_for_large_int():
    dtype = QInt(100)
    x = [2**88 - 1, 2**12 - 1, 2**54 - 1, 1 - 2**72, 1 - 2**62]
    bloq = TestMultiDimensionalReg(dtype, len(x))
    np.testing.assert_equal(bloq.call_classically(x=np.array(x))[0], x)


def test_multidimensional_classical_sim_for_gqf():
    dtype = QGF(2, 2)
    x = dtype.gf_type.elements
    bloq = TestMultiDimensionalReg(dtype, len(x), (dtype.gf_type,))
    y = bloq.call_classically(x=x)[0]
    assert isinstance(y, dtype.gf_type)
    np.testing.assert_equal(y, x)
