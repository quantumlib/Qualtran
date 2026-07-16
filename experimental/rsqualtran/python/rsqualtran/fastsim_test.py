#  Copyright 2026 Google LLC
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
import pytest

import numpy as np
import qualtran as qlt
from qualtran.bloqs.for_testing.nd_array_bloq import TestND3Grid
from qualtran.l1 import load_bloq as lb, L1ModuleBuilder

from rsqualtran import fastsim
from rsqualtran import nodes as rust_nodes_module


@pytest.fixture(scope="module")
def negate_simulator():
    bloq = lb('qualtran.bloqs.arithmetic.Negate(QInt(8))')
    return fastsim.QLTFastsim.from_bloq(bloq)


@pytest.fixture(scope="module")
def cswap_simulator():
    bloq = lb('qualtran.bloqs.basic_gates.CSwap(5)')
    return fastsim.QLTFastsim.from_bloq(bloq)


@pytest.fixture(scope="module")
def nd3grid_simulator():
    return fastsim.QLTFastsim.from_bloq(TestND3Grid())


def test_call_classically_negate(negate_simulator):
    for x in [-128, -5, -1, 0, 1, 5, 127]:
        (result_x,) = negate_simulator.call_classically(x=x)
        assert result_x == (-x if x != -128 else -128)


def test_call_classically_cswap(cswap_simulator):
    for ctrl, x, y in itertools.product([0, 1], [0, 5, 31], [0, 5, 31]):
        result = cswap_simulator.call_classically(ctrl=ctrl, x=x, y=y)

        expected_x = y if ctrl == 1 else x
        expected_y = x if ctrl == 1 else y
        assert result == (ctrl, expected_x, expected_y)


def test_call_classically_returns_tuple(negate_simulator):
    result = negate_simulator.call_classically(x=5)
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert isinstance(result[0], int)


def test_call_classically_cswap_returns_three_values(cswap_simulator):
    result = cswap_simulator.call_classically(ctrl=0, x=1, y=2)
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_simulate_negate(negate_simulator):
    for x in [-128, -5, -1, 0, 1, 5, 127]:
        outputs, phase_exponent = negate_simulator.simulate(x=x)
        assert outputs == {'x': -x if x != -128 else -128}
        assert phase_exponent == 0.0


def test_simulate_cswap(cswap_simulator):
    for ctrl, x, y in itertools.product([0, 1], [0, 5, 31], [0, 5, 31]):
        outputs, phase_exponent = cswap_simulator.simulate(ctrl=ctrl, x=x, y=y)

        expected_x = y if ctrl == 1 else x
        expected_y = x if ctrl == 1 else y
        assert outputs == {'ctrl': ctrl, 'x': expected_x, 'y': expected_y}
        assert phase_exponent == 0.0


def test_simulate_returns_dict_and_phase_exponent(negate_simulator):
    result = negate_simulator.simulate(x=5)
    assert isinstance(result, tuple)
    assert len(result) == 2
    outputs, phase_exponent = result
    assert isinstance(outputs, dict)
    assert isinstance(phase_exponent, float)


def test_invalid_input_register(negate_simulator):
    with pytest.raises(ValueError, match="Unexpected input register 'y' for subroutine 'Negate'"):
        negate_simulator.call_classically(x=5, y=10)


def test_missing_input_register(negate_simulator):
    with pytest.raises(ValueError, match="Missing required input register: 'x'"):
        negate_simulator.call_classically()


def test_invalid_input_type(negate_simulator):
    with pytest.raises(TypeError, match="Unsupported input type for register 'x'"):
        negate_simulator.call_classically(x={"invalid": "type"})


def test_simulate_invalid_input_register(negate_simulator):
    with pytest.raises(ValueError, match="Unexpected input register 'y'"):
        negate_simulator.simulate(x=5, y=10)


def test_simulate_missing_input_register(negate_simulator):
    with pytest.raises(ValueError, match="Missing required input register: 'x'"):
        negate_simulator.simulate()


# ── ND-array tests ────────────────────────────────────────────────────────


def test_ndarray_simulate_zero_cube(nd3grid_simulator):
    """Simulate with an all-zero (2,2,2) cube.

    The TestND3Grid circuit applies X to cube[0,0,0], so starting from
    all zeros the only bit that flips is [0,0,0] → 1.
    """
    cube = np.zeros((2, 2, 2), dtype=np.uint8)
    outputs, phase = nd3grid_simulator.simulate(cube=cube, aux=0)

    expected_cube = np.zeros((2, 2, 2), dtype=np.uint64)
    expected_cube[0, 0, 0] = 1
    np.testing.assert_array_equal(outputs["cube"], expected_cube)
    assert outputs["aux"] == 0
    assert phase == 0.0


def test_ndarray_simulate_cube_with_first_bit_set(nd3grid_simulator):
    """Simulate with cube[0,0,0]=1 and everything else zero.

    X flips cube[0,0,0] from 1 → 0. All CNOT/Toffoli controls are 0,
    so nothing else changes. Output is all zeros.
    """
    cube = np.zeros((2, 2, 2), dtype=np.uint8)
    cube[0, 0, 0] = 1
    outputs, phase = nd3grid_simulator.simulate(cube=cube, aux=0)

    np.testing.assert_array_equal(outputs["cube"], np.zeros((2, 2, 2), dtype=np.uint64))
    assert outputs["aux"] == 0
    assert phase == 0.0


def test_ndarray_simulate_all_ones_cube(nd3grid_simulator):
    """Simulate with an all-ones (2,2,2) cube.

    Cross-check: the CLI outputs flat value 86 (0b01010110) for this input.
    """
    cube = np.ones((2, 2, 2), dtype=np.uint8)
    outputs, phase = nd3grid_simulator.simulate(cube=cube, aux=0)

    # Verify against the known CLI flat-integer result.
    flat = 0
    for v in outputs["cube"].flat:
        flat = (flat << 1) | int(v)
    assert flat == 86
    assert phase == 0.0


def test_ndarray_call_classically_returns_ndarray(nd3grid_simulator):
    """call_classically returns np.ndarray for shaped registers."""
    cube_in = np.zeros((2, 2, 2), dtype=np.uint8)
    result = nd3grid_simulator.call_classically(cube=cube_in, aux=0)

    assert isinstance(result, tuple)
    assert len(result) == 2  # cube and aux
    assert isinstance(result[0], np.ndarray)
    assert result[0].shape == (2, 2, 2)
    assert isinstance(result[1], int)  # aux is scalar


def test_ndarray_simulate_returns_ndarray_in_dict(nd3grid_simulator):
    """simulate returns np.ndarray for shaped register in output dict."""
    cube_in = np.zeros((2, 2, 2), dtype=np.uint8)
    outputs, _ = nd3grid_simulator.simulate(cube=cube_in, aux=0)

    assert isinstance(outputs["cube"], np.ndarray)
    assert outputs["cube"].shape == (2, 2, 2)
    assert isinstance(outputs["aux"], int)


def test_ndarray_int_input_still_works(nd3grid_simulator):
    """Passing a plain int for a shaped register still works (flat value)."""
    result_int = nd3grid_simulator.call_classically(cube=0, aux=0)
    result_arr = nd3grid_simulator.call_classically(
        cube=np.zeros((2, 2, 2), dtype=np.uint8), aux=0
    )

    # Both should produce the same ndarray output.
    np.testing.assert_array_equal(result_int[0], result_arr[0])
    assert result_int[1] == result_arr[1]


def test_ndarray_wrong_shape_raises(nd3grid_simulator):
    """Passing an ndarray with wrong shape raises ValueError."""
    wrong_shape = np.zeros((4, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not match expected shape"):
        nd3grid_simulator.simulate(cube=wrong_shape, aux=0)


def test_ndarray_on_scalar_register_raises(negate_simulator):
    """Passing an ndarray for a scalar register raises TypeError."""
    with pytest.raises(TypeError, match="Got np.ndarray for scalar register"):
        negate_simulator.simulate(x=np.array([1, 2, 3]))


def test_ndarray_round_trip(nd3grid_simulator):
    """Output ndarray from one run can be fed back as input to another."""
    cube_in = np.zeros((2, 2, 2), dtype=np.uint8)
    cube_in[0, 0, 1] = 1
    cube_in[1, 0, 1] = 1

    outputs1, _ = nd3grid_simulator.simulate(cube=cube_in, aux=0)
    cube_mid = outputs1["cube"].astype(np.uint8)

    # Feed the output back in.
    outputs2, _ = nd3grid_simulator.simulate(cube=cube_mid, aux=outputs1["aux"])

    # The circuit is its own inverse (all gates are self-inverse), so
    # two applications should return the original input.
    np.testing.assert_array_equal(outputs2["cube"], cube_in.astype(np.uint64))
    assert outputs2["aux"] == 0


def test_multi_register_partition_cmodadd():
    """Regression: multi-register `_PartitionBase` casts must compile & run.

    `CModAdd(cv=1)` decomposes through `ControlledViaAnd` -> `CtrlSpecAnd`,
    which emits multi-register `Partition`/`CtrlSpecMerge`/`CtrlSpecPartition`
    bloqs. Previously these were rejected as invalid single-register qcasts.
    Verify the fast simulator now agrees with qualtran's reference classical
    simulator across all inputs.
    """
    import itertools

    from qualtran import QUInt
    from qualtran.bloqs.mod_arithmetic import CModAdd

    bloq = CModAdd(QUInt(4), mod=13, cv=1)
    sim = fastsim.QLTFastsim.from_bloq(bloq)

    for ctrl, x, y in itertools.product([0, 1], [0, 3, 7, 12], [0, 3, 7, 12]):
        ref = tuple(bloq.call_classically(ctrl=ctrl, x=x, y=y))
        got = tuple(sim.call_classically(ctrl=ctrl, x=x, y=y))
        assert got == ref, f"input=(ctrl={ctrl}, x={x}, y={y}): got {got}, expected {ref}"

