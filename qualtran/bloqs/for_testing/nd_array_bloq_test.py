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

# pylint: disable=unbalanced-tuple-unpacking
"""Tests for multi-dimensional quantum variable array test bloqs."""

import itertools

import numpy as np
import pytest

from qualtran.bloqs.for_testing.nd_array_bloq import TestND3Grid, TestNDGrid

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reference_nd_grid(grid: np.ndarray, ctrl: np.ndarray, flag: int):
    """Pure-Python reference implementation matching TestNDGrid.on_classical_vals."""
    g = grid.copy()
    c = ctrl.copy()
    f = int(flag)
    g[0, 0] ^= 1
    g[1, 0] ^= g[0, 1]
    g[1, 1] ^= c[0]
    g[2, 1] ^= g[2, 0] & c[1]
    f ^= g[2, 1]
    g[0, 1] ^= g[1, 0]
    return g, c, f


def _reference_nd3_grid(cube: np.ndarray, aux: int):
    """Pure-Python reference implementation matching TestND3Grid.on_classical_vals."""
    c = cube.copy()
    a = int(aux)
    c[0, 0, 0] ^= 1
    c[0, 1, 0] ^= c[0, 0, 1]
    c[1, 0, 0] ^= c[0, 1, 1]
    c[1, 1, 0] ^= c[1, 0, 0] & c[1, 0, 1]
    c[1, 1, 1] ^= c[1, 1, 0]
    a ^= c[1, 1, 1]
    return c, a


# ===========================================================================
# TestNDGrid tests (2-D grid register)
# ===========================================================================


def test_nd_grid_all_zeros():
    """All-zero input: only the X on grid[0,0] fires."""
    bloq = TestNDGrid()
    grid = np.zeros((3, 2), dtype=np.uint8)
    ctrl = np.zeros(2, dtype=np.uint8)
    out_grid, out_ctrl, out_flag = bloq.call_classically(grid=grid, ctrl=ctrl, flag=0)
    expected_grid = np.zeros((3, 2), dtype=np.uint8)
    expected_grid[0, 0] = 1
    np.testing.assert_array_equal(out_grid, expected_grid)
    np.testing.assert_array_equal(out_ctrl, ctrl)
    assert out_flag == 0


def test_nd_grid_ctrl0_set():
    """Setting ctrl[0]=1 should XOR into grid[1,1]."""
    bloq = TestNDGrid()
    grid = np.zeros((3, 2), dtype=np.uint8)
    ctrl = np.array([1, 0], dtype=np.uint8)
    out_grid, out_ctrl, out_flag = bloq.call_classically(grid=grid, ctrl=ctrl, flag=0)
    assert isinstance(out_grid, np.ndarray)
    assert out_grid[1, 1] == 1  # ctrl[0] XOR'd into grid[1,1]
    assert out_grid[0, 0] == 1  # X gate
    assert out_flag == 0


def test_nd_grid_toffoli_fires():
    """When grid[2,0]=1 and ctrl[1]=1 the Toffoli flips grid[2,1]."""
    bloq = TestNDGrid()
    grid = np.zeros((3, 2), dtype=np.uint8)
    grid[2, 0] = 1
    ctrl = np.array([0, 1], dtype=np.uint8)
    out_grid, _, out_flag = bloq.call_classically(grid=grid, ctrl=ctrl, flag=0)
    assert isinstance(out_grid, np.ndarray)
    assert out_grid[2, 1] == 1  # Toffoli fired
    assert out_flag == 1  # CNOT from grid[2,1] into flag


def test_nd_grid_toffoli_no_fire_one_ctrl():
    """Toffoli should NOT fire if only one control is set."""
    bloq = TestNDGrid()
    grid = np.zeros((3, 2), dtype=np.uint8)
    grid[2, 0] = 1
    ctrl = np.array([0, 0], dtype=np.uint8)
    out_grid, _, out_flag = bloq.call_classically(grid=grid, ctrl=ctrl, flag=0)
    assert isinstance(out_grid, np.ndarray)
    assert out_grid[2, 1] == 0
    assert out_flag == 0


def test_nd_grid_row_col_asymmetry():
    """grid[0,1] and grid[1,0] play different roles (CNOT direction).

    Setting grid[0,1] vs grid[1,0] should give different outputs.
    """
    bloq = TestNDGrid()
    ctrl = np.zeros(2, dtype=np.uint8)

    # Case A: grid[0,1]=1
    grid_a = np.zeros((3, 2), dtype=np.uint8)
    grid_a[0, 1] = 1
    out_a, _, _ = bloq.call_classically(grid=grid_a, ctrl=ctrl, flag=0)

    # Case B: grid[1,0]=1
    grid_b = np.zeros((3, 2), dtype=np.uint8)
    grid_b[1, 0] = 1
    out_b, _, _ = bloq.call_classically(grid=grid_b, ctrl=ctrl, flag=0)

    assert not np.array_equal(out_a, out_b)


def test_nd_grid_exhaustive_vs_reference():
    """Compare call_classically against the Python reference for all 2^8 grid/ctrl inputs."""
    bloq = TestNDGrid()
    for bits in itertools.product([0, 1], repeat=8):
        grid = np.array(bits[:6], dtype=np.uint8).reshape(3, 2)
        ctrl = np.array(bits[6:8], dtype=np.uint8)
        flag = 0
        out_grid, out_ctrl, out_flag = bloq.call_classically(grid=grid, ctrl=ctrl, flag=flag)
        ref_grid, ref_ctrl, ref_flag = _reference_nd_grid(grid, ctrl, flag)
        np.testing.assert_array_equal(out_grid, ref_grid, err_msg=f"grid mismatch for {bits}")
        np.testing.assert_array_equal(out_ctrl, ref_ctrl, err_msg=f"ctrl mismatch for {bits}")
        assert out_flag == ref_flag, f"flag mismatch for {bits}"


def test_nd_grid_flag_set():
    """Starting with flag=1 should toggle its output when grid[2,1] → 1."""
    bloq = TestNDGrid()
    grid = np.zeros((3, 2), dtype=np.uint8)
    grid[2, 0] = 1
    ctrl = np.array([0, 1], dtype=np.uint8)
    _, _, out_flag = bloq.call_classically(grid=grid, ctrl=ctrl, flag=1)
    # Toffoli fires → grid[2,1]=1 → CNOT XORs flag: 1 ^ 1 = 0
    assert out_flag == 0


def test_nd_grid_signature_shapes():
    """Verify register shapes in the signature."""
    bloq = TestNDGrid()
    sig = bloq.signature
    assert sig.get_left('grid').shape == (3, 2)
    assert sig.get_left('ctrl').shape == (2,)
    assert sig.get_left('flag').shape == ()


# ===========================================================================
# TestND3Grid tests (rank-3 cube register)
# ===========================================================================


def test_nd3_grid_all_zeros():
    """All-zero input: only the X on cube[0,0,0] fires."""
    bloq = TestND3Grid()
    cube = np.zeros((2, 2, 2), dtype=np.uint8)
    out_cube, out_aux = bloq.call_classically(cube=cube, aux=0)
    expected = np.zeros((2, 2, 2), dtype=np.uint8)
    expected[0, 0, 0] = 1
    np.testing.assert_array_equal(out_cube, expected)
    assert out_aux == 0


def test_nd3_grid_chain_propagation():
    """All ones in the cube propagate through the chain."""
    bloq = TestND3Grid()
    cube = np.ones((2, 2, 2), dtype=np.uint8)
    out_cube, out_aux = bloq.call_classically(cube=cube, aux=0)
    ref_cube, ref_aux = _reference_nd3_grid(cube, 0)
    np.testing.assert_array_equal(out_cube, ref_cube)
    assert out_aux == ref_aux


def test_nd3_grid_single_bit_positions_distinct():
    """Setting individual bits produces distinct outputs (no position aliasing)."""
    bloq = TestND3Grid()
    results = set()
    for idx in np.ndindex(2, 2, 2):
        cube = np.zeros((2, 2, 2), dtype=np.uint8)
        cube[idx] = 1
        out_cube, out_aux = bloq.call_classically(cube=cube, aux=0)
        assert isinstance(out_cube, np.ndarray)
        results.add((tuple(out_cube.flat), out_aux))
    assert len(results) == 8


def test_nd3_grid_exhaustive_vs_reference():
    """Compare call_classically against the reference for all 2^9 inputs."""
    bloq = TestNDGrid() if False else TestND3Grid()  # pylint: disable=using-constant-test
    for bits in itertools.product([0, 1], repeat=9):
        cube = np.array(bits[:8], dtype=np.uint8).reshape((2, 2, 2))
        aux = bits[8]
        out_cube, out_aux = bloq.call_classically(cube=cube, aux=aux)
        ref_cube, ref_aux = _reference_nd3_grid(cube, aux)
        np.testing.assert_array_equal(out_cube, ref_cube, err_msg=f"cube mismatch for {bits}")
        assert out_aux == ref_aux, f"aux mismatch for {bits}"


def test_nd3_grid_exhaustive_vs_reference_fastsim():
    """Compare call_classically against the reference for all 2^9 inputs."""
    rsqualtran = pytest.importorskip('rsqualtran')

    bloq = TestND3Grid()
    simulator = rsqualtran.QLTFastsim.from_bloq(bloq)
    for bits in itertools.product([0, 1], repeat=9):
        cube = np.array(bits[:8], dtype=np.uint8).reshape((2, 2, 2))
        aux = bits[8]
        out_cube, out_aux = simulator.call_classically(
            cube=cube, aux=aux
        )  # pylint: disable=unbalanced-tuple-unpacking
        ref_cube, ref_aux = _reference_nd3_grid(cube, aux)
        np.testing.assert_array_equal(out_cube, ref_cube, err_msg=f"cube mismatch for {bits}")
        assert out_aux == ref_aux, f"aux mismatch for {bits}"


def test_nd3_grid_signature_shapes():
    """Verify register shapes in the signature."""
    bloq = TestND3Grid()
    sig = bloq.signature
    assert sig.get_left('cube').shape == (2, 2, 2)
    assert sig.get_left('aux').shape == ()


def test_nd3_grid_aux_toggles():
    """When aux starts as 1 it should be toggled by the chain."""
    bloq = TestND3Grid()
    cube = np.zeros((2, 2, 2), dtype=np.uint8)
    cube[0, 0, 1] = 1
    cube[0, 1, 1] = 1
    _, out_aux_0 = bloq.call_classically(cube=cube.copy(), aux=0)
    _, out_aux_1 = bloq.call_classically(cube=cube.copy(), aux=1)
    assert out_aux_0 != out_aux_1
