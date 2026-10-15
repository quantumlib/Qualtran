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

from __future__ import annotations

import io
import pathlib

import pytest

import kickmix as km


def test_circuit():
    with pytest.raises(ValueError, match="unknown operation name"):
        km.Circuit("test")
    c = km.Circuit("X q0")
    assert len(c) == 1


def test_circuit_file_io(tmp_path: pathlib.Path):
    circuit = km.Circuit("""
        APPEND_TO_REGISTER q0 r0
        APPEND_TO_REGISTER q1 r0
        APPEND_TO_REGISTER b0 r1
        REGISTER r0 "quantum_reg"
        REGISTER r1 "classical_reg"
        CCX q0 q1 q2
        CX q0 q1
        X q0
        HMR q2 b0
        Z_POW q0 0.25 if b0
    """)

    # Default format ('kmx' for to_file, 'auto' for from_file) with pathlib.Path and str.
    kmx_path = tmp_path / "test.kmx"
    circuit.to_file(kmx_path)
    assert kmx_path.read_text() == f"{circuit}\n"
    assert km.Circuit.from_file(kmx_path) == circuit
    assert km.Circuit.from_file(kmx_path, format="auto") == circuit
    assert km.Circuit.from_file(str(kmx_path), format="kmx") == circuit

    # Explicit format='kmx' with str path.
    kmx_str_path = str(tmp_path / "test_str.kmx")
    circuit.to_file(kmx_str_path, format="kmx")
    assert km.Circuit.from_file(kmx_str_path) == circuit

    # Binary format ('kmb') with pathlib.Path and str.
    kmb_path = tmp_path / "test.kmb"
    circuit.to_file(kmb_path, format="kmb")
    raw_kmb = kmb_path.read_bytes()
    assert raw_kmb.startswith(bytes.fromhex("d750c7d5c329d326e3cc9f6834f2b8bf"))
    assert km.Circuit.from_file(kmb_path) == circuit
    assert km.Circuit.from_file(kmb_path, format="auto") == circuit
    assert km.Circuit.from_file(kmb_path, format="kmb") == circuit
    assert km.Circuit.from_file(str(kmb_path), format="kmb") == circuit

    # Opened file objects: 'kmx' supports both text and binary modes.
    kmx_open_text_path = tmp_path / "open_text.kmx"
    with open(kmx_open_text_path, "w") as f:
        circuit.to_file(f)
    with open(kmx_open_text_path, "r") as f:
        assert km.Circuit.from_file(f) == circuit
    with open(kmx_open_text_path, "r") as f:
        assert km.Circuit.from_file(f, format="kmx") == circuit

    kmx_open_bin_path = tmp_path / "open_bin.kmx"
    with open(kmx_open_bin_path, "wb") as f:
        circuit.to_file(f, format="kmx")
    with open(kmx_open_bin_path, "rb") as f:
        assert km.Circuit.from_file(f) == circuit
    with open(kmx_open_bin_path, "rb") as f:
        assert km.Circuit.from_file(f, format="kmx") == circuit

    # Opened file objects: 'kmb' in binary mode.
    kmb_open_bin_path = tmp_path / "open_bin.kmb"
    with open(kmb_open_bin_path, "wb") as f:
        circuit.to_file(f, format="kmb")
    with open(kmb_open_bin_path, "rb") as f:
        assert km.Circuit.from_file(f) == circuit
    with open(kmb_open_bin_path, "rb") as f:
        assert km.Circuit.from_file(f, format="kmb") == circuit

    # In-memory StringIO and BytesIO streams.
    sio = io.StringIO()
    circuit.to_file(sio, format="kmx")
    sio.seek(0)
    assert km.Circuit.from_file(sio) == circuit

    bio_kmx = io.BytesIO()
    circuit.to_file(bio_kmx, format="kmx")
    bio_kmx.seek(0)
    assert km.Circuit.from_file(bio_kmx) == circuit

    bio_kmb = io.BytesIO()
    circuit.to_file(bio_kmb, format="kmb")
    bio_kmb.seek(0)
    assert km.Circuit.from_file(bio_kmb) == circuit
    bio_kmb.seek(0)
    assert km.Circuit.from_file(bio_kmb, format="kmb") == circuit

    # Empty circuit round-trip in both formats.
    empty = km.Circuit()
    empty_kmx = tmp_path / "empty.kmx"
    empty_kmb = tmp_path / "empty.kmb"
    empty.to_file(empty_kmx)
    empty.to_file(empty_kmb, format="kmb")
    assert km.Circuit.from_file(empty_kmx) == empty
    assert km.Circuit.from_file(empty_kmb) == empty
    assert km.Circuit.from_file(empty_kmb, format="kmb") == empty


def test_circuit_file_io_errors(tmp_path: pathlib.Path):
    circuit = km.Circuit("X q0")
    kmx_path = tmp_path / "valid.kmx"
    kmb_path = tmp_path / "valid.kmb"
    circuit.to_file(kmx_path, format="kmx")
    circuit.to_file(kmb_path, format="kmb")

    # Invalid format.
    bad_format_path = tmp_path / "should_not_exist.kmx"
    with pytest.raises(ValueError, match="Unrecognized format"):
        circuit.to_file(bad_format_path, format="bmx")  # type: ignore[arg-type]
    assert not bad_format_path.exists()

    with pytest.raises(ValueError, match="Unrecognized format"):
        km.Circuit.from_file(kmx_path, format="bmx")  # type: ignore[arg-type]

    # Wrong format when reading.
    with pytest.raises(ValueError, match="magic bytes"):
        km.Circuit.from_file(kmx_path, format="kmb")
    with pytest.raises(ValueError):
        km.Circuit.from_file(kmb_path, format="kmx")

    # 'kmb' requires binary mode when passing an open file object.
    with open(tmp_path / "bad_text.kmb", "w") as f:
        with pytest.raises((TypeError, ValueError)):
            circuit.to_file(f, format="kmb")
    with open(kmb_path, "r") as f:
        with pytest.raises((TypeError, ValueError)):
            km.Circuit.from_file(f, format="kmb")
    with open(kmb_path, "r") as f:
        with pytest.raises((TypeError, ValueError)):
            km.Circuit.from_file(f)

    # Missing file / unwritable file.
    with pytest.raises(ValueError, match="Failed to open file for reading"):
        km.Circuit.from_file(tmp_path / "missing.kmx")
    with pytest.raises(ValueError, match="Failed to open file for writing"):
        circuit.to_file(tmp_path / "missing_dir" / "out.kmx")

    # Invalid path type.
    with pytest.raises(TypeError):
        km.Circuit.from_file(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        circuit.to_file(123)  # type: ignore[arg-type]


def test_circuit_operation_statistics_resources():
    empty_stats = km.Circuit("").operation_statistics()
    assert empty_stats.num_toffoli_like == 0
    assert empty_stats.num_t_like == 0
    assert empty_stats.num_instructions == 0
    assert empty_stats.num_z_pow() == 0
    assert empty_stats.z_pow_clifford_hierarchy_counts == {}

    c = km.Circuit("""
        CCX q0 q1 q2
        CCZ q0 q1 q2 if b0
        Z q0
        Z_POW q0 0.0
        Z_POW q0 0.5
        Z_POW q0 1.0
        Z_POW q0 1.5
        Z_POW q0 -0.5 if b0
        Z_POW q0 0.25
        Z_POW q0 0.75
        Z_POW q0 -0.25 if b0
        Z_POW q0 0.125
        Z_POW q0 0.1 if b1
    """)
    stats = c.operation_statistics()
    assert stats.num_toffoli_like == 2
    assert stats.num_t_like == 3
    assert stats.num_z_pow() == 10
    assert stats.num_z_pow(max_clifford_hierarchy_level=2) == 5
    assert stats.num_z_pow(min_clifford_hierarchy_level=3, max_clifford_hierarchy_level=3) == 3
    assert stats.num_z_pow(min_clifford_hierarchy_level=4) == 2
    assert stats.num_instructions == 13
    assert stats.z_pow_clifford_hierarchy_counts[0] == 1
    assert stats.z_pow_clifford_hierarchy_counts[1] == 1
    assert stats.z_pow_clifford_hierarchy_counts[2] == 3
    assert stats.z_pow_clifford_hierarchy_counts[3] == 3
    assert stats.z_pow_clifford_hierarchy_counts[4] == 1
    assert sum(stats.z_pow_clifford_hierarchy_counts.values()) == 10
    assert stats == km.Circuit(str(c)).operation_statistics()
    assert stats != empty_stats
    assert (
        repr(stats)
        == "<km.OperationStatistics num_instructions=13 num_toffoli_like=2 num_t_like=3>"
    )


def test_circuit_operation_statistics_mapping():
    empty_stats = km.Circuit("").operation_statistics()
    assert not empty_stats
    assert len(empty_stats) == 0
    assert dict(empty_stats) == {}
    assert "X" not in empty_stats
    assert empty_stats["X"] == 0
    assert empty_stats.get("X") is None
    assert empty_stats.get("X", 0) == 0

    c = km.Circuit("""
        X q0
        X q0 if b0
        CX q0 q1
        CX q0 q1 if b0
        PUSH_CONDITION if b0
        CCX q0 q1 q2
        CCZ q0 q1 q2 if b1
        CSWAP q0 q1 q2
        CSWAP q0 q1 q2 if b1
        Z_POW q0 0.25
        Z_POW q0 0.125 if b1
        INFO_TAG "hello"
        POP_CONDITION
        DEBUG_PRINT
        DEBUG_PRINT q0
        DEBUG_PRINT b0 if b1
    """)
    stats = c.operation_statistics()
    counts = dict(stats)
    assert bool(stats)
    assert len(stats) == 10
    assert "X" in stats
    assert "CSWAP" in stats
    assert "INFO_TAG" in stats
    assert "CZ" not in stats
    assert "NOT_AN_OP" not in stats
    assert 123 not in stats
    assert stats["X"] == 2
    assert stats["CSWAP"] == 2
    assert stats["INFO_TAG"] == 1
    assert stats.num_toffoli_like == 4
    assert stats["CZ"] == 0
    assert stats["HMR"] == 0
    assert stats.get("X") == 2
    assert stats.get("CZ") is None
    assert stats.get("CZ", 0) == 0
    assert stats.get("NOT_AN_OP") is None
    assert stats.get("NOT_AN_OP", 99) == 99
    assert counts == {
        "X": 2,
        "CX": 2,
        "CSWAP": 2,
        "CCX": 1,
        "CCZ": 1,
        "Z_POW": 2,
        "INFO_TAG": 1,
        "DEBUG_PRINT": 3,
        "POP_CONDITION": 1,
        "PUSH_CONDITION": 1,
    }
    assert list(stats) == stats.keys()
    assert stats.items() == list(counts.items())
    assert stats.values() == list(counts.values())

    with pytest.raises(KeyError):
        _ = stats["NOT_AN_OP"]

    sim = km.Simulator(batch_size=4, counting_operations=True, debug_print=False)
    assert dict(sim.operation_statistics()) == {}
