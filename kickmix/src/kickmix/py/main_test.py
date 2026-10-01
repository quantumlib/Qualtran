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

    # Default format ('kmx') with pathlib.Path and str.
    kmx_path = tmp_path / "test.kmx"
    circuit.to_file(kmx_path)
    assert kmx_path.read_text() == f"{circuit}\n"
    assert km.Circuit.from_file(kmx_path) == circuit
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
    assert km.Circuit.from_file(kmb_path, format="kmb") == circuit
    assert km.Circuit.from_file(str(kmb_path), format="kmb") == circuit

    # Empty circuit round-trip in both formats.
    empty = km.Circuit()
    empty_kmx = tmp_path / "empty.kmx"
    empty_kmb = tmp_path / "empty.kmb"
    empty.to_file(empty_kmx)
    empty.to_file(empty_kmb, format="kmb")
    assert km.Circuit.from_file(empty_kmx) == empty
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
