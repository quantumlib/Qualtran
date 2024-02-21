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

import numpy as np
import pytest

from qualtran import BloqBuilder
from qualtran.bloqs.chemistry.prepare_mps.compile_gate import CompileGateGivenVectorsWithoutPG
from qualtran.testing import assert_valid_bloq_decomposition


def accuracy(state1, state2):
    return abs(np.dot(state1, state2.conj()))


def cInPolar (c):
  return f"{round(abs(c),2)} ∠ {round(np.angle(c, deg=True),2)}º"

def formatU (U):
  formatted = ""
  for r in U:
    formatted += " ".join([f"({cInPolar(c)})" for c in r]) + "\n"
  return formatted


# these gates can be approximated exactly with the given rot_reg_size
@pytest.mark.parametrize(
    "rot_reg_size, gate_cols",
    [
        [2, (((0.5+0.5j), (0.5-0.5j)),
             ((-0.5-0.5j), (0.5-0.5j)))],
        [4, (((-0.191341716182545+0.961939766255644j), (-0.038060233744357+0.191341716182545j)),
             ((0.038060233744356-0.191341716182545j), (-0.191341716182545+0.961939766255644j)))],
        [2,  (((0.5625+0.125j), (-0.125-0.4375j), (0.0625-0.25j), (0.5-0.0625j)),
              ((0.375+0.0625j), (0.4375-0.125j), (-0.25+0.0625j), (-0.4375-0.5j)),
              ((-0.375-0.125j), (0.125-0.375j), (-0.625-0.375j), (0.125+0.125j)),
              ((-0.5+0j), (-0-0.5j), (0.5-0j), -0.5j))]
    ],
)
def test_exact_gate_compilation(rot_reg_size, gate_cols):
    gate_compiler = CompileGateGivenVectorsWithoutPG(rot_reg_size, tuple(gate_cols))
    assert_valid_bloq_decomposition(gate_compiler)
    compiled_gate = gate_compiler.tensor_contract().T
    assert np.allclose(compiled_gate, np.array(gate_cols))


@pytest.mark.parametrize(
    "rot_reg_size, gate_cols",
    [
        [2, (((0.5+0.5j), (0.5-0.5j)),
             ((-0.5-0.5j), (0.5-0.5j)))],
        [4, (((-0.191341716182545+0.961939766255644j), (-0.038060233744357+0.191341716182545j)),
             ((0.038060233744356-0.191341716182545j), (-0.191341716182545+0.961939766255644j)))],
        [2,  (((0.5625+0.125j), (-0.125-0.4375j), (0.0625-0.25j), (0.5-0.0625j)),
              ((0.375+0.0625j), (0.4375-0.125j), (-0.25+0.0625j), (-0.4375-0.5j)),
              ((-0.375-0.125j), (0.125-0.375j), (-0.625-0.375j), (0.125+0.125j)),
              ((-0.5+0j), (-0-0.5j), (0.5-0j), -0.5j))]
    ],
)
def test_partial_gate_compilation(rot_reg_size, gate_cols):
    gate_compiler = CompileGateGivenVectorsWithoutPG(rot_reg_size, tuple(gate_cols))
    assert_valid_bloq_decomposition(gate_compiler)
    compiled_gate = gate_compiler.tensor_contract().T
    assert np.allclose(compiled_gate[range(len(gate_cols)),:], np.array(gate_cols))