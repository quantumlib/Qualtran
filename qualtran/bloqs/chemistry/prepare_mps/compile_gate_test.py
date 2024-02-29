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
import attrs
from typing import Tuple, Dict

from qualtran import BloqBuilder, Bloq, Signature, SoquetT
from qualtran.bloqs.chemistry.prepare_mps.compile_gate import CompileGateFromColumnsNoPG, CompileGateFromColumns
from qualtran.testing import assert_valid_bloq_decomposition
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState



# these gates can be approximated exactly with the given phase_bitsize
@pytest.mark.parametrize(
    "phase_bitsize, gate_cols",
    [
        [2, (((0.5+0.5j), (0.5-0.5j)),
             ((-0.5-0.5j), (0.5-0.5j)))],
        [4, (((-0.191341716182545+0.961939766255644j), (-0.038060233744357+0.191341716182545j)),
             ((0.038060233744356-0.191341716182545j), (-0.191341716182545+0.961939766255644j)))],
        [2,  (((-0-0.5j), (0.5-0j), (-0.5+0.5j), -0j),
              ((0.5+0.5j), (0.5-0.5j), 0j, 0j),
              (0.5j, (-0.5+0j), (-0.5+0.5j), 0j),
              (0j, 0j, (-0+0j), (-1+0j)))]
    ],
)
def test_exact_gate_compilation(phase_bitsize: int, gate_cols: Tuple[complex,...]):
    gate_compiler = CompileGateFromColumnsNoPG(phase_bitsize, tuple(gate_cols))
    assert_valid_bloq_decomposition(gate_compiler)
    compiled_gate = gate_compiler.tensor_contract().T
    assert np.allclose(compiled_gate, np.array(gate_cols))


@pytest.mark.parametrize(
    "phase_bitsize, gate_cols",
    [
        [4, (((-0.191341716182545+0.961939766255644j), (-0.038060233744357+0.191341716182545j)),)],
        [2,  (((-0-0.5j), (0.5-0j), (-0.5+0.5j), -0j),
              ((0.5+0.5j), (0.5-0.5j), 0j, 0j),
              (0.5j, (-0.5+0j), (-0.5+0.5j), 0j))]
    ],
)
def test_partial_gate_compilation(phase_bitsize: int, gate_cols: Tuple[complex,...]):
    gate_compiler = CompileGateFromColumnsNoPG(phase_bitsize, tuple(gate_cols))
    assert_valid_bloq_decomposition(gate_compiler)
    compiled_gate = gate_compiler.tensor_contract().T
    assert np.allclose(compiled_gate[range(len(gate_cols)),:], np.array(gate_cols))


# this class is just for testing, as I want to be able to contract the
# entire matrix without providing a precise gate input
@attrs.frozen
class UUt (Bloq):
    phase_bitsize: int # number of ancilla qubits used to encode the state preparation's rotations
    gate_cols: Tuple[Tuple] # tuple with the columns/rows of the gate that are specified

    @property
    def signature(self):
        return Signature.build(gate_input=self.gate_bitsize)

    @property
    def gate_bitsize(self):
        return (len(self.gate_cols[0])-1).bit_length()
    
    def build_composite_bloq(self, bb: BloqBuilder, *, gate_input: SoquetT) -> Dict[str, SoquetT]:
        gate_compiler = CompileGateFromColumns(gate_cols=self.gate_cols, phase_bitsize=self.phase_bitsize, uncompute=False)
        gate_compiler_adj = CompileGateFromColumns(gate_cols=self.gate_cols, phase_bitsize=self.phase_bitsize, uncompute=True)
        soqs = {}
        soqs["gate_input"] = gate_input
        soqs["phase_grad"] = bb.add(PhaseGradientState(bitsize=self.phase_bitsize))
        soqs["reflection_ancilla"] = bb.allocate(1)
        soqs = bb.add_d(gate_compiler, **soqs)
        soqs = bb.add_d(gate_compiler_adj, **soqs)
        bb.free(soqs.pop("reflection_ancilla"))
        bb.add(PhaseGradientState(bitsize=self.phase_bitsize).adjoint(), phase_grad=soqs.pop("phase_grad"))
        return soqs

@pytest.mark.parametrize(
    "phase_bitsize, gate_cols",
    [
        [2, (((0.5+0.5j), (0.5-0.5j)),
             ((-0.5-0.5j), (0.5-0.5j)))],
        [2,  (((-0-0.5j), (0.5-0j), (-0.5+0.5j), -0j),
              ((0.5+0.5j), (0.5-0.5j), 0j, 0j),
              (0.5j, (-0.5+0j), (-0.5+0.5j), 0j),
              (0j, 0j, (-0+0j), (-1+0j)))]
    ],
)
def test_gate_compilation_adjoint(phase_bitsize: int, gate_cols: Tuple[complex,...]):
    uut = UUt(phase_bitsize, gate_cols)
    assert np.allclose(uut.tensor_contract(), np.eye(len(gate_cols[0])))