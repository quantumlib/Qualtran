#  Copyright 2024 Google LLC
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

from typing import cast

import cirq
import numpy as np
import pytest
import sympy

from qualtran import BloqBuilder, QAny, Register, Signature, Soquet
from qualtran.bloqs.basic_gates import CNOT, Hadamard, TGate, XGate, ZeroEffect, ZeroState
from qualtran.bloqs.block_encoding.tensor_product import (
    _tensor_product_block_encoding,
    _tensor_product_block_encoding_properties,
    _tensor_product_block_encoding_symb,
    TensorProduct,
)
from qualtran.bloqs.block_encoding.unitary import Unitary
from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim
from qualtran.testing import execute_notebook


def test_tensor_product(bloq_autotester):
    bloq_autotester(_tensor_product_block_encoding)
    bloq_autotester(_tensor_product_block_encoding_properties)
    bloq_autotester(_tensor_product_block_encoding_symb)


def test_tensor_product_signature():
    assert _tensor_product_block_encoding().signature == Signature([Register("system", QAny(2))])
    assert _tensor_product_block_encoding_properties().signature == Signature(
        [Register("system", QAny(3)), Register("ancilla", QAny(3)), Register("resource", QAny(2))]
    )
    assert _tensor_product_block_encoding_symb().signature == Signature(
        [
            Register("system", QAny(2)),
            Register("ancilla", QAny(sympy.Symbol('a1') + sympy.Symbol('a2'))),
        ]
    )
    with pytest.raises(ValueError):
        _ = TensorProduct(())


def test_tensor_product_params():
    bloq = _tensor_product_block_encoding()
    assert bloq.system_bitsize == 1 + 1
    assert bloq.alpha == 1
    assert bloq.epsilon == 0
    assert bloq.ancilla_bitsize == 0
    assert bloq.resource_bitsize == 0

    bloq = _tensor_product_block_encoding_properties()
    assert bloq.system_bitsize == 1 + 2
    assert bloq.alpha == 0.5 * 0.5
    assert bloq.epsilon == 0.5 * 0.01 + 0.5 * 0.1
    assert bloq.ancilla_bitsize == 2 + 1
    assert bloq.resource_bitsize == 1 + 1

    bloq = _tensor_product_block_encoding_symb()
    assert bloq.system_bitsize == 2
    assert bloq.alpha == sympy.Symbol('alpha1') * sympy.Symbol('alpha2')
    assert bloq.epsilon == sympy.Symbol('alpha1') * sympy.Symbol('eps1') + sympy.Symbol(
        'alpha2'
    ) * sympy.Symbol('eps2')
    assert bloq.ancilla_bitsize == sympy.Symbol('a1') + sympy.Symbol('a2')
    assert bloq.resource_bitsize == 0


def test_tensor_product_tensors():
    from_gate = TGate().tensor_contract()
    from_tensors = TensorProduct((Unitary(TGate()),)).tensor_contract()
    np.testing.assert_allclose(from_gate, from_tensors)

    from_gate = np.kron(TGate().tensor_contract(), Hadamard().tensor_contract())
    from_tensors = _tensor_product_block_encoding().tensor_contract()
    np.testing.assert_allclose(from_gate, from_tensors)


def test_tensor_product_override_tensors():
    bb = BloqBuilder()
    system = bb.add_register("system", 3)
    ancilla = bb.join(np.array([bb.add(ZeroState()), bb.add(ZeroState()), bb.add(ZeroState())]))
    resource = bb.join(np.array([bb.add(ZeroState()), bb.add(ZeroState())]))
    system, ancilla, resource = bb.add_t(
        _tensor_product_block_encoding_properties(),
        system=system,
        ancilla=ancilla,
        resource=resource,
    )
    for q in bb.split(cast(Soquet, ancilla)):
        bb.add(ZeroEffect(), q=q)
    for q in bb.split(cast(Soquet, resource)):
        bb.add(ZeroEffect(), q=q)
    bloq = bb.finalize(system=system)

    from_gate = np.kron(TGate().tensor_contract(), CNOT().tensor_contract())
    from_tensors = bloq.tensor_contract()
    np.testing.assert_allclose(from_gate, from_tensors)


def test_tensor_product_cirq():
    qubits = cirq.LineQubit.range(4)
    op = TensorProduct(
        (TensorProduct((Unitary(XGate()), Unitary(XGate()))), Unitary(CNOT()))
    ).on_registers(system=qubits)
    circuit = cirq.Circuit(
        cirq.decompose_once(
            op,
            context=cirq.DecompositionContext(
                cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
            ),
        )
    )
    initial_state = [0, 1, 1, 1]
    final_state = [1, 0, 1, 0]
    assert_circuit_inp_out_cirqsim(circuit, qubits, initial_state, final_state)


def test_tensor_product_signal_state():
    assert isinstance(_tensor_product_block_encoding().signal_state.prepare, PrepareIdentity)
    _ = _tensor_product_block_encoding().signal_state.decompose_bloq()


@pytest.mark.notebook
def test_notebook():
    execute_notebook('tensor_product')
