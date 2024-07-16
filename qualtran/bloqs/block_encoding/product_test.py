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
from qualtran.bloqs.basic_gates import (
    CNOT,
    Hadamard,
    IntEffect,
    IntState,
    TGate,
    XGate,
    ZeroEffect,
    ZeroState,
)
from qualtran.bloqs.block_encoding.product import (
    _product_block_encoding,
    _product_block_encoding_properties,
    _product_block_encoding_symb,
    Product,
)
from qualtran.bloqs.block_encoding.unitary import Unitary
from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim


def test_product(bloq_autotester):
    bloq_autotester(_product_block_encoding)
    bloq_autotester(_product_block_encoding_properties)
    bloq_autotester(_product_block_encoding_symb)


def test_product_signature():
    assert _product_block_encoding().signature == Signature(
        [Register("system", QAny(1)), Register("ancilla", QAny(1))]
    )
    assert _product_block_encoding_properties().signature == Signature(
        [Register("system", QAny(1)), Register("ancilla", QAny(3)), Register("resource", QAny(1))]
    )
    assert _product_block_encoding_symb().signature == Signature(
        [
            Register("system", QAny(1)),
            Register("ancilla", QAny(sympy.Max(sympy.Symbol('a1'), sympy.Symbol('a2')) + 1)),
        ]
    )


def test_product_checks():
    with pytest.raises(ValueError):
        _ = Product(())
    with pytest.raises(ValueError):
        _ = Product((Unitary(TGate()), Unitary(CNOT())))


def test_product_params():
    bloq = _product_block_encoding()
    assert bloq.system_bitsize == 1
    assert bloq.alpha == 1
    assert bloq.epsilon == 0
    assert bloq.ancilla_bitsize == 1
    assert bloq.resource_bitsize == 0

    bloq = _product_block_encoding_properties()
    assert bloq.system_bitsize == 1
    assert bloq.alpha == 0.5 * 0.5
    assert bloq.epsilon == 0.5 * 0.01 + 0.5 * 0.1
    assert bloq.ancilla_bitsize == max(2, 1) + 1
    assert bloq.resource_bitsize == max(1, 1)

    bloq = _product_block_encoding_symb()
    assert bloq.system_bitsize == 1
    assert bloq.alpha == sympy.Symbol('alpha1') * sympy.Symbol('alpha2')
    assert bloq.epsilon == sympy.Symbol('alpha1') * sympy.Symbol('eps1') + sympy.Symbol(
        'alpha2'
    ) * sympy.Symbol('eps2')
    assert bloq.ancilla_bitsize == sympy.Max(sympy.Symbol('a1'), sympy.Symbol('a2')) + 2 - 1
    assert bloq.resource_bitsize == 0


def test_product_tensors():
    bb = BloqBuilder()
    system = bb.add_register("system", 1)
    ancilla = cast(Soquet, bb.add(ZeroState()))
    system, ancilla = bb.add_t(_product_block_encoding(), system=system, ancilla=ancilla)
    bb.add(ZeroEffect(), q=ancilla)
    bloq = bb.finalize(system=system)

    from_gate = np.matmul(TGate().tensor_contract(), Hadamard().tensor_contract())
    from_tensors = bloq.tensor_contract()
    np.testing.assert_allclose(from_gate, from_tensors)


def test_product_single_tensors():
    bb = BloqBuilder()
    system = bb.add_register("system", 1)
    system = cast(Soquet, bb.add(Product((Unitary(TGate()),)), system=system))
    bloq = bb.finalize(system=system)

    from_gate = TGate().tensor_contract()
    from_tensors = bloq.tensor_contract()
    np.testing.assert_allclose(from_gate, from_tensors)


def test_product_properties_tensors():
    bb = BloqBuilder()
    system = bb.add_register("system", 1)
    ancilla = cast(Soquet, bb.add(IntState(0, 3)))
    resource = cast(Soquet, bb.add(ZeroState()))
    system, ancilla, resource = bb.add_t(
        _product_block_encoding_properties(), system=system, ancilla=ancilla, resource=resource
    )
    bb.add(ZeroEffect(), q=resource)
    bb.add(IntEffect(0, 3), val=ancilla)
    bloq = bb.finalize(system=system)

    from_gate = np.matmul(TGate().tensor_contract(), Hadamard().tensor_contract())
    from_tensors = bloq.tensor_contract()
    np.testing.assert_allclose(from_gate, from_tensors)


def test_product_cirq():
    qubits = cirq.LineQubit.range(3)
    op = Product((Product((Unitary(XGate()), Unitary(XGate()))), Unitary(XGate()))).on_registers(
        system=qubits[:1], ancilla=qubits[1:]
    )
    circuit = cirq.Circuit(
        cirq.decompose_once(
            op,
            context=cirq.DecompositionContext(
                cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
            ),
        )
    )
    initial_state = [0, 0, 0]
    final_state = [1, 0, 0]
    assert_circuit_inp_out_cirqsim(circuit, qubits, initial_state, final_state)
