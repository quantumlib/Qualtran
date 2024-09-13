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

from typing import cast, Tuple

import cirq
import numpy as np
import pytest
import sympy
from attr import frozen

from qualtran import BloqBuilder, QAny, QBit, Register, Signature, Soquet
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
from qualtran.bloqs.block_encoding.block_encoding_base import BlockEncoding
from qualtran.bloqs.block_encoding.product import (
    _product_block_encoding,
    _product_block_encoding_properties,
    _product_block_encoding_symb,
    Product,
)
from qualtran.bloqs.block_encoding.unitary import Unitary
from qualtran.bloqs.for_testing.matrix_gate import MatrixGate
from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim
from qualtran.testing import assert_equivalent_bloq_example_counts, execute_notebook


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


@frozen
class TestPrepareOracle(PrepareOracle):
    @property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (Register('z', QBit()),)

    @property
    def junk_registers(self) -> Tuple[Register, ...]:
        return (Register('a', QAny(5)),)


@frozen
class TestBlockEncoding(BlockEncoding):
    alpha: float = 1
    epsilon: float = 0
    system_bitsize: int = 1
    ancilla_bitsize: int = 1
    resource_bitsize: int = 1

    @property
    def signature(self) -> Signature:
        return Signature.build(system=1, ancilla=1, resource=1)

    @property
    def signal_state(self) -> BlackBoxPrepare:
        return BlackBoxPrepare(TestPrepareOracle())


def test_product_checks():
    with pytest.raises(ValueError):
        _ = Product(())
    with pytest.raises(ValueError):
        _ = Product((Unitary(TGate()), Unitary(CNOT())))
    with pytest.raises(ValueError):
        _ = Product((Unitary(TGate()), TestBlockEncoding()))


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


def test_product_random():
    random_state = np.random.RandomState(1234)

    for _ in range(10):
        n = random_state.randint(3, 6)
        bitsize = random_state.randint(1, 3)
        gates = [MatrixGate.random(bitsize, random_state=random_state) for _ in range(n)]

        bloq = Product(tuple(Unitary(gate) for gate in gates))
        bb = BloqBuilder()
        system = bb.add_register("system", cast(int, bloq.system_bitsize))
        ancilla = cast(Soquet, bb.add(IntState(0, bloq.ancilla_bitsize)))
        system, ancilla = bb.add_t(bloq, system=system, ancilla=ancilla)
        bb.add(IntEffect(0, cast(int, bloq.ancilla_bitsize)), val=ancilla)
        bloq = bb.finalize(system=system)

        from_gate = np.linalg.multi_dot(tuple(gate.tensor_contract() for gate in gates))
        from_tensors = bloq.tensor_contract()
        np.testing.assert_allclose(from_gate, from_tensors)


def test_product_signal_state():
    assert isinstance(_product_block_encoding().signal_state.prepare, PrepareIdentity)
    _ = _product_block_encoding().signal_state.decompose_bloq()


def test_product_counts():
    assert_equivalent_bloq_example_counts(_product_block_encoding)


@pytest.mark.notebook
def test_notebook():
    execute_notebook('product')
