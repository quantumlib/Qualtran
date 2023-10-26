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

from functools import cached_property
from typing import Dict

import cirq
import numpy as np
import quimb.tensor as qtn
from attrs import frozen
from numpy.typing import NDArray

from qualtran import Bloq, BloqBuilder, Register, Side, Signature, Soquet, SoquetT
from qualtran._infra.composite_bloq import _get_dangling_soquets
from qualtran.bloqs.basic_gates import CNOT, XGate, ZGate
from qualtran.simulation.tensor import bloq_to_dense, get_right_and_left_inds
from qualtran.testing import assert_valid_bloq_decomposition


@frozen
class TensorAdderTester(Bloq):
    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('x', bitsize=2, side=Side.LEFT),
                Register('qubits', bitsize=1, shape=(2,)),
                Register('y', bitsize=1, side=Side.RIGHT),
            ]
        )

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        tag,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):
        assert sorted(incoming.keys()) == ['qubits', 'x']
        in_qubits = incoming['qubits']
        assert in_qubits.shape == (2,)
        assert incoming['x'].reg.bitsize == 2

        assert sorted(outgoing.keys()) == ['qubits', 'y']
        out_qubits = outgoing['qubits']
        assert out_qubits.shape == (2,)
        assert outgoing['y'].reg.bitsize == 1

        data = np.zeros((2**2, 2, 2, 2, 2, 2))
        data[3, 0, 1, 0, 1, 0] = 1
        tn.add(
            qtn.Tensor(
                data=data,
                inds=(
                    incoming['x'],
                    in_qubits[0],
                    in_qubits[1],
                    outgoing['y'],
                    out_qubits[0],
                    out_qubits[1],
                ),
                tags=[tag],
            )
        )


def _old_bloq_to_dense(bloq: Bloq) -> NDArray:
    """Old code for tensor-contracting a bloq without wrapping it in length-1 composite bloq."""
    tn = qtn.TensorNetwork([])
    lsoqs = _get_dangling_soquets(bloq.signature, right=False)
    rsoqs = _get_dangling_soquets(bloq.signature, right=True)
    bloq.add_my_tensors(tn, None, incoming=lsoqs, outgoing=rsoqs)

    inds = get_right_and_left_inds(bloq.signature)
    matrix = tn.to_dense(*inds)
    return matrix


def test_bloq_to_dense():
    mat1 = _old_bloq_to_dense(TensorAdderTester())
    mat2 = bloq_to_dense(TensorAdderTester())
    np.testing.assert_allclose(mat1, mat2, atol=1e-8)

    # Right inds: qubits=(1,0), y=0
    right = 1 * 2**2 + 0 * 2**1 + 0 * 2**0

    # Left inds: x=3, qubits=(0,1)
    left = 3 * 2**2 + 0 * 2**1 + 1 * 2**0

    assert np.where(mat2) == (right, left)


@frozen
class XNest(Bloq):
    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(r=1)

    def build_composite_bloq(self, bb: 'BloqBuilder', r: 'SoquetT') -> Dict[str, 'SoquetT']:
        r = bb.add(XGate(), q=r)
        return {'r': r}


@frozen
class XDoubleNest(Bloq):
    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(s=1)

    def build_composite_bloq(self, bb: 'BloqBuilder', s: 'SoquetT') -> Dict[str, 'SoquetT']:
        s = bb.add(XNest(), r=s)
        return {'s': s}


def test_nest():
    x = XNest()
    should_be = cirq.unitary(cirq.X)
    np.testing.assert_allclose(should_be, x.tensor_contract())
    np.testing.assert_allclose(should_be, x.decompose_bloq().tensor_contract())


def test_double_nest():
    xx = XDoubleNest()
    should_be = cirq.unitary(cirq.X)
    np.testing.assert_allclose(should_be, xx.tensor_contract())
    np.testing.assert_allclose(should_be, xx.decompose_bloq().tensor_contract())


@frozen
class BloqWithNonTrivialInds(Bloq):
    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('q0', 1), Register('q1', 1)])

    def build_composite_bloq(
        self, bb: 'BloqBuilder', q0: Soquet, q1: Soquet
    ) -> Dict[str, 'SoquetT']:
        q0 = bb.add(XGate(), q=q0)
        q0, q1 = bb.add(CNOT(), ctrl=q0, target=q1)
        q1 = bb.add(ZGate(), q=q1)
        return {'q0': q0, 'q1': q1}


def test_bloq_with_non_trivial_inds():
    bloq = BloqWithNonTrivialInds()
    assert_valid_bloq_decomposition(bloq)
    cirq_qubits = cirq.LineQubit.range(2)
    cirq_quregs = {'q0': [cirq_qubits[0]], 'q1': [cirq_qubits[1]]}
    cirq_circuit, _ = bloq.decompose_bloq().to_cirq_circuit(**cirq_quregs)
    cirq_unitary = cirq_circuit.unitary(qubit_order=cirq_qubits)
    np.testing.assert_allclose(cirq_unitary, bloq.decompose_bloq().tensor_contract())
    np.testing.assert_allclose(cirq_unitary, bloq.tensor_contract())
