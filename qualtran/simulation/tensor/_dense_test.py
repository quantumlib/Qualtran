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
from typing import Dict, List

import cirq
import numpy as np
import quimb.tensor as qtn
from attrs import frozen

from qualtran import (
    Bloq,
    BloqBuilder,
    Connection,
    ConnectionT,
    QAny,
    QBit,
    Register,
    Side,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.basic_gates import CNOT, XGate, ZGate
from qualtran.simulation.tensor import bloq_to_dense
from qualtran.testing import assert_valid_bloq_decomposition


@frozen
class TensorAdderTester(Bloq):
    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('x', QAny(bitsize=2), side=Side.LEFT),
                Register('qubits', QBit(), shape=(2,)),
                Register('y', QBit(), side=Side.RIGHT),
            ]
        )

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        assert sorted(incoming.keys()) == ['qubits', 'x']
        in_qubits = incoming['qubits']
        assert isinstance(in_qubits, np.ndarray)
        assert in_qubits.shape == (2,)
        assert isinstance(incoming['x'], Connection)
        assert incoming['x'].right.reg.bitsize == 2

        assert sorted(outgoing.keys()) == ['qubits', 'y']
        out_qubits = outgoing['qubits']
        assert isinstance(out_qubits, np.ndarray)
        assert out_qubits.shape == (2,)
        assert isinstance(outgoing['y'], Connection)
        assert outgoing['y'].left.reg.bitsize == 1

        data = np.zeros((2**2, 2, 2, 2, 2, 2))
        data[3, 0, 1, 0, 1, 0] = 1
        data = data.reshape((2,) * 7)
        return [
            qtn.Tensor(
                data=data,
                inds=[
                    (incoming['x'], 0),
                    (incoming['x'], 1),
                    (in_qubits[0], 0),
                    (in_qubits[1], 0),
                    (outgoing['y'], 0),
                    (out_qubits[0], 0),
                    (out_qubits[1], 0),
                ],
            )
        ]


def test_bloq_to_dense():
    mat2 = bloq_to_dense(TensorAdderTester())

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
        return Signature([Register('q0', QBit()), Register('q1', QBit())])

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
    cirq_circuit = bloq.decompose_bloq().to_cirq_circuit(cirq_quregs=cirq_quregs)
    cirq_unitary = cirq_circuit.unitary(qubit_order=cirq_qubits)
    np.testing.assert_allclose(cirq_unitary, bloq.decompose_bloq().tensor_contract())
    np.testing.assert_allclose(cirq_unitary, bloq.tensor_contract())


class BloqWithTensorsAndDecomp(Bloq):
    def __init__(self):
        self.called_build_composite_bloq = False
        self.called_my_tensors = False

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(a=1, b=1)

    def build_composite_bloq(self, bb: 'BloqBuilder', a: Soquet, b: Soquet) -> Dict[str, 'SoquetT']:
        self.called_build_composite_bloq = True
        a, b = bb.add(CNOT(), ctrl=a, target=b)
        a, b = bb.add(CNOT(), ctrl=a, target=b)
        return {'a': a, 'b': b}

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        self.called_my_tensors = True
        return [
            qtn.Tensor(
                data=np.eye(4).reshape((2, 2, 2, 2)),
                inds=[
                    (outgoing['a'], 0),
                    (outgoing['b'], 0),
                    (incoming['a'], 0),
                    (incoming['b'], 0),
                ],
            )
        ]


def test_bloq_stop_flattening():
    bloq = BloqWithTensorsAndDecomp()
    u2 = bloq_to_dense(bloq, full_flatten=True)
    assert bloq.called_build_composite_bloq
    assert not bloq.called_my_tensors

    bloq = BloqWithTensorsAndDecomp()
    u1 = bloq_to_dense(bloq, full_flatten=False)
    assert not bloq.called_build_composite_bloq
    assert bloq.called_my_tensors

    np.testing.assert_allclose(u1, u2)
