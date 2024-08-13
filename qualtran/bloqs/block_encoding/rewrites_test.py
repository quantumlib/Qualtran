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

from qualtran.bloqs.basic_gates import Hadamard, XGate
from qualtran.bloqs.block_encoding import LinearCombination, Phase, Product, TensorProduct, Unitary
from qualtran.bloqs.block_encoding.rewrites import collect_like_terms


def test_collect_like_terms_atomic():
    assert collect_like_terms(Unitary(XGate())) == Unitary(XGate())
    assert collect_like_terms(
        Product((Phase(Unitary(XGate()), phi=1, eps=1), Unitary(XGate())))
    ) == Product((Phase(Unitary(XGate()), phi=1, eps=1), Unitary(XGate())))


def test_collect_like_terms_nested():
    assert collect_like_terms(
        TensorProduct(
            (
                LinearCombination(
                    (
                        LinearCombination(
                            (Unitary(XGate()), Unitary(Hadamard())), (2.0, 2.0), lambd_bits=1
                        ),
                        LinearCombination(
                            (Unitary(XGate()), Phase(Unitary(Hadamard()), phi=1, eps=1)),
                            (1.0, -1.0),
                            lambd_bits=1,
                        ),
                    ),
                    (1.0, -1.0),
                    lambd_bits=1,
                ),
                Unitary(XGate()),
            )
        )
    ) == TensorProduct(
        (
            LinearCombination(
                (Unitary(XGate()), Unitary(Hadamard()), Phase(Unitary(Hadamard()), phi=1, eps=1)),
                (1.0, 2.0, 1.0),
                lambd_bits=2,
            ),
            Unitary(XGate()),
        )
    )
