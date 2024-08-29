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

from typing import Union

import pytest
import sympy

from qualtran import BQUInt, QAny, QBit, QDType, QFxp, QInt, QIntOnesComp, QMontgomeryUInt, QUInt
from qualtran._infra.data_types import QMontgomeryUInt
from qualtran.serialization.data_types import data_type_from_proto, data_type_to_proto


def round_trip_qdt(data: QDType):
    serialized_qbit = data_type_to_proto(data)
    original_qbit = data_type_from_proto(serialized_qbit)

    assert data == original_qbit


def test_qbit():
    round_trip_qdt(QBit())


@pytest.mark.parametrize("num_qbits", [10, 1000, sympy.Symbol("a") * sympy.Symbol("b")])
def test_basic_data_types(num_qbits: Union[int, sympy.Expr]):
    round_trip_qdt(QInt(num_qbits))
    round_trip_qdt(QAny(num_qbits))
    round_trip_qdt(QIntOnesComp(num_qbits))
    round_trip_qdt(QMontgomeryUInt(num_qbits))
    round_trip_qdt(QUInt(num_qbits))


@pytest.mark.parametrize(
    "num_qbits, iteration_length",
    [
        (10, 1),
        (5, 10),
        (1000, 1000),
        (sympy.Symbol("a") * sympy.Symbol("b"), 10),
        (sympy.Symbol("a"), sympy.Symbol("b")),
    ],
)
def test_bounded_quint(num_qbits: int, iteration_length):
    round_trip_qdt(BQUInt(num_qbits, iteration_length))


@pytest.mark.parametrize(
    "num_qbits, num_frac, signed",
    [(10, 5, True), (10, 5, False), (5, 5, False), (sympy.Symbol("a"), sympy.Symbol("a"), False)],
)
def test_qfxp(num_qbits: int, num_frac: int, signed: bool):
    round_trip_qdt(QFxp(num_qbits, num_frac, signed))
