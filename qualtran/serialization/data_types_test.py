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

from qualtran.serialization.data_types import data_type_from_proto, data_type_to_proto
import pytest

import sympy

from typing import Union

from qualtran import QDType, QBit, QAny, QInt, QIntOnesComp, QUInt, BoundedQUInt, QFxp

def round_trip(data: QDType):
    serialized_qbit = data_type_to_proto(data)
    original_qbit = data_type_from_proto(serialized_qbit)
    
    assert data.num_qubits == original_qbit.num_qubits
    if isinstance(serialized_qbit, BoundedQUInt):        
        assert data.iteration_length == original_qbit.iteration_length
    elif isinstance(serialized_qbit, QFxp):
        assert data.num_frac == original_qbit.num_frac
        assert data.signed == original_qbit.signed

def test_qbit():
    round_trip(QBit())

@pytest.mark.parametrize("num_qbits",
                          [10, 1000, sympy.Symbol("a") * sympy.Symbol("b")])
def test_qint_succeeds(num_qbits: Union[int, sympy.Expr]):
    round_trip(QInt(num_qbits))

@pytest.mark.parametrize("num_qbits",
                          [10, 1000, sympy.Symbol("a") * sympy.Symbol("b")])
def test_basic_data_types(num_qbits:int):
    round_trip(QAny(num_qbits))
    round_trip(QIntOnesComp(num_qbits))
    round_trip(QUInt(num_qbits))
    
@pytest.mark.parametrize("num_qbits, iteration_length",[
    (10, 1), (5, 10), (1000, 1000),
    (sympy.Symbol("a") * sympy.Symbol("b"), 10), 
    (sympy.Symbol("a"),  sympy.Symbol("b"))
   ])
def test_bounded_q_u_int(num_qbits: int, iteration_length):
    round_trip(BoundedQUInt(num_qbits, iteration_length))
    
@pytest.mark.parametrize("num_qbits, num_frac, signed", [
    (10, 5, True), (10, 5, False), (5,5,False), 
    (sympy.Symbol("a"),  sympy.Symbol("a"), False)
    ])
def test_qfxp(num_qbits:int, num_frac:int, signed:bool):
    round_trip(QFxp(num_qbits, num_frac, signed))
        
