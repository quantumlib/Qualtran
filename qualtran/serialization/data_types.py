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

import sympy

from qualtran import BoundedQUInt, QAny, QBit, QDType, QFxp, QInt, QIntOnesComp, QUInt
from qualtran.protos import data_types_pb2
from qualtran.serialization.args import int_or_sympy_from_proto, int_or_sympy_to_proto


def int_or_sympy_from_qdt_proto(val: data_types_pb2.QDataType) -> Union[int, sympy.Expr]:
    if val.HasField('qbit'):
        raise ValueError('QBit case should have been handled already.')
    if val.HasField('qany'):
        return int_or_sympy_from_proto(val.qany.bitsize)
    if val.HasField('qint'):
        return int_or_sympy_from_proto(val.qint.bitsize)
    if val.HasField('qintoc'):
        return int_or_sympy_from_proto(val.qintoc.bitsize)
    if val.HasField('quint'):
        return int_or_sympy_from_proto(val.quint.bitsize)
    if val.HasField('bquint'):
        return int_or_sympy_from_proto(val.bquint.bitsize)
    if val.HasField('qfxp'):
        return int_or_sympy_from_proto(val.qfxp.bitsize)
    raise ValueError(f"Cannot deserialize {val=}")


def data_type_to_proto(data: QDType) -> data_types_pb2.QDataType:
    if isinstance(data, QBit):
        return data_types_pb2.QDataType(qbit=data_types_pb2.QBit())

    bitsize = int_or_sympy_to_proto(data.bitsize)
    if isinstance(data, QAny):
        return data_types_pb2.QDataType(qany=data_types_pb2.QAny(bitsize=bitsize))
    elif isinstance(data, QInt):
        return data_types_pb2.QDataType(qint=data_types_pb2.QInt(bitsize=bitsize))
    elif isinstance(data, QIntOnesComp):
        return data_types_pb2.QDataType(qintoc=data_types_pb2.QIntOnesComp(bitsize=bitsize))
    elif isinstance(data, QUInt):
        return data_types_pb2.QDataType(quint=data_types_pb2.QUInt(bitsize=bitsize))
    elif isinstance(data, BoundedQUInt):
        iteration_length = int_or_sympy_to_proto(data.iteration_length)
        return data_types_pb2.QDataType(
            bquint=data_types_pb2.BoundedQUInt(bitsize=bitsize, iteration_length=iteration_length)
        )
    elif isinstance(data, QFxp):
        num_frac = int_or_sympy_to_proto(data.num_frac)
        return data_types_pb2.QDataType(
            qfxp=data_types_pb2.QFxp(bitsize=bitsize, num_frac=num_frac, signed=data.signed)
        )
    else:
        raise TypeError(
            f"Data type {type(data)} is not recognized."
            " It must be of one of the following subtypes: QBit, "
            "QAny, QInt, QIntOnesComp, QUInt, BoundedQUInt, "
            "QFixedPoint"
        )


def data_type_from_proto(serialized: data_types_pb2.QDataType) -> QDType:
    if serialized.HasField('qbit'):
        return QBit()

    bitsize = int_or_sympy_from_qdt_proto(serialized)
    if serialized.HasField('qany'):
        return QAny(bitsize=bitsize)
    elif serialized.HasField('qint'):
        return QInt(bitsize=bitsize)
    elif serialized.HasField('qint_ones_comp'):
        return QIntOnesComp(bitsize=bitsize)
    elif serialized.HasField('quint'):
        return QUInt(bitsize=bitsize)
    elif serialized.HasField('bounded_quint'):
        iteration_length = int_or_sympy_from_proto(serialized.bquint.iteration_length)
        return BoundedQUInt(bitsize=bitsize, iteration_length=iteration_length)
    elif serialized.HasField('qfxp'):
        num_frac = int_or_sympy_from_proto(serialized.qfxp.num_frac)
        return QFxp(bitsize=bitsize, num_frac=num_frac, signed=serialized.qfxp.signed)
    else:
        raise TypeError(
            f"Data type {type(serialized)} is not recognized."
            " It must be of one of the following subtypes: QBit, "
            "QAny, QInt, QIntOnesComp, QUInt, BoundedQUInt, "
            "QFixedPoint"
        )
