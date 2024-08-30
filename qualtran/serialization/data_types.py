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
from qualtran import BQUInt, QAny, QBit, QDType, QFxp, QInt, QIntOnesComp, QMontgomeryUInt, QUInt
from qualtran.protos import data_types_pb2
from qualtran.serialization.args import int_or_sympy_from_proto, int_or_sympy_to_proto


def data_type_to_proto(data: QDType) -> data_types_pb2.QDataType:
    if isinstance(data, QBit):
        return data_types_pb2.QDataType(qbit=data_types_pb2.QBit())

    bitsize = int_or_sympy_to_proto(data.bitsize)  # type: ignore[attr-defined]
    if isinstance(data, QAny):
        return data_types_pb2.QDataType(qany=data_types_pb2.QAny(bitsize=bitsize))
    elif isinstance(data, QInt):
        return data_types_pb2.QDataType(qint=data_types_pb2.QInt(bitsize=bitsize))
    elif isinstance(data, QIntOnesComp):
        return data_types_pb2.QDataType(qint_ones_comp=data_types_pb2.QIntOnesComp(bitsize=bitsize))
    elif isinstance(data, QUInt):
        return data_types_pb2.QDataType(quint=data_types_pb2.QUInt(bitsize=bitsize))
    elif isinstance(data, QMontgomeryUInt):
        return data_types_pb2.QDataType(
            qmontgomery_uint=data_types_pb2.QMontgomeryUInt(bitsize=bitsize)
        )
    elif isinstance(data, BQUInt):
        iteration_length = int_or_sympy_to_proto(data.iteration_length)
        return data_types_pb2.QDataType(
            bquint=data_types_pb2.BQUInt(bitsize=bitsize, iteration_length=iteration_length)
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
            "QAny, QInt, QIntOnesComp, QUInt, BQUInt, "
            "QFxp, QMontgomeryUInt"
        )


def data_type_from_proto(serialized: data_types_pb2.QDataType) -> QDType:
    if serialized.HasField('qbit'):
        return QBit()

    if serialized.HasField('qany'):
        bitsize = int_or_sympy_from_proto(serialized.qany.bitsize)
        return QAny(bitsize=bitsize)
    elif serialized.HasField('qint'):
        bitsize = int_or_sympy_from_proto(serialized.qint.bitsize)
        return QInt(bitsize=bitsize)
    elif serialized.HasField('qint_ones_comp'):
        bitsize = int_or_sympy_from_proto(serialized.qint_ones_comp.bitsize)
        return QIntOnesComp(bitsize=bitsize)
    elif serialized.HasField('quint'):
        bitsize = int_or_sympy_from_proto(serialized.quint.bitsize)
        return QUInt(bitsize=bitsize)
    elif serialized.HasField('qmontgomery_uint'):
        bitsize = int_or_sympy_from_proto(serialized.qmontgomery_uint.bitsize)
        return QMontgomeryUInt(bitsize=bitsize)
    elif serialized.HasField('bquint'):
        bitsize = int_or_sympy_from_proto(serialized.bquint.bitsize)
        iteration_length = int_or_sympy_from_proto(serialized.bquint.iteration_length)
        return BQUInt(bitsize=bitsize, iteration_length=iteration_length)
    elif serialized.HasField('qfxp'):
        bitsize = int_or_sympy_from_proto(serialized.qfxp.bitsize)
        num_frac = int_or_sympy_from_proto(serialized.qfxp.num_frac)
        return QFxp(bitsize=bitsize, num_frac=num_frac, signed=serialized.qfxp.signed)
    else:
        raise TypeError(
            f"Data type {type(serialized)} is not recognized."
            " It must be of one of the following subtypes: QBit, "
            "QAny, QInt, QIntOnesComp, QUInt, BQUInt, "
            "QFxp, QMontgomeryUInt"
        )
