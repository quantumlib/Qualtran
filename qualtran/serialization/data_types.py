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

from qualtran import QDType, QBit, QAny, QInt, QIntOnesComp, QUInt, BoundedQUInt, QFxp
from qualtran.protos import data_types_pb2
from qualtran.serialization.args import int_or_sympy_from_proto,int_or_sympy_to_proto


SerializedDataTypes = Union[data_types_pb2.QBit, data_types_pb2.QAny,
                            data_types_pb2.QInt, data_types_pb2.QIntOnesComp,
                            data_types_pb2.QUInt, data_types_pb2.BoundedQUInt,
                            data_types_pb2.QFxp]

def data_type_to_proto(data: QDType) -> SerializedDataTypes:
    if isinstance(data, QBit):
        return data_types_pb2.QBit()
    
    bitsize = int_or_sympy_to_proto(data.bitsize)
    if isinstance(data, QAny):
        return data_types_pb2.QAny(bitsize=bitsize)
    elif isinstance(data, QInt):
        return data_types_pb2.QInt(bitsize=bitsize)
    elif isinstance(data, QIntOnesComp):
        return data_types_pb2.QIntOnesComp(bitsize=bitsize)
    elif isinstance(data, QUInt):
        return data_types_pb2.QUInt(bitsize=bitsize)
    elif isinstance(data, BoundedQUInt):
        iteration_length = int_or_sympy_to_proto(data.iteration_length)
        return data_types_pb2.BoundedQUInt(bitsize=bitsize,
                                           iteration_length=iteration_length)
    elif isinstance(data, QFxp):
        num_frac = int_or_sympy_to_proto(data.num_frac)
        return data_types_pb2.QFxp(bitsize=bitsize,
                                   num_frac=num_frac,
                                   signed=data.signed)
    else:
        raise TypeError("Data must be of one of the following subtypes: " \
                        "QBit, QAny, QInt, QIntOnesComp, QUInt, BoundedQUInt,"\
                            " QFixedPoint")

def data_type_from_proto(serialized):
    if isinstance(serialized, data_types_pb2.QBit):
        return QBit()
    
    bitsize = int_or_sympy_from_proto(serialized.bitsize)
    if isinstance(serialized, data_types_pb2.QAny):
        return QAny(bitsize=bitsize)
    elif isinstance(serialized, data_types_pb2.QInt):
        return QInt(bitsize=bitsize)
    elif isinstance(serialized, data_types_pb2.QIntOnesComp):
        return QIntOnesComp(bitsize=bitsize)
    elif isinstance(serialized, data_types_pb2.QUInt):
        return QUInt(bitsize=bitsize)
    elif isinstance(serialized, data_types_pb2.QIntOnesComp):
        return QIntOnesComp(bitsize=bitsize)
    elif isinstance(serialized, data_types_pb2.QUInt):
        return QUInt(bitsize=bitsize)
    elif isinstance(serialized, data_types_pb2.BoundedQUInt):
        iteration_length = int_or_sympy_from_proto(serialized.iteration_length)
        return BoundedQUInt(bitsize=bitsize,
            iteration_length=iteration_length)
    elif isinstance(serialized, data_types_pb2.QFxp):
        num_frac = int_or_sympy_from_proto(serialized.num_frac)
        return QFxp(bitsize=bitsize, num_frac=num_frac,
                    signed=serialized.signed)
    else:
        raise TypeError("Data type {} is not recognized."\
                        " It must be of one of the following subtypes: QBit, "\
                            "QAny, QInt, QIntOnesComp, QUInt, BoundedQUInt, "\
                                "QFixedPoint".format(str(type(serialized))))
        
    