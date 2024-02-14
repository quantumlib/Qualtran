import dataclasses
import inspect
import sympy

from typing import Any, Callable, Dict, List, Optional, Union
import attrs

from qualtran import QDType, QBit, QAny, QInt, QIntOnesComp, QUInt, BoundedQUInt, QFxp
from qualtran.protos import data_types_pb2

def data_type_to_proto(data: QDType) -> data_types_pb2.QDType:
    
    if isinstance(data.bitsize, int):
        dtype = data_types_pb2.QDType(bitsize_int=data.bitsize)
    elif isinstance(data.bitsize, sympy.Expr):
        dtype = data_types_pb2.QDType(bitsize_expr=str(data.bitsize))
    else:
        raise TypeError("Bitsize must either be an integer or a sympy expression")
        
    if isinstance(data, QBit):
        dtype.qbit.SetInParent()
    elif isinstance(data, QAny):
        dtype.qany.SetInParent()
    elif isinstance(data, QInt):
        dtype.qint.SetInParent()
    elif isinstance(data, QIntOnesComp):
        dtype.q_ones_comp.SetInParent()
    elif isinstance(data, QUInt):
        dtype.quint.SetInParent()
    elif isinstance(data, BoundedQUInt):
        dtype.bounded_q_int.SetInParent()
        dtype.bounded_q_int.iteration_length = data.iteration_length
    elif isinstance(data, QFxp):
        dtype.q_fxp.SetInParent()
        dtype.num_frac = data.num_frac
        dtype.signed = data.signed
    else:
        raise TypeError("data must be of one of the following subtypes: QBit, QAny, QInt, QIntOnesComp, QUInt, BoundedQUInt, QFixedPoint")
    return dtype    

def data_type_from_proto(data: data_types_pb2.QDType):
    if isinstance(data.qbit, data_types_pb2.QBit):
        return QBit()
    elif isinstance(data.qany, data_types_pb2.QAny):
        return QAny(data.bitsize_int)
    elif isinstance(data.qint, data_types_pb2.QInt):    
        return QInt(data.bitsize_int)    
    elif isinstance(data.q_ones_comp, data_types_pb2.QIntOnesComp):    
        return QIntOnesComp(data.bitsize_int)
    elif isinstance(data.quint, data_types_pb2.QUInt):    
        return QUInt(data.bitsize_int)  
    elif isinstance(data.bounded_q_int, data_types_pb2.BoundedQInt):    
        return BoundedQUInt(data.bitsize_int, data.bounded_q_int.iteration_length) 
    elif isinstance(data.q_fxp, data_types_pb2.QFixedPoint):
        return QFxp(data.bitsize_int, data.signed)
a_qbit = BoundedQUInt(8, 2)

serialized_qbit = data_type_to_proto(a_qbit)
original_qbit = data_type_from_proto(serialized_qbit)


