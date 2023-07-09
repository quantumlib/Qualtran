import numpy as np
import pytest
import sympy

from qualtran.serialization import args_to_proto


@pytest.mark.parametrize(
    'arg',
    [
        1,
        2.0,
        'hello world',
        sympy.Symbol('a') * sympy.Symbol('b') + sympy.Symbol('c') / 10,
        np.array([*range(100)], dtype=np.complex128).reshape((10, 10)),
    ],
)
def test_arg_to_proto_round_trip(arg):
    proto = args_to_proto.arg_to_proto(name='custom_name', val=arg)
    arg_dict = args_to_proto.arg_from_proto(proto)
    if isinstance(arg, np.ndarray):
        arr = arg_dict['custom_name']
        assert arr.shape == arg.shape
        assert arr.dtype == arg.dtype
        assert np.allclose(arr, arg)
    else:
        assert arg_dict['custom_name'] == arg
