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

import sympy
import pytest
from qualtran.serialization.bloq import arg_to_proto, arg_from_proto

@pytest.mark.parametrize(
    'expr',
    [
        (sympy.parse_expr("5")+sympy.symbols("x")+sympy.parse_expr("1/2")+sympy.pi+sympy.parse_expr("2j")),
        (sympy.parse_expr("(-b + sqrt(-4*a*c + b**2))/(2*a)"))
    ],
)
def parameter_test(expr: sympy.Expr):
    """
    Test types of expressions including fraction, complex, and constant symbol (such as pi).
    """

    serialized = arg_to_proto(name="test", val=expr)
    expr_clone = arg_from_proto(serialized)['test']
    assert expr == expr_clone

def float_fraction_test():
    """
    Test that floats and fractions can be properly combined and serialzed.
    """
    float = sympy.parse_expr("1.4")
    fraction = sympy.parse_expr("1/2")
    expr = float*fraction

    serialized = arg_to_proto(name="test", val=expr)
    expr_clone = arg_from_proto(serialized)['test']
    assert abs(expr - expr_clone) < .001

