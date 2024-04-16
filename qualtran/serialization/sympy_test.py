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

import pytest
import sympy

from qualtran.serialization.bloq import arg_from_proto, arg_to_proto

x = sympy.Symbol('x', positive=True)
a, b, c = sympy.symbols("a b c")

# These should return a `sympy_pb2.Parameter` proto object?
sympy_parameters_to_test = [
    # Only symbols
    sympy.Symbol('x'),
    sympy.Symbol('N'),
    sympy.Symbol('E'),
    # Sympy constants
    sympy.pi,
    sympy.oo,  # infinity
    sympy.E,
    sympy.I,
    sympy.EulerGamma,
    # Integers, Floats, Rationals
    sympy.Integer(5),
    sympy.Float(0.1),
    sympy.Rational("1/2"),
    sympy.Rational('1/10'),
]
sympy_exprs_to_test = [
    5 * x + sympy.sqrt(a),
    # Complex Fractions
    sympy.Rational("1/10") * sympy.I + 5,
    # Basic operations
    a / b + c - 5,
    # Trig operations
    sympy.sin(a) + sympy.cos(b) + sympy.tan(c),
    # Integer operations
    sympy.floor(5.43) + sympy.ceiling(a),
    sympy.Max(a, b),
    # Nested Operations
    a ** (b * c**2),
]


@pytest.mark.parametrize('expr', sympy_parameters_to_test + sympy_exprs_to_test)
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
    float_const = sympy.parse_expr("1.4")
    fraction = sympy.parse_expr("1/2")
    expr = float_const * fraction

    serialized = arg_to_proto(name="test", val=expr)
    expr_clone = arg_from_proto(serialized)['test']
    assert abs(expr - expr_clone) < 0.001
