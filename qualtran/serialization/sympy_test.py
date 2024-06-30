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

import pytest
import sympy
from sympy.codegen.cfunctions import log2

from qualtran.serialization.sympy import sympy_expr_from_proto, sympy_expr_to_proto

x = sympy.Symbol('x', positive=True)
a, b, c = sympy.symbols("a b c")

# These should return a `sympy_pb2.Parameter` proto object
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
    sympy.Float(0.25),
    sympy.Rational("1/2"),
    sympy.Rational('1/10'),
]
sympy_exprs_to_test = [
    5 * a + sympy.sqrt(a),
    log2(a),
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
def test_parameter(expr: sympy.Expr):
    """
    Test types of expressions including fraction, complex, and constant symbol (such as pi).
    """
    serialized = sympy_expr_to_proto(expr)
    expr_clone = sympy_expr_from_proto(serialized)
    assert expr == expr_clone


def test_float_fraction():
    """
    Test that floats and fractions can be properly combined and serialzed.
    """
    float_const = sympy.parse_expr("1.4")
    fraction = sympy.parse_expr("1/2")
    expr = float_const * fraction

    serialized = sympy_expr_to_proto(expr)
    expr_clone = sympy_expr_from_proto(serialized)
    assert abs(expr - expr_clone) < 0.001
