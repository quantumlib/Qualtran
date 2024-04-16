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
from qualtran.protos import sympy_pb2
from typing import Union, Any
def _get_sympy_function_type(expr: sympy.Expr) -> int:
    """
    Helper function for serializing a sympy function.

    This method converts a sympy function to its sympy_pb2.Function enum representation.
    """
    if isinstance(expr, sympy.core.mul.Mul):
        return sympy_pb2.Function.Mul
    elif isinstance(expr, sympy.core.add.Add):
        return sympy_pb2.Function.Add
    elif isinstance(expr, sympy.core.power.Pow):
        return sympy_pb2.Function.Pow
    elif isinstance(expr, sympy.core.Mod):
        return sympy_pb2.Function.Mod
    elif isinstance(expr, sympy.functions.elementary.exponential.log):
        return sympy_pb2.Function.Log
    elif isinstance(expr, sympy.functions.elementary.integers.floor):
        return sympy_pb2.Function.Floor
    elif isinstance(expr, sympy.functions.elementary.integers.ceiling):
        return sympy_pb2.Function.Ceiling
    elif isinstance(expr, sympy.functions.elementary.miscellaneous.Max):
        return sympy_pb2.Function.Max
    elif isinstance(expr, sympy.functions.elementary.miscellaneous.Min):
        return sympy_pb2.Function.Min
    elif isinstance(expr, sympy.functions.elementary.trigonometric.sin):
        return sympy_pb2.Function.Sin
    elif isinstance(expr, sympy.functions.elementary.trigonometric.cos):
        return sympy_pb2.Function.Cos
    elif isinstance(expr, sympy.functions.elementary.trigonometric.tan):
        return sympy_pb2.Function.Tan
    else:
        return sympy_pb2.Function.NONE

def _get_sympy_function_from_enum(enum: int)->Any:
    """
    Helper function for sympy function deserialization.

    Sympy functions are represented as a sympy_pb2.Function enum.  This mehtod converts this int enum
    back into the sympy function.
    """
    enum_to_sympy = {
        sympy_pb2.Function.Mul: sympy.core.mul.Mul,
        sympy_pb2.Function.Add: sympy.core.add.Add,
        sympy_pb2.Function.Pow: sympy.core.power.Pow,
        sympy_pb2.Function.Mod: sympy.core.Mod,
        sympy_pb2.Function.Log: sympy.functions.elementary.exponential.log,
        sympy_pb2.Function.Floor: sympy.functions.elementary.integers.floor,
        sympy_pb2.Function.Ceiling: sympy.functions.elementary.integers.ceiling,
        sympy_pb2.Function.Max: sympy.functions.elementary.miscellaneous.Max,
        sympy_pb2.Function.Min: sympy.functions.elementary.miscellaneous.Min,
        sympy_pb2.Function.Sin: sympy.functions.elementary.trigonometric.sin,
        sympy_pb2.Function.Cos: sympy.functions.elementary.trigonometric.cos,
        sympy_pb2.Function.Tan: sympy.functions.elementary.trigonometric.tan,
        sympy_pb2.Function.NONE: None,
    }

    return enum_to_sympy[enum]

def _get_sympy_const_from_enum(enum: int)->Any:
    """Helper function for deserializing a sympy symbolic constant.

    Symbolic constants are serialzed as an enum of type sympy_pb2.ConstSymbol. This method converts the
    enum representation back to its original sympy representation.
    """
    enum_to_sympy = {
        sympy_pb2.ConstSymbol.Pi: sympy.pi,
        sympy_pb2.ConstSymbol.E: sympy.E,
        sympy_pb2.ConstSymbol.EulerGamma: sympy.EulerGamma,
        sympy_pb2.ConstSymbol.Infinity: sympy.core.numbers.Infinity(),
        sympy_pb2.ConstSymbol.ImaginaryUnit: sympy.core.numbers.ImaginaryUnit()
    }
    return enum_to_sympy[enum]

def _get_const_symbolic_operand(expr: sympy.Expr)->sympy_pb2.Parameter:
    """
    Helper function for serializing a symbolic constant from a sympy expression.

    Currently supported symbolic constants are: pi, natural exponent, EulerGamma, sqrt(-1), and infinity.
    """
    if expr == sympy.pi:
        return sympy_pb2.Parameter(const_symbol=sympy_pb2.ConstSymbol.Pi)
    if expr == sympy.E:
        return sympy_pb2.Parameter(const_symbol=sympy_pb2.ConstSymbol.E)
    if expr == sympy.EulerGamma:
        return sympy_pb2.Parameter(const_symbol=sympy_pb2.ConstSymbol.EulerGamma)
    if isinstance(expr,sympy.core.numbers.Infinity):
        return sympy_pb2.Parameter(const_symbol=sympy_pb2.ConstSymbol.Infinity)
    if isinstance(expr, sympy.core.numbers.ImaginaryUnit):
        return sympy_pb2.Parameter(const_symbol=sympy_pb2.ConstSymbol.ImaginaryUnit)
    else:
        raise NotImplementedError(f"Sympy expression {str(expr)} cannot be serialized.")

def _get_sympy_operand(expr: Union[sympy.Expr, int, float])->sympy_pb2.Parameter:
    """
    Converts the input to a serializable sympy_pb2 Parameter.

    A parameter represents a single, irreducable numeric entity such as a variable, constant, or an explicit number.
    """

    # Expression is a single, symbolic variable.
    if isinstance(expr, sympy.core.symbol.Symbol):
        return sympy_pb2.Parameter(symbol=str(expr))

    # Expression is an integer
    if issubclass(expr.__class__, sympy.core.numbers.Integer):
        result = expr.numerator
        if not isinstance(result, int):
            raise NotImplementedError(f"Sympy expression {str(expr)} cannot be serialized.")
        return sympy_pb2.Parameter(const_int=result)

    # Expression cannot be broken down further, but is a constant.
    if issubclass(expr.__class__, sympy.core.numbers.Number):
        if isinstance(expr, sympy.core.numbers.Float):
            return sympy_pb2.Parameter(const_float=float(expr))
        if isinstance(expr, sympy.core.numbers.Rational):
            numerator = _get_sympy_operand(expr.numerator)
            denominator = _get_sympy_operand(expr.denominator)
            fraction = sympy_pb2.Fraction(numerator=numerator, denominator=denominator)
            return sympy_pb2.Parameter(const_rat=fraction)
        else:
            raise NotImplementedError(f"Sympy expression {str(expr)} cannot be serialized.")
    if type(expr) == int:
        return sympy_pb2.Parameter(const_int=expr)
    if type(expr) == float:
        return sympy_pb2.Parameter(const_float=expr)
    else:
        return _get_const_symbolic_operand(expr)

def sympy_expr_to_proto(expr: sympy.Expr)-> sympy_pb2.Term:
    """Serializes a sympy expression."""

    function = _get_sympy_function_type(expr)
    operands = []
    if function == sympy_pb2.Function.NONE:
        parameter = _get_sympy_operand(expr)
        operands.append(sympy_pb2.Operand(parameter=parameter))

    else:
        for term in expr.args:
            inner_term = sympy_expr_to_proto(term)

            operands.append(sympy_pb2.Operand(term=inner_term))

    return sympy_pb2.Term(function=function, operands=operands)

def _get_parameter(serialized_input: Union[sympy_pb2.Operand, sympy_pb2.Parameter])->Any:
    """
    Deserializes a parameter.

    Deserializes a parameter or operand into either its sympy representation or its
    python primative numeric representation.
    """
    if isinstance(serialized_input, sympy_pb2.Operand):
        serialized_parameter = serialized_input.parameter
    else:
        serialized_parameter = serialized_input

    parameter_type = serialized_parameter.WhichOneof("parameter")
    if parameter_type == "symbol":
        deserialized_parameter = sympy.symbols(serialized_parameter.symbol)
    elif parameter_type == "const_int":
        deserialized_parameter = serialized_parameter.const_int
    elif parameter_type == "const_rat":
        fraction = serialized_parameter.const_rat
        numerator = _get_parameter(fraction.numerator)
        denominator = _get_parameter(fraction.denominator)
        deserialized_parameter = sympy.Rational(numerator, denominator)
    elif parameter_type == "const_float":
        deserialized_parameter = serialized_parameter.const_float
    elif parameter_type == "const_symbol":
        deserialized_parameter = _get_sympy_const_from_enum(serialized_parameter.const_symbol)
    else:
        raise TypeError(f"Type is not supported for {serialized_input}")

    return deserialized_parameter


def sympy_expr_from_proto(term: sympy_pb2.Term)->Any:
    """Deserialize a sympy expression."""

    function = _get_sympy_function_from_enum(term.function)
    parameters = []
    for operand in term.operands:
        if operand.HasField("term"):
            parameters.append(sympy_expr_from_proto(operand.term))
        else:
            parameter = _get_parameter(operand)
            parameters.append(parameter)

    if function:
        return function(*parameters)
    elif len(parameters) == 1:
        return parameters[0]
    else:
        raise NotImplementedError(f"{term.function} has not been fully implimented.")