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

"""Utilities for resolving symbolic expressions in FLASQ cost formulas."""
from functools import lru_cache
from typing import Any, Union

import numpy as np
import pandas as pd
import sympy
from frozendict import frozendict  # type: ignore[import-untyped]


@lru_cache(maxsize=None)
def _get_cached_lambdified_evaluator(
    expression: sympy.Expr, sorted_symbols: tuple[sympy.Symbol, ...]
):
    """
    Creates and caches a lambdified version of an expression for a given set of symbols.

    Args:
        expression: The sympy expression to lambdify.
        sorted_symbols: A tuple of sympy.Symbol objects, sorted to ensure canonical cache keys.
            These are the symbols that will be arguments to the lambdified function.

    Returns:
        A callable function that evaluates the expression.
    """
    return sympy.lambdify(sorted_symbols, expression, modules="numpy")


@lru_cache(maxsize=None)
def substitute_until_fixed_point(
    expression: Union[sympy.Expr, int, float],
    resolver: frozendict[Union[sympy.Symbol, str], Any],
    try_make_number: bool = True,
) -> Union[sympy.Expr, int, float]:
    """Iteratively substitutes symbols in a sympy expression until a fixed point.

    Args:
        expression: The sympy expression to process.
        resolver: A frozendict mapping sympy Symbols (or their string names)
            to the values they should be substituted with. These values can
            be numbers or other sympy expressions.
        try_make_number: If True, attempts to convert the final expression to
            a Python int or float if possible.

    Returns:
        The resolved expression, potentially converted to an int or float, or the
        simplified sympy expression if conversion to a number is not possible
        or not requested.
    """
    # This function is now cached with lru_cache.
    # If the input is already a concrete number, no substitution is needed.
    if isinstance(expression, (int, float, np.number)):
        return expression

    # Fast path for direct numeric substitution using lambdify
    if (
        isinstance(expression, sympy.Expr)
        and expression.free_symbols
        and try_make_number
    ):
        resolver_keys_as_symbols = {
            (sympy.Symbol(k) if isinstance(k, str) else k) for k in resolver.keys()
        }
        relevant_symbols = expression.free_symbols.intersection(
            resolver_keys_as_symbols
        )

        if (
            relevant_symbols == expression.free_symbols
        ):  # All free symbols are in resolver
            # Check if all relevant resolver values are numeric
            numeric_values = []
            all_numeric = True
            # Sort symbols to ensure canonical order for lambdify and argument passing
            sorted_relevant_symbols = tuple(sorted(list(relevant_symbols), key=str))

            for s in sorted_relevant_symbols:
                val = resolver.get(
                    s, resolver.get(str(s))
                )  # Check for Symbol then str key
                if isinstance(val, (int, float)):
                    numeric_values.append(val)
                else:
                    all_numeric = False
                    break

            if all_numeric:
                try:
                    evaluator = _get_cached_lambdified_evaluator(
                        expression, sorted_relevant_symbols
                    )
                    result = evaluator(*numeric_values)
                    return result
                except (KeyError, ValueError, TypeError):
                    # Lambdify might fail on special symbols like zoo.
                    # Fall back to slow path.
                    pass

    if try_make_number:
        if expression.is_number:
            if expression.is_integer:
                return int(expression)
            else:
                return float(expression)

    try:
        old = expression
        # Perform at least one substitution initially
        new = expression.subs(resolver, simultaneous=True)

        if isinstance(new, (int, float)):
            return new

        if try_make_number:
            if new.is_number:
                if new.is_integer:
                    return int(new)
                else:
                    return float(new)

        while old != new:
            old = new
            new = old.subs(resolver, simultaneous=True)
    except (ValueError, TypeError):
        return expression

    if try_make_number:
        if new.is_number:
            if new.is_integer:
                return int(new)
            else:
                return float(new)

    try:
        simplified_expr = sympy.simplify(new)
        simplified_expr_log = sympy.logcombine(simplified_expr, force=True)
        expanded_expr = sympy.expand_log(simplified_expr_log, force=True)
        return expanded_expr
    except (ValueError, TypeError):
        return new


def convert_sympy_exprs_in_df(df: pd.DataFrame) -> pd.DataFrame:
    """Converts fully evaluated SymPy numbers in a DataFrame to Python types.

    Scans the DataFrame columns and converts any SymPy expressions that are
    pure numbers into standard Python `int` or `float` objects.

    Args:
        df: The pandas DataFrame to process.

    Returns:
        The processed DataFrame with SymPy numbers converted.
    """
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, sympy.Expr)).any():
            df[col] = df[col].apply(
                lambda x: (
                    float(x)
                    if isinstance(x, sympy.Expr) and x.is_number and not x.is_integer
                    else int(x) if isinstance(x, sympy.Expr) and x.is_integer else x
                )
            )
    return df


