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
from frozendict import frozendict  # type: ignore[import-untyped]

from qualtran.surface_code.flasq.utils import (
    substitute_until_fixed_point,
)


def test_basic_substitution():
    x = sympy.Symbol("x")
    expr = x + 1
    resolver = frozendict({x: 2})
    result = substitute_until_fixed_point(expr, resolver)
    assert result == 3
    assert isinstance(result, int)


def test_no_substitution_needed():
    x = sympy.Symbol("x")
    expr = x + 1
    resolver = frozendict({"y": 2})  # Symbol 'y' not in expression
    result = substitute_until_fixed_point(expr, resolver)
    assert result == x + 1
    assert isinstance(result, sympy.Expr)


def test_string_keys_in_resolver():
    x = sympy.Symbol("x")
    expr = x + 1
    resolver = frozendict({"x": 5})  # Use string key
    result = substitute_until_fixed_point(expr, resolver)
    assert result == 6
    assert isinstance(result, int)


def test_try_make_number_false():
    x = sympy.Symbol("x")
    expr = x + 1
    resolver = frozendict({x: 2})
    result = substitute_until_fixed_point(expr, resolver, try_make_number=False)
    assert result == sympy.Integer(3)
    assert isinstance(result, sympy.Expr)


def test_nested_substitution():
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    expr = x + y + z
    resolver = frozendict({x: y + 1, y: z * 2, z: 3})

    # Check that one pass of subs doesn't fully resolve
    assert not expr.subs(resolver, simultaneous=True).is_number

    # Check that substitute_until_fixed_point fully resolves
    final_result = substitute_until_fixed_point(expr, resolver)
    assert final_result == 16  # (3*2 + 1) + (3*2) + 3 = 7 + 6 + 3 = 16
    assert isinstance(final_result, int)


def test_substitute_with_symbolic_value():
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")
    expr = x + 1
    resolver = frozendict({x: y * z})
    result = substitute_until_fixed_point(expr, resolver)
    assert result == y * z + 1
    assert isinstance(result, sympy.Expr)


def test_substitute_int_input():
    """Test that providing an int returns the int."""
    expr = 5
    resolver = frozendict({"x": 10})
    result_number = substitute_until_fixed_point(expr, resolver, try_make_number=True)
    assert result_number == 5
    assert isinstance(result_number, int)
    result_nofloat = substitute_until_fixed_point(expr, resolver, try_make_number=False)
    assert result_nofloat == 5
    assert isinstance(result_nofloat, int)


def test_substitute_float_input():
    """Test that providing a float returns the float."""
    expr = 3.14
    resolver = frozendict({"x": 10})
    result = substitute_until_fixed_point(expr, resolver)
    assert result == 3.14
    assert isinstance(result, float)


def test_substitute_to_float():
    """Test substitution resulting in a non-integer number."""
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    expr = x / y
    resolver = frozendict({x: 5, y: 2})
    result = substitute_until_fixed_point(expr, resolver)
    assert result == 2.5
    assert isinstance(result, float)


def test_substitute_until_fixed_point_caching():
    """Tests the lru_cache functionality of substitute_until_fixed_point."""
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    expr1 = x + y
    resolver1 = frozendict({x: 1, y: 2})
    resolver2 = frozendict({x: 10, y: 20})

    substitute_until_fixed_point.cache_clear()
    assert substitute_until_fixed_point.cache_info().hits == 0
    assert substitute_until_fixed_point.cache_info().misses == 0
    assert substitute_until_fixed_point.cache_info().currsize == 0

    # First call - miss
    res1 = substitute_until_fixed_point(expr1, resolver1)
    assert res1 == 3
    assert substitute_until_fixed_point.cache_info().hits == 0
    assert substitute_until_fixed_point.cache_info().misses == 1
    assert substitute_until_fixed_point.cache_info().currsize == 1

    # Second call with same args - hit
    res2 = substitute_until_fixed_point(expr1, resolver1)
    assert res2 == 3
    assert substitute_until_fixed_point.cache_info().hits == 1
    assert substitute_until_fixed_point.cache_info().misses == 1
    assert substitute_until_fixed_point.cache_info().currsize == 1

    # Call with different resolver - miss
    res3 = substitute_until_fixed_point(expr1, resolver2)
    assert res3 == 30
    assert substitute_until_fixed_point.cache_info().hits == 1
    assert substitute_until_fixed_point.cache_info().misses == 2
    assert substitute_until_fixed_point.cache_info().currsize == 2

    # Call with different expression - miss
    expr2 = x * y
    res4 = substitute_until_fixed_point(expr2, resolver1)
    assert res4 == 2  # 1 * 2
    assert substitute_until_fixed_point.cache_info().hits == 1
    assert substitute_until_fixed_point.cache_info().misses == 3
    assert substitute_until_fixed_point.cache_info().currsize == 3

    substitute_until_fixed_point.cache_clear()  # Clean up


def test_substitute_lambdify_path_numeric():
    """Test the lambdify fast path for direct numeric substitution."""
    from qualtran.surface_code.flasq.utils import _get_cached_lambdified_evaluator

    _get_cached_lambdified_evaluator.cache_clear()
    substitute_until_fixed_point.cache_clear()

    x, y = sympy.symbols("x y")
    expr = x * y + x / 2
    resolver = frozendict({x: 4, y: 5})  # All numeric, all symbols in expr covered

    # First call - should use lambdify, miss in _get_cached_lambdified_evaluator
    result1 = substitute_until_fixed_point(expr, resolver)
    assert result1 == 22.0  # (4*5) + (4/2) = 20 + 2 = 22
    assert isinstance(result1, float)
    assert _get_cached_lambdified_evaluator.cache_info().misses == 1
    assert _get_cached_lambdified_evaluator.cache_info().hits == 0
    assert substitute_until_fixed_point.cache_info().misses == 1

    # Second call with same expression and resolver structure (different values)
    # Should hit _get_cached_lambdified_evaluator, miss in substitute_until_fixed_point
    resolver2 = frozendict({x: 2, y: 3})
    result2 = substitute_until_fixed_point(expr, resolver2)
    assert result2 == 7.0  # (2*3) + (2/2) = 6 + 1 = 7
    assert (
        _get_cached_lambdified_evaluator.cache_info().misses == 1
    )  # No new miss for lambdify
    assert _get_cached_lambdified_evaluator.cache_info().hits == 1  # Hit for lambdify
    assert (
        substitute_until_fixed_point.cache_info().misses == 2
    )  # New miss for main function

    # Third call, same as first - should hit both caches
    result3 = substitute_until_fixed_point(expr, resolver)
    assert result3 == 22.0
    assert (
        _get_cached_lambdified_evaluator.cache_info().hits == 1
    )  # Not called again, still 1 hit from previous
    assert substitute_until_fixed_point.cache_info().hits == 1  # Hit for main function

    _get_cached_lambdified_evaluator.cache_clear()
    substitute_until_fixed_point.cache_clear()


def test_substitute_fallback_path_symbolic_value_in_resolver():
    """Test fallback to subs if resolver contains symbolic values."""
    from qualtran.surface_code.flasq.utils import _get_cached_lambdified_evaluator

    _get_cached_lambdified_evaluator.cache_clear()
    substitute_until_fixed_point.cache_clear()

    x, y, z = sympy.symbols("x y z")
    expr = x + y
    resolver = frozendict({x: 1, y: z * 2})  # y is symbolic

    result = substitute_until_fixed_point(expr, resolver)
    assert result == 1 + z * 2
    assert isinstance(result, sympy.Expr)
    assert _get_cached_lambdified_evaluator.cache_info().hits == 0
    assert (
        _get_cached_lambdified_evaluator.cache_info().misses == 0
    )  # Lambdify path not taken

    _get_cached_lambdified_evaluator.cache_clear()
    substitute_until_fixed_point.cache_clear()


def test_substitute_fallback_path_incomplete_resolver():
    """Test fallback to subs if resolver doesn't cover all free symbols."""
    from qualtran.surface_code.flasq.utils import _get_cached_lambdified_evaluator

    _get_cached_lambdified_evaluator.cache_clear()
    substitute_until_fixed_point.cache_clear()

    x, y = sympy.symbols("x y")
    expr = x + y
    resolver = frozendict({x: 1})  # y is not in resolver

    result = substitute_until_fixed_point(expr, resolver)
    assert result == 1 + y
    assert isinstance(result, sympy.Expr)
    assert _get_cached_lambdified_evaluator.cache_info().hits == 0
    assert (
        _get_cached_lambdified_evaluator.cache_info().misses == 0
    )  # Lambdify path not taken

    _get_cached_lambdified_evaluator.cache_clear()
    substitute_until_fixed_point.cache_clear()


