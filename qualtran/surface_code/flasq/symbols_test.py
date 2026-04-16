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

# symbols_test.py
# Phase 1 characterization tests for qualtran.surface_code.flasq/symbols.py
#
# Verifies that the symbolic constants (especially MIXED_FALLBACK_T_COUNT)
# match the published formulas from Kliuchnikov et al. (2023) as referenced
# in the FLASQ paper.

import math

import pytest
import sympy

from qualtran.surface_code.flasq.symbols import (
    MIXED_FALLBACK_T_COUNT,
    ROTATION_ERROR,
    T_REACT,
    V_CULT_FACTOR,
)


class SymbolDefinitionsTestSuite:
    """Verify that the symbolic constants are correctly defined."""

    def test_symbols_are_sympy_symbols(self):
        """All exported symbols should be sympy.Symbol instances."""
        assert isinstance(ROTATION_ERROR, sympy.Symbol)
        assert isinstance(V_CULT_FACTOR, sympy.Symbol)
        assert isinstance(T_REACT, sympy.Symbol)

    def test_symbol_names(self):
        """Verify the string names of each symbol."""
        assert str(ROTATION_ERROR) == "ROTATION_ERROR"
        assert str(V_CULT_FACTOR) == "V_CULT_FACTOR"
        assert str(T_REACT) == "t_react"

    def test_mixed_fallback_t_count_is_expression(self):
        """MIXED_FALLBACK_T_COUNT should be a sympy expression, not a symbol."""
        assert not isinstance(MIXED_FALLBACK_T_COUNT, sympy.Symbol)
        # It should depend on ROTATION_ERROR
        assert ROTATION_ERROR in MIXED_FALLBACK_T_COUNT.free_symbols


class MixedFallbackTCountTestSuite:
    """Verify MIXED_FALLBACK_T_COUNT against the paper formula.

    The formula is: 4.86 + 0.53 * log2(1/epsilon)
    which is equivalent to: 4.86 - 0.53 * log2(epsilon)

    Reference: Kliuchnikov et al. (2023), used in the FLASQ paper.
    """

    @pytest.mark.parametrize(
        "rotation_error, expected_raw",
        [
            (1e-1, 4.86 + 0.53 * math.log2(1 / 1e-1)),
            (1e-3, 4.86 + 0.53 * math.log2(1 / 1e-3)),
            (1e-6, 4.86 + 0.53 * math.log2(1 / 1e-6)),
            (1e-10, 4.86 + 0.53 * math.log2(1 / 1e-10)),
        ],
    )
    def test_mixed_fallback_t_count_concrete_values(self, rotation_error, expected_raw):
        """Substituting concrete ROTATION_ERROR values should match the formula."""
        result = MIXED_FALLBACK_T_COUNT.subs(ROTATION_ERROR, rotation_error)
        result_float = float(result)
        assert result_float == pytest.approx(
            expected_raw, rel=1e-10
        ), f"For eps={rotation_error}: got {result_float}, expected {expected_raw}"

    def test_mixed_fallback_monotonically_increasing(self):
        """Smaller rotation error should require more T gates."""
        errors = [1e-2, 1e-4, 1e-8, 1e-12]
        t_counts = [float(MIXED_FALLBACK_T_COUNT.subs(ROTATION_ERROR, e)) for e in errors]
        for i in range(len(t_counts) - 1):
            assert t_counts[i] <= t_counts[i + 1], (
                f"T count should increase as error decreases: "
                f"eps={errors[i]} -> {t_counts[i]}, "
                f"eps={errors[i+1]} -> {t_counts[i+1]}"
            )
