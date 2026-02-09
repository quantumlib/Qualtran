#  Copyright 2025 Google LLC
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

"""A representation of a State as defined in https://arxiv.org/abs/1403.2975 and its operations."""

from typing import Union

import attrs

import qualtran.rotation_synthesis._math_config as mc
import qualtran.rotation_synthesis._typing as rst
from qualtran.rotation_synthesis.lattice import _geometry
from qualtran.rotation_synthesis.lattice import _grid_operators as go
from qualtran.rotation_synthesis.rings import _zw


def _grid_operation_path_converter(seq):
    return tuple(seq)


@attrs.frozen
class GridOperatorAction:
    """A grid operator and the sequence of operations that built it.

    Attributes:
        g: The grid operator.
        path: The sequence of operations that built the grid operator.
    """

    g: go.GridOperator
    path: tuple[str, ...] = attrs.field(default=(), converter=_grid_operation_path_converter)

    def shift(self, k: int) -> "GridOperatorAction":
        """Returns the Action corresponding to shift(k)-self-shift(k)."""
        return attrs.evolve(
            self, g=self.g.shift(k), path=(f"Shift({k})",) + self.path + (f"Shift({-k})",)
        )

    def followed_by(self, other: "GridOperatorAction") -> "GridOperatorAction":
        """Returns the Action corresponding to self followed by other."""
        return GridOperatorAction(self.g @ other.g, self.path + other.path)


@attrs.frozen
class SelingerState:
    r"""A state is a region in space defined by two ellipses (A, B).

    A point $p$ belongs to the region iff $p \in A$ and $p^\bullet \in B$. Where
    $p^\bullet$ is the sqrt2-conjugate of $p$.

    Attributes:
        m1: The first ellipse.
        m2: The second ellipse.

    References:
        Appendix A of https://arxiv.org/abs/1403.2975
    """

    m1: _geometry.Ellipse
    m2: _geometry.Ellipse

    def skew(self, config: mc.MathConfig) -> rst.Real:
        """Returns the skew of the state."""
        return self.m1.parametric_form(config).b ** 2 + self.m2.parametric_form(config).b ** 2

    def bias(self, config: mc.MathConfig) -> rst.Real:
        """Returns the bias of the state."""
        return self.m2.parametric_form(config).z - self.m1.parametric_form(config).z

    @staticmethod
    def from_parametric_forms(
        e1: _geometry.EllipseParametricForm,
        e2: _geometry.EllipseParametricForm,
        config: mc.MathConfig,
    ) -> "SelingerState":
        """Constructs a state from the parametric form of the two regions."""
        l_value = 1 + config.sqrt(2)
        return SelingerState(e1.to_ellipse(l_value), e2.to_ellipse(l_value))

    def shift(self, k: int, config: mc.MathConfig) -> "SelingerState":
        """Shift the state by k resulting in a state with the original bias+2k and the same skew"""
        e1 = self.m1.parametric_form(config)
        e2 = self.m2.parametric_form(config)
        e1 = attrs.evolve(e1, z=e1.z - k)
        sgn = -1 if k % 2 == 1 else 1
        e2 = attrs.evolve(e2, z=e2.z + k, b=e2.b * sgn)
        return SelingerState.from_parametric_forms(e1, e2, config)

    def get_grid_operator(self, config: mc.MathConfig) -> GridOperatorAction:
        """Returns the grid operator that improves the skewness of the state.

        The method follows the rules in appendix A of https://arxiv.org/abs/1403.2975.

        Args:
            config: A MathConfig object.
        """
        k = config.floor((1 - self.bias(config)) / 2)
        if k:
            # Lemma A.9
            return self.shift(k, config).get_grid_operator(config).shift(k)
        assert abs(self.bias(config)) <= 1

        e1 = self.m1.parametric_form(config)
        e2 = self.m2.parametric_form(config)

        # See Appendix A.6 for the mapping.
        if e2.b < 0:
            action = GridOperatorAction(go.ZSqrt2)
            action = self.apply(action, config).get_grid_operator(config)
            return attrs.evolve(action, g=go.ZSqrt2 @ action.g)

        if e1.z + e2.z < 0:
            action = GridOperatorAction(go.XSqrt2)
            action = self.apply(action, config).get_grid_operator(config)
            return attrs.evolve(action, g=go.XSqrt2 @ action.g)

        l_value = 1 + config.sqrt2
        if e1.b >= 0:
            if -0.8 <= e1.z <= 0.8 and -0.8 <= e2.z <= 0.8:
                g = go.RSqrt2
                name = "R"
            elif e1.z <= 0.3 and 0.8 <= e2.z:
                g = go.KSqrt2
                name = "K"
            elif 0.3 <= e1.z and 0.3 <= e2.z:
                c = min(e1.z, e2.z)  # type: ignore[type-var]
                n = max(1, config.floor(l_value**c / 2))
                g = go.ASqrt2**n
                name = f"A^{n}"
            elif 0.8 <= e1.z and e2.z <= 0.3:
                g = go.KConjSqrt2
                name = "K*"
            else:
                raise ValueError("Shouldn't be here.")
        else:
            if -0.8 <= e1.z <= 0.8 and -0.8 <= e2.z <= 0.8:
                g = go.RSqrt2
                name = "R"
            elif e1.z >= -0.2 and e2.z >= -0.2:
                c = min(e1.z, e2.z)  # type: ignore[type-var]
                n = max(1, config.floor(l_value**c / config.sqrt2))
                g = go.BSqrt2**n
                name = f"B^{n}"
            else:
                raise ValueError("Shouldn't be here.")
        return GridOperatorAction(g, (name,))

    def apply(
        self, operator_or_action: Union[go.GridOperator, GridOperatorAction], config: mc.MathConfig
    ) -> "SelingerState":
        """Applies the given gridoperator/action on the state.

        Args:
            operator_or_action: A GridOperator or GridOpeartionAction to apply.
            config: A MathConfig object.
        """
        m1, m2 = self.m1.D, self.m2.D

        if isinstance(operator_or_action, GridOperatorAction):
            operator = operator_or_action.g
        else:
            operator = operator_or_action
        g = operator.value(config.sqrt2)
        g_conj = operator.sqrt2_conj().value(config.sqrt2)
        m1 = g.T @ m1 @ g / 2
        m2 = g_conj.T @ m2 @ g_conj / 2
        return SelingerState(
            _geometry.Ellipse(
                m1, operator.scaled_inverse().actual_value(config.sqrt2) @ self.m1.center
            ),
            _geometry.Ellipse(
                m2,
                operator.sqrt2_conj().scaled_inverse().actual_value(config.sqrt2) @ self.m2.center,
            ),
        )

    def contains(self, x: _zw.ZW, config: mc.MathConfig) -> bool:
        """Returns whether the given point belongs in the region defined by the state.

        Args:
            x: the point to check.
            config: A MathConfig object.
        """
        u = x.value(config.sqrt2)
        v = x.sqrt2_conj().value(config.sqrt2)
        return self.m1.contains(u.real, u.imag, config) and self.m2.contains(v.real, v.imag, config)
