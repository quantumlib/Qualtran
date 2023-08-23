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

import abc

import attr
import numpy as np
from attrs import frozen

_PRETTY_FLOAT = attr.ib(type=float, repr=lambda x: f'{x:g}')


class RotationCostModel(abc.ABC):
    """Analytical estimate of number of T gates needed to approximate a rotation given an error budget."""

    @abc.abstractmethod
    def mean_cost(self, error_budge: float | np.ndarray) -> float:
        """Returns the mean number of T gates needed to approx a rotation."""

    @abc.abstractmethod
    def max_cost(self, error_budge: float | np.ndarray) -> float:
        """Returns the max number of T gates needed to approx a rotation."""


@frozen
class RotationCostLinearModel(RotationCostModel):
    r"""RotationCostLinearModel is a linear model in the log of the error budget.

        #T gates = $-A \log_2{budget} + B$

    Attributes:
        A_mean: Mean value of the coefficient of $log_2{budget}$.
        B_mean: Mean value of the offset/overhead.
        A_max: Max value of the coefficient of $log_2{budget}$.
        B_max: Max value of the offset/overhead.
    """
    A_mean = _PRETTY_FLOAT
    B_mean = _PRETTY_FLOAT
    A_max = _PRETTY_FLOAT
    B_max = _PRETTY_FLOAT

    gateset = attr.ib(type=str, default='')
    approximation_protocol = attr.ib(type=str, default='')
    reference = attr.ib(type=str, default='')

    def mean_cost(self, error_budge: float | np.ndarray) -> float:
        return self.A_mean * np.log2(1.0 / error_budge) + self.B_mean

    def max_cost(self, error_budge: float | np.ndarray) -> float:
        return self.A_max * np.log2(1.0 / error_budge) + self.B_max


MixedFallBackCliffordT = RotationCostLinearModel(
    A_mean=0.53,
    B_mean=4.86,
    A_max=0.57,
    B_max=8.83,
    gateset='Clifford+T',
    approximation_protocol='Mixed fallback',
    reference='https://arxiv.org/abs/2203.10064:Table1',
)

BeverlandEtAl = RotationCostLinearModel(
    A_mean=0.53,
    B_mean=5.3,
    A_max=0.53,
    B_max=5.3,
    gateset='Clifford+T',
    approximation_protocol='Mixed fallback',
    reference='https://arxiv.org/abs/2211.07629:D2',
)
