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
"""Some utility functions for chemistry tutorials"""
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from numpy.typing import NDArray


def linear(x: NDArray[np.float64], a: float, c: float) -> NDArray[np.float64]:
    r"""Evaluate the linear function $y = a * x + c$.

    Args:
        x: The x values to evaluate the function at.
        a: The slope.
        c: The intercept.

    Returns:
        y: An array of the linear function values evaluated for each value of x.
    """
    return a * x + c


def fit_linear(x: NDArray[np.float64], y: NDArray[np.float64]) -> Tuple[float, float]:
    """Fit a line given x and y values.

    Args:
        x: the independent variable (x value) for the linear fit.
        y: the dependent (y) for the linear fit.

    Returns:
        slope: The slope of the linear fit.
        intercept: the intercept of the linear fit.

    Raises:
        np.linalg.LinAlgError: if fitting fails.
    """
    # pylint: disable-next=unbalanced-tuple-unpacking
    popt, _ = scipy.optimize.curve_fit(linear, x, y)
    return popt


def gen_random_chem_ham(num_spin_orb: int):
    """Generate random chemistry hamiltonian with 8-fold symmetry.

    Args:
        num_spin_orb: The number of spin orbitals.

    Returns:
        tpq: 2D array of one-body matrix elements of (size num_spin_orb // 2)^2.
        eris: 4D array of one-body matrix elements of size (num_spin_orb // 2)^4.
    """
    tpq = np.random.random((num_spin_orb // 2, num_spin_orb // 2))
    tpq = 0.5 * (tpq + tpq.T)
    eris = np.random.normal(size=(num_spin_orb // 2,) * 4)
    eris += np.transpose(eris, (0, 1, 3, 2))
    eris += np.transpose(eris, (1, 0, 2, 3))
    eris += np.transpose(eris, (2, 3, 0, 1))
    return tpq, eris


def plot_linear_log_log(
    ax: plt.Axes,
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    label: Optional[str] = None,
    color: str = 'C0',
):
    """Fit a power law to the input data set and plot on existing axis.

    Plots

    $$
        y = a * x^b + c
    $$

    on a log-log plot.

    Args:
        ax: The matplotlib axis.
        xs: The x-values for the fit.
        ys: The y-values for the fit.
        label: An optioanl text label for the data set. In None the legend reads $N^b$.
        color: The color for the data set.
    """
    slope, intr = fit_linear(np.log(xs), np.log(ys))
    x_min = xs[0]
    x_max = xs[-1]
    x_vals = np.linspace(x_min, x_max, 200)
    y_vals = np.exp(intr) * x_vals**slope
    if label is None:
        label = ''
    ax.loglog(
        xs,
        ys,
        marker='o',
        ls='None',
        label=rf'{label} ${{{np.exp(intr):3.1f}}}N^{{{slope:3.2f}}}$',
        color=color,
    )
    ax.loglog(x_vals, y_vals, marker='None', linestyle='--', color=color)
    ax.legend()
