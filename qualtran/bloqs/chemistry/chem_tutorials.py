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
from typing import Optional

import numpy as np
import scipy.optimize
from numpy.typing import NDArray


def linear(x, a, c):
    return a * x + c


def fit_linear(x, y):
    try:
        # pylint: disable-next=unbalanced-tuple-unpacking
        popt, _ = scipy.optimize.curve_fit(linear, x, y)
        return popt
    except np.linalg.LinAlgError:
        return None


def gen_random_chem_ham(num_spin_orb: int):
    tpq = np.random.random((num_spin_orb // 2, num_spin_orb // 2))
    tpq = 0.5 * (tpq + tpq.T)
    eris = np.random.normal(size=(num_spin_orb // 2,) * 4)
    eris += np.transpose(eris, (0, 1, 3, 2))
    eris += np.transpose(eris, (1, 0, 2, 3))
    eris += np.transpose(eris, (2, 3, 0, 1))
    return tpq, eris


def plot_linear_log_log(
    fig,
    ax,
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    label: Optional[str] = None,
    color='C0',
):
    slope, intr = fit_linear(np.log(xs), np.log(ys))
    x_min = xs[0]
    x_max = xs[-1]
    x_vals = np.linspace(x_min, x_max, 200)
    y_vals = np.exp(intr) * x_vals**slope
    if label is None:
        label = ''
    ax.loglog(xs, ys, marker='o', ls='None', label=rf'{label} $N^{{{slope:3.2f}}}$', color=color)
    ax.loglog(x_vals, y_vals, marker='None', linestyle='--', color=color)
    ax.legend()
