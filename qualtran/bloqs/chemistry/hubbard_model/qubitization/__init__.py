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
r"""Simulating the Hubbard model Hamiltonian using qubitization.

This module follows section V. of Encoding Electronic Spectra in Quantum Circuits with Linear T
Complexity. Babbush et. al. 2018. [arxiv:1805.03662](https://arxiv.org/abs/1805.03662).

The 2D Hubbard model is a special case of the electronic structure Hamiltonian
restricted to spins on a planar grid.

$$
H = -t \sum_{\langle p,q \rangle, \sigma} a_{p,\sigma}^\dagger a_{q,\sigma}
    + \frac{u}{2} \sum_{p,\alpha\ne\beta} n_{p, \alpha} n_{p, \beta}
$$

Under the Jordan-Wigner transformation to Pauli operators, this is

$$
\def\Zvec{\overrightarrow{Z}}
\def\hop#1{#1_{p,\sigma} \Zvec #1_{q,\sigma}}
H = -\frac{t}{2} \sum_{\langle p,q \rangle, \sigma} (\hop{X} + \hop{Y})
  + \frac{u}{8} \sum_{p,\alpha\ne\beta} Z_{p,\alpha}Z_{p,\beta}
  - \frac{u}{4} \sum_{p,\sigma} Z_{p,\sigma} + \frac{uN}{4}\mathbb{1}
$$

This can be simulated using a qubitization circuit, which consists of PREPARE and SELECT
operations. This module contains `SelectHubbard` and `PrepareHubbard`, with particular
compilation optimizations for the Hubbard model. For more insight into how Select and Prepare
operations can be combined into a quantum walk, please see
[Qubitization Walk Operator](./qubitization_walk_operator.ipynb).

With these operators, our selection register has indices
for $p$, $\alpha$, $q$, and $\beta$ as well as two indicator bits $U$ and $V$. There are four cases
considered in both the PREPARE and SELECT operations corresponding to the terms in the Hamiltonian:

 - $U=1$, single-body Z
 - $V=1$, spin-spin ZZ term
 - $p<q$, XZX term
 - $p>q$, YZY term.
"""
from .prepare_hubbard import PrepareHubbard
from .select_hubbard import SelectHubbard
from .walk_operator import get_walk_operator_for_hubbard_model
