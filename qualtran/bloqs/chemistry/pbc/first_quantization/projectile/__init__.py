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
r"""SELECT and PREPARE for the first quantized chemistry Hamiltonian with a quantum projectile.

Here we consider the simulation setup from [Quantum computation of stopping
power for inertial fusion target design](https://arxiv.org/abs/2308.12352),
which is concerned with the dyamics on a (quantum) projectile traversing a
medium, which is modelled using a first quantizated representation. The main
modifications are that we need to add the Hamiltonian terms for the projectile
and allow for a different (larger) set of momenta for the projectile.

Specifically, the modified Hamiltonian is given by

$$
H = T + U + V + T_\mathrm{proj} + U_{\mathrm{proj}} + V_{\mathrm{elec-proj}},
$$
where $T$, $U$, and $V$ are the same as they were for the [first quantized
Hamiltonian](../first_quantization.ipynb). The new terms are
$$
T_\mathrm{proj} =
\sum_{p\in\tilde{G}}
\frac{\lVert k_p - k_\mathrm{proj}\rVert^2}{2 M_\mathrm{proj}}
|p\rangle\langle p|_\mathrm{proj},
$$
which describes the kinetic energy of the projectile in the center of momentum frame,
$$
U_\mathrm{proj} = -\frac{4\pi}{\Omega}
\sum_{\ell=1}^L \sum_{i}^\eta
\sum\limits_{\substack{{p,q\in \tilde{G} \\ p\ne q}}}
\left(
    \zeta_{\ell}
    \zeta_{\mathrm{proj}}
    \frac{e^{i k_{q-p}\cdot R_\ell}}{\lVert k_{p-q}\rVert^2}
    |p\rangle\langle q|_\mathrm{proj}
\right),
$$
which describes the Coulomb interaction of the projectile with the nuclei of the medium, and
$$
V = -\frac{4\pi}{\Omega}
\sum_{i=1}^\eta
\sum_{p\in G}
\sum_{q\in \tilde{G}}
\sum\limits_{\substack{\nu \in G_0 \\ (p+\nu)\in G \\ (q-\nu)\in \tilde{G}}}
\left(
    \frac{\zeta_\mathrm{proj}}{\lVert k_{\nu}\rVert^2}
    |p + \nu\rangle\langle p|_i
    |q -\nu\rangle\langle q|_\mathrm{proj}
\right),
$$
which describes the interaction of the projectile with the electrons of the medium.

The projectile is represented by a single system register (of size $n_n$ bits),
and is initially modelled as a Gaussian wavepacket. The projectile has charge
$\zeta_\mathrm{proj}$ and mass $M_\mathrm{proj}$. In practice, to model stopping
power a momentum `kick' of $k_\mathrm{proj}$ is applied at $t=0$ and
we monitor how the kinetic energy of the projectile changes with time. Here, we
will only concern ourselves with the additional block encoding costs of
including this quantum projectile.  Again, state preparation costs are currently
ignored as this is logarithmic in the size of the basis set.
"""

from .select_and_prepare import PrepareFirstQuantizationWithProj, SelectFirstQuantizationWithProj
