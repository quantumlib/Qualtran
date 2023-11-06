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
r"""SELECT and PREPARE for the first quantized chemistry Hamiltonian.

Here we assume the Born-Oppenheimer Hamiltonian and seek to simulate a
collection of $\eta$ electrons and $L$ static nuclei with a Hamiltonian given
by:
$$
H_{BO} = T + U + V + \frac{1}{2}
\sum_{\ell\ne\kappa=1}^L\frac{\zeta_\ell\zeta_\kappa}{\lVert R_\ell - R_\kappa \rVert}
$$

In the first quantized approach we assume periodic boundary conditions and use a
plane wave Galerkin discretization.
A plane wave basis function is given by
$$
\phi_p(r) = \frac{1}{\sqrt{\Omega}} e^{-i k_p\cdot r}
$$
where $r$ is a position vector in real space, $\Omega$ is the simulation cell
volume and $k_p$ is a reciprocal lattice vector.
In three dimensions we have
$$
k_p = \frac{2\pi p }{\Omega}
$$
for $p \in G$ and
$$
G = [-\frac{N^{1/3} -
1}{2},\frac{N^{1/3} - 1}{2}]^3 \subset \mathcal{Z}^3.
$$
and $N$ is the total number of planewaves.

With these definitions we can write the components of the Hamiltonian as:
$$
T = \sum_{i}^\eta\sum_{p\in G}\frac{\lVert k_p\rVert^2}{2} |p\rangle\langle p|_i
$$
which defines the kinetic energy of the electrons,
$$
U = -\frac{4\pi}{\Omega}
\sum_{\ell=1}^L \sum_{i}^\eta
\sum_{p,q\in G, p\ne q}
\left(
    \zeta_{\ell}
    \frac{e^{i k_{q-p}\cdot R_\ell}}{\lVert k_{p-q}\rVert^2}
    |p\rangle\langle q|_i
\right)
$$
describes the interaction of the electrons and the nuclei, and,
$$
V = \frac{2\pi}{\Omega}
\sum_{i\ne j=1}^\eta
\sum_{p,q\in G, p\ne q}
\sum_{\nu \in G_0}
\left(
    \frac{1}{\lVert k_{\nu}\rVert^2}
    |p + \nu\rangle\langle p|_i
    |q -\nu\rangle\langle q|_i
\right)
$$
describes the electron-electron interaction. The notation $|p\rangle\langle p|_i$ is shorthand for
$I_1\otimes\cdots\otimes |p\rangle \langle p |_j \otimes \cdots \otimes I_\eta$.
The system is represented using a set of $\eta$ signed integer registers each of
size $3 n_p$ where $n_p =  \lceil \log (N^{1/3} + 1) \rceil$, with the factor of
3 accounting for the 3 spatial dimensions.

In the first quantized approach, fermion antisymmetry is encoded through initial
state preparation. Spin labels are also absent and should be accounted for
during state preparation. The cost of initial state preparation is typically
ignored.
"""

from .prepare_t import PrepareTFirstQuantization
from .prepare_uv import PrepareUVFistQuantization
from .select_t import SelectTFirstQuantization
from .select_uv import SelectUVFirstQuantization
