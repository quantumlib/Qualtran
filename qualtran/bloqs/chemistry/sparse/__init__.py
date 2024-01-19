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
r"""SELECT and PREPARE for the second quantized sparse chemistry Hamiltonian.

Starting from the traditional second quantized chemistry Hamiltonian
$$
H = \sum_\sigma \sum_{pq} T_{pq} a_{p\sigma}^\dagger a_{q\sigma}
+
\frac{1}{2}\sum_{\alpha\beta}
\sum_{pqrs}
V_{pqrs} a_{p\sigma}^\dagger a_{r\beta}^\dagger a_{s\beta} a_{q\alpha},
$$
where $a_{p\sigma}$ ($a_{q\sigma}^\dagger$) annihilate (create) an electron in the
$p$-th orbital of spin $\sigma$.
We can rewrite this expression using the Jordan-Wigner transformation as
$$
H = T' + V',
$$
where
$$
T' = \frac{1}{2} \sum_\sigma \sum_{pq} T_{pq}'Q_{pq\sigma},
$$
$$
V' = \sum_{\alpha\beta}\sum_{pqrs}V_{pqrs}Q_{pq\alpha}Q_{rs\beta},
$$
and $V = (pq|rs)$ are the usual two-electron integrals in chemist's notation,
$$
T'_{pq} = T_{pq} - \sum_r V_{pqrr},
$$
and
$$
Q_{pq\sigma} =
\begin{cases}
X_{p\sigma}\vec{Z}X_{q\sigma} & p < q \\
Y_{p\sigma}\vec{Z}Y_{q\sigma} & p > q \\
-Z_{p\sigma} & p = q
\end{cases}.
$$
The sparse Hamiltonian simply sets to zero any term in the Hamiltonian where
$|V_{pqrs}|$ is less than some threshold. This reduces the
amount of data that is required to be loaded during state preparation as only
non-zero symmetry inequivalent terms are required (the two electron integrals
exhibit 8-fold permutational symmetry). Symmetries are restored by initially
appropriately weighting these non-zero terms and then using $|+\rangle$ states
to perform control swaps between the $pqrs$ registers.
"""

from .prepare import PrepareSparse
from .select_bloq import SelectSparse
