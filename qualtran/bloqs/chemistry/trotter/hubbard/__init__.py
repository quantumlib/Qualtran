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

r"""Bloqs implementing Trotterized unitary evolution under the Hubbard Hamiltonian.

The Hubbard model is given as a sum of two terms

$$
H = H_h + H_I
$$

where the hopping hamiltonian is given as 
$$
H_h = -\tau \sum_{\langle p, q\rangle, \sigma} 
    \left(a_{p\sigma}^{\dagger} a_{q\sigma} + \mathrm{h.c.} \right)
$$
where the sum is over nearest neighbour lattice sites (under periodic boundary conditions).

Following the [reference](https://arxiv.org/abs/2012.09238) we assume the
shifted form of the interacting Hamiltonian:
$$
H_I = \frac{u}{4} \sum_{p} z_{p\uparrow}z_{p\downarrow}
$$
where $z_{p\sigma} = (2 n_{p\sigma} - 1)$.


For Trotterization we assume the plaquette splitting from the
[reference](https://arxiv.org/abs/2012.09238).
The plaquette splitting rewrites $H_h$ as a sum of $H_h^p$ and $H_h^g$ (for pink and gold
respectively) which when combined tile the entire lattice. Each plaquette
contains four sites and paritions the lattice such that each edge of the lattice
belongs to a single plaquette. Each term within a grouping commutes so that the
unitary can be be implemented as
$$
e^{i H_h^{x}} = \prod_{k\sigma} e^{i H_h^{x(k,\sigma)}}
$$
without further trotter error.
"""
