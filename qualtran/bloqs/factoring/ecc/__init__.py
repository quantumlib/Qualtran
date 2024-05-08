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
#
# isort:skip_file

r"""Bloqs for breaking elliptic curve cryptography systems via the discrete log.

Elliptic curve cryptography is a form of public key cryptography based on the finite
field of elliptic curves. For our purposes, we will denote the group operation as addition
(whose definition we will explore later) $A + B$. We will denote repeated addition
 as $[k] A = A + \dots + A$ ($k$ times).

Within this algebra, the cryptographic scheme relates the public and private keys via
$$
Q = [k] P
$$
for private key $k$, public key $Q$, and a choice of base point $P$. The cryptographic
security comes from the difficulty of inverting the multiplication. I.e. it is difficult
to do a discrete logarithm in this field.

Using Shor's algorithm for the discrete logarithm, we can find $k$ in polynomial time
with a quantum algorithm.
"""

from .ec_point import ECPoint
from .ec_add import ECAdd
from .ec_add_r import ECAddR, ECWindowAddR
from .ec_phase_estimate_r import ECPhaseEstimateR
from .find_ecc_private_key import FindECCPrivateKey
