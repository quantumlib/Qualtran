#  Copyright 2024 Google LLC
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

r"""Bloqs for breaking RSA cryptography systems via integer factorization.

RSA cryptography is a form of public key cryptography based on the difficulty of
factoring the product of two large prime numbers.

Using RSA, the cryptographic scheme chooses two large prime numbers p, q, their product n,
λ(n) = lcm(p - 1, q - 1) where λ is Carmichael's totient function, an integer e such that
1 < e < λ(n), and finally d as d ≡ e^-1 (mod λ(n)). The public key consists of the modulus n and
the public (or encryption) exponent e. The private key consists of the private (or decryption)
exponent d, which must be kept secret. p, q, and λ(n) must also be kept secret because they can be
used to calculate d.

Using Shor's algorithm for factoring, we can find p and q (the factors of n) in polynomial time
with a quantum algorithm.

References:
    [RSA (cryptosystem)](https://en.wikipedia.org/wiki/RSA_(cryptosystem)).
"""

from .rsa_phase_estimate import RSAPhaseEstimate
from .rsa_mod_exp import ModExp
