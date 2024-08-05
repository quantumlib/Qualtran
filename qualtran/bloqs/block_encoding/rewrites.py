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

r"""Rewrite rules to optimize the construction of block encodings."""

from collections import defaultdict
from typing import Dict

from qualtran.bloqs.block_encoding import (
    BlockEncoding,
    LinearCombination,
    Phase,
    Product,
    SparseMatrix,
    TensorProduct,
    Unitary,
)
from qualtran.symbolics import smax


def collect_like_terms(x: BlockEncoding) -> BlockEncoding:
    """Optimize a `BlockEncoding` by collecting like terms in nested `LinearCombination`s.

    The effect is to replace instances of
    ```
    LinearCombination(
        (LinearCombination((a, b), (l_a, l_b)), LinearCombination((c, d), (l_c, l_d)),
        (l_ab, l_cd)
    )
    ```
    with
    ```
    LinearCombination(
        (a, b, c, d), (l_ab * l_a, l_ab * l_b, l_cd * l_c, l_cd * l_d)
    )
    ```
    which requires fewer resources.
    """

    if isinstance(x, (Unitary, SparseMatrix)):
        return x
    if isinstance(x, Phase):
        return Phase(collect_like_terms(x.block_encoding), x.phi, x.eps)
    if isinstance(x, TensorProduct):
        return TensorProduct(tuple(collect_like_terms(y) for y in x.block_encodings))
    if isinstance(x, Product):
        return Product(tuple(collect_like_terms(y) for y in x.block_encodings))
    if isinstance(x, LinearCombination):
        block_encodings: Dict[BlockEncoding, float] = defaultdict(float)
        terms = tuple(collect_like_terms(y) for y in x._block_encodings)
        lambd_bits = x.lambd_bits
        for y, l in zip(terms, x._lambd):
            if isinstance(y, LinearCombination):
                lambd_bits = smax(lambd_bits, x.lambd_bits + y.lambd_bits)
                for z, m in zip(y._block_encodings, y._lambd):
                    block_encodings[z] += l * m
            else:
                block_encodings[y] += l
        return LinearCombination(
            tuple(block_encodings.keys()), tuple(block_encodings.values()), lambd_bits
        )
    return x
