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
from qualtran.bloqs.factoring.ecc import ECPoint


def test_ec_point_overrides():
    p = ECPoint(15, 13, mod=17, curve_a=0)
    neg_p = -p
    assert p + neg_p == ECPoint.inf(mod=17, curve_a=0)
    assert 1 * p == p
    assert 2 * p == (p + p)
    assert 3 * p == (p + p + p)


def test_ec_point_addition():
    g = ECPoint(15, 13, mod=17, curve_a=0)
    _script = """
    # https://github.com/nakov/Practical-Cryptography-for-Developers-Book/blob/master/asymmetric-key-ciphers/elliptic-curve-cryptography-ecc.md
    from tinyec.ec import SubGroup, Curve

    field = SubGroup(p=17, g=(15, 13), n=18, h=1)
    curve = Curve(a=0, b=7, field=field, name='p1707')

    print('ref_multiples = [')
    for k in range(0, 25):
        p = k * curve.g
        print(f"  ({p.x}, {p.y}),")
    print(']')
    """
    ref_multiples = [
        (None, None),
        (15, 13),
        (2, 10),
        (8, 3),
        (12, 1),
        (6, 6),
        (5, 8),
        (10, 15),
        (1, 12),
        (3, 0),
        (1, 5),
        (10, 2),
        (5, 9),
        (6, 11),
        (12, 16),
        (8, 14),
        (2, 7),
        (15, 4),
        (None, None),
        (15, 13),
        (2, 10),
        (8, 3),
        (12, 1),
        (6, 6),
        (5, 8),
    ]

    for k in range(1, 25):
        res: ECPoint = g * k
        ref_x, ref_y = ref_multiples[k]
        ref_x = ref_x if ref_x is not None else 0
        ref_y = ref_y if ref_y is not None else 0
        assert (res.x, res.y) == (ref_x, ref_y)
