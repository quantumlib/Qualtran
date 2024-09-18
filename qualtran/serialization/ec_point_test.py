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

import pytest

from qualtran.bloqs.factoring.ecc import ECPoint
from qualtran.serialization.ec_point import ec_point_from_proto, ec_point_to_proto


@pytest.mark.parametrize(
    "ec_point",
    [
        ECPoint(x=15, y=13, mod=17, curve_a=0),
        ECPoint(x=0, y=2, mod=7, curve_a=3),
        ECPoint(x=0, y=2, mod=7),
    ],
)
def test_ec_point_to_proto_roundtrip(ec_point: ECPoint):
    ec_point_proto = ec_point_to_proto(ec_point)
    ec_point_roundtrip = ec_point_from_proto(ec_point_proto)
    assert ec_point == ec_point_roundtrip
