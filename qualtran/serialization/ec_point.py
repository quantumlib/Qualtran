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

from qualtran.bloqs.factoring.ecc import ECPoint
from qualtran.protos import ec_point_pb2
from qualtran.serialization.args import int_or_sympy_from_proto, int_or_sympy_to_proto


def ec_point_from_proto(point: ec_point_pb2.ECPoint) -> ECPoint:
    return ECPoint(
        x=int_or_sympy_from_proto(point.x),
        y=int_or_sympy_from_proto(point.y),
        mod=int_or_sympy_from_proto(point.mod),
        curve_a=int_or_sympy_from_proto(point.curve_a),
    )


def ec_point_to_proto(point: ECPoint) -> ec_point_pb2.ECPoint:
    return ec_point_pb2.ECPoint(
        x=int_or_sympy_to_proto(point.x),
        y=int_or_sympy_to_proto(point.y),
        mod=int_or_sympy_to_proto(point.mod),
        curve_a=int_or_sympy_to_proto(point.curve_a),
    )
