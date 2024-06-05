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

from qualtran import CtrlSpec
from qualtran.protos import ctrl_spec_pb2
from qualtran.serialization import args, data_types


def ctrl_spec_from_proto(spec: ctrl_spec_pb2.CtrlSpec) -> CtrlSpec:
    return CtrlSpec(
        qdtypes=[data_types.data_type_from_proto(dtype) for dtype in spec.qdtypes],
        cvs=[args.ndarray_from_proto(cvs) for cvs in spec.cvs],
    )


def ctrl_spec_to_proto(spec: CtrlSpec) -> ctrl_spec_pb2.CtrlSpec:
    return ctrl_spec_pb2.CtrlSpec(
        qdtypes=[data_types.data_type_to_proto(dtype) for dtype in spec.qdtypes],
        cvs=[args.ndarray_to_proto(cvs) for cvs in spec.cvs],
    )
