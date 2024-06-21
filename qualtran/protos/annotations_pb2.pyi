"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file

Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import builtins
import google.protobuf.descriptor
import google.protobuf.message
import typing

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing.final
class TComplexity(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    CLIFFORD_FIELD_NUMBER: builtins.int
    ROTATIONS_FIELD_NUMBER: builtins.int
    T_FIELD_NUMBER: builtins.int
    clifford: builtins.int
    rotations: builtins.int
    t: builtins.int
    def __init__(
        self,
        *,
        clifford: builtins.int = ...,
        rotations: builtins.int = ...,
        t: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["clifford", b"clifford", "rotations", b"rotations", "t", b"t"]) -> None: ...

global___TComplexity = TComplexity
