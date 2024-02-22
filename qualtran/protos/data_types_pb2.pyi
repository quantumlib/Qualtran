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
import qualtran.protos.args_pb2
import sys
import typing

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class QBit(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    BITSIZE_FIELD_NUMBER: builtins.int
    bitsize: builtins.int
    def __init__(
        self,
        *,
        bitsize: builtins.int | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["_bitsize", b"_bitsize", "bitsize", b"bitsize"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["_bitsize", b"_bitsize", "bitsize", b"bitsize"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_bitsize", b"_bitsize"]) -> typing_extensions.Literal["bitsize"] | None: ...

global___QBit = QBit

@typing_extensions.final
class QAny(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    BITSIZE_FIELD_NUMBER: builtins.int
    @property
    def bitsize(self) -> qualtran.protos.args_pb2.IntOrSympy: ...
    def __init__(
        self,
        *,
        bitsize: qualtran.protos.args_pb2.IntOrSympy | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["_bitsize", b"_bitsize", "bitsize", b"bitsize"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["_bitsize", b"_bitsize", "bitsize", b"bitsize"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_bitsize", b"_bitsize"]) -> typing_extensions.Literal["bitsize"] | None: ...

global___QAny = QAny

@typing_extensions.final
class QInt(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    BITSIZE_FIELD_NUMBER: builtins.int
    @property
    def bitsize(self) -> qualtran.protos.args_pb2.IntOrSympy: ...
    def __init__(
        self,
        *,
        bitsize: qualtran.protos.args_pb2.IntOrSympy | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["_bitsize", b"_bitsize", "bitsize", b"bitsize"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["_bitsize", b"_bitsize", "bitsize", b"bitsize"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_bitsize", b"_bitsize"]) -> typing_extensions.Literal["bitsize"] | None: ...

global___QInt = QInt

@typing_extensions.final
class QIntOnesComp(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    BITSIZE_FIELD_NUMBER: builtins.int
    @property
    def bitsize(self) -> qualtran.protos.args_pb2.IntOrSympy: ...
    def __init__(
        self,
        *,
        bitsize: qualtran.protos.args_pb2.IntOrSympy | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["_bitsize", b"_bitsize", "bitsize", b"bitsize"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["_bitsize", b"_bitsize", "bitsize", b"bitsize"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_bitsize", b"_bitsize"]) -> typing_extensions.Literal["bitsize"] | None: ...

global___QIntOnesComp = QIntOnesComp

@typing_extensions.final
class QUInt(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    BITSIZE_FIELD_NUMBER: builtins.int
    @property
    def bitsize(self) -> qualtran.protos.args_pb2.IntOrSympy: ...
    def __init__(
        self,
        *,
        bitsize: qualtran.protos.args_pb2.IntOrSympy | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["_bitsize", b"_bitsize", "bitsize", b"bitsize"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["_bitsize", b"_bitsize", "bitsize", b"bitsize"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_bitsize", b"_bitsize"]) -> typing_extensions.Literal["bitsize"] | None: ...

global___QUInt = QUInt

@typing_extensions.final
class BoundedQUInt(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    BITSIZE_FIELD_NUMBER: builtins.int
    ITERATION_LENGTH_FIELD_NUMBER: builtins.int
    @property
    def bitsize(self) -> qualtran.protos.args_pb2.IntOrSympy: ...
    @property
    def iteration_length(self) -> qualtran.protos.args_pb2.IntOrSympy: ...
    def __init__(
        self,
        *,
        bitsize: qualtran.protos.args_pb2.IntOrSympy | None = ...,
        iteration_length: qualtran.protos.args_pb2.IntOrSympy | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["_bitsize", b"_bitsize", "_iteration_length", b"_iteration_length", "bitsize", b"bitsize", "iteration_length", b"iteration_length"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["_bitsize", b"_bitsize", "_iteration_length", b"_iteration_length", "bitsize", b"bitsize", "iteration_length", b"iteration_length"]) -> None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_bitsize", b"_bitsize"]) -> typing_extensions.Literal["bitsize"] | None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_iteration_length", b"_iteration_length"]) -> typing_extensions.Literal["iteration_length"] | None: ...

global___BoundedQUInt = BoundedQUInt

@typing_extensions.final
class QFxp(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    BITSIZE_FIELD_NUMBER: builtins.int
    NUM_FRAC_FIELD_NUMBER: builtins.int
    SIGNED_FIELD_NUMBER: builtins.int
    @property
    def bitsize(self) -> qualtran.protos.args_pb2.IntOrSympy: ...
    @property
    def num_frac(self) -> qualtran.protos.args_pb2.IntOrSympy: ...
    signed: builtins.bool
    def __init__(
        self,
        *,
        bitsize: qualtran.protos.args_pb2.IntOrSympy | None = ...,
        num_frac: qualtran.protos.args_pb2.IntOrSympy | None = ...,
        signed: builtins.bool | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["_bitsize", b"_bitsize", "_num_frac", b"_num_frac", "_signed", b"_signed", "bitsize", b"bitsize", "num_frac", b"num_frac", "signed", b"signed"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["_bitsize", b"_bitsize", "_num_frac", b"_num_frac", "_signed", b"_signed", "bitsize", b"bitsize", "num_frac", b"num_frac", "signed", b"signed"]) -> None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_bitsize", b"_bitsize"]) -> typing_extensions.Literal["bitsize"] | None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_num_frac", b"_num_frac"]) -> typing_extensions.Literal["num_frac"] | None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_signed", b"_signed"]) -> typing_extensions.Literal["signed"] | None: ...

global___QFxp = QFxp
