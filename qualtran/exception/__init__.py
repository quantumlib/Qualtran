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

"""Qualtran exceptions that may be raised by the framework."""

from qualtran._infra.bloq import DecomposeNotImplementedError, DecomposeTypeError
from qualtran._infra.composite_bloq import BloqError, DidNotFlattenAnythingError

__all__ = [
    'DecomposeTypeError',
    'DecomposeNotImplementedError',
    'BloqError',
    'DidNotFlattenAnythingError',
]
