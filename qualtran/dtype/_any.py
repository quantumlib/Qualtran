#  Copyright 2026 Google LLC
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

from typing import Any, Iterable

import attrs
from numpy.typing import NDArray

from qualtran.symbolics import is_symbolic, SymbolicInt

from ._base import BitEncoding, QDType
from ._uint import _UInt


@attrs.frozen
class QAny(QDType[Any]):
    """Opaque bag-of-qubits type."""

    bitsize: SymbolicInt

    @property
    def _bit_encoding(self) -> BitEncoding[Any]:
        return _UInt(self.bitsize)

    def __attrs_post_init__(self):
        if is_symbolic(self.bitsize):
            return

        if not isinstance(self.bitsize, int):
            raise ValueError(f"Bad bitsize for QAny: {self.bitsize}")

    def get_classical_domain(self) -> Iterable[Any]:
        raise TypeError(f"Ambiguous domain for {self}. Please use a more specific type.")

    def assert_valid_classical_val(self, val: Any, debug_str: str = 'val'):
        pass

    def assert_valid_classical_val_array(self, val_array: NDArray, debug_str: str = 'val'):
        pass
