#  Copyright 2025 Google LLC
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

from typing import Union

import mpmath
import numpy as np
from typing_extensions import TypeIs

# mypy has a bug where it doesn't understand numbers.* https://github.com/python/mypy/issues/3186
# So we define our own types
Real = Union[float, np.floating, mpmath.ctx_mp_python.mpf]
Integral = Union[int, np.integer]
Complex = Union[complex, np.complexfloating, mpmath.ctx_mp_python.mpc]


def is_int(x) -> TypeIs[Integral]:
    return isinstance(x, (int, np.integer))
