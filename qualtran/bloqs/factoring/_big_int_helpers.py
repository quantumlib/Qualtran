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
from typing import SupportsInt, Union

import attrs
import sympy


def int_or_expr(x: Union[SupportsInt, sympy.Expr]) -> Union[int, sympy.Expr]:
    if isinstance(x, sympy.Expr):
        return x
    return int(x)


def python_int_field():
    """An attrs field that requires an infinite-precision Python integer."""
    return attrs.field(
        validator=attrs.validators.instance_of((int, sympy.Expr)), converter=int_or_expr
    )
