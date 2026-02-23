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

import pytest

from qualtran import CtrlSpec
from qualtran.l1._parse_eval import load_objectstring


def test_load_objectstring_safe():
    # Normal loading of safe context object
    obj = load_objectstring("CtrlSpec()")
    assert isinstance(obj, CtrlSpec)
    assert obj.cvs == (1,)


def test_load_objectstring_unsafe():
    # If safe=True (default), it creates an UnevaluatedCValue for unknown things
    from qualtran.l1._eval import UnevaluatedCValue

    obj = load_objectstring("MyFakeClass()")
    assert isinstance(obj, UnevaluatedCValue)

    # If safe=False, it tries to import
    with pytest.raises(ValueError, match="Unknown CValueNode"):
        load_objectstring("MyFakeClass()", safe=False)


def test_load_objectstring_unsafe_import():
    # If safe=False, it tries to import it
    obj = load_objectstring("sympy.Symbol('n')", safe=False)
    import sympy

    assert isinstance(obj, sympy.Symbol)
