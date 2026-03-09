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

import qualtran as qlt
from qualtran.l1 import load_bloq, load_module, load_objectstring


def test_load_objectstring_safe():
    # Normal loading of safe context object
    obj = load_objectstring("CtrlSpec()")
    assert isinstance(obj, qlt.CtrlSpec)
    assert obj.cvs == (1,)


def test_issue_1713():
    # https://github.com/quantumlib/Qualtran/issues/1713
    s = "qualtran.bloqs.mcmt.MultiControlZ((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))"
    bloq = load_bloq(s)
    print(bloq)


def test_load_bloq():
    s = "qualtran.bloqs.reflections.ReflectionUsingPrepare(qualtran.bloqs.chemistry.hubbard_model.qubitization.PrepareHubbard(5, 5, 2.0, 0.1), None, -1, 1e-11)"
    bloq = load_bloq(s)
    print(bloq)


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


def test_load_negate():
    module = load_module(
        """
    # Qualtran-L1
    # 1.0.0

    qdef Negate
    [
        x: QUInt(8),
    ] {
        x2                   = BitwiseNot(8)    [x=x]
        x3                   = AddK(k=1)        [x=x2]
                               return           [x=x3]
    }


    extern qdef BitwiseNot(8)
    from qualtran.bloqs.arithmetic.BitwiseNot(QUInt(8))
    [x: QUInt(8)]

    extern qdef AddK(k=1)
    from qualtran.bloqs.arithmetic.AddK(QUInt(8), 1)
    [x: QUInt(8)]"""
    )

    assert set(module.keys()) == {'Negate', 'BitwiseNot(8)', 'AddK(k=1)'}
    cbloq: qlt.CompositeBloq = module['Negate']
    assert len(cbloq.bloq_instances) == 2
