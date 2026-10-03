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

from __future__ import annotations

import kickmix as km


def test_add():
    a = km.Circuit('''
        CCX q0 q1 q2
        CZ q0 q1
    ''')
    b = km.Circuit('''
        X q2
    ''')
    assert a + b == km.Circuit('''
        CCX q0 q1 q2
        CZ q0 q1
        X q2
    ''')
    assert a + a == km.Circuit('''
        CCX q0 q1 q2
        CZ q0 q1
        CCX q0 q1 q2
        CZ q0 q1
    ''')


def test_add_with_register_data():
    a = km.Circuit('''
        REGISTER r0 "x"
        CCX q0 q1 q2
        CZ q0 q1
    ''')
    b = km.Circuit('''
        REGISTER r0 "y"
        X q2
    ''')
    c = km.Circuit('''
        X q2
    ''')
    assert a + b == a + c == km.Circuit('''
        REGISTER r0 "x"
        CCX q0 q1 q2
        CZ q0 q1
        X q2
    ''')
    assert b + a == km.Circuit('''
        REGISTER r0 "y"
        X q2
        CCX q0 q1 q2
        CZ q0 q1
    ''')
    assert c + a == km.Circuit('''
        REGISTER r0 "x"
        X q2
        CCX q0 q1 q2
        CZ q0 q1
    ''')
    assert a + a == km.Circuit('''
        REGISTER r0 "x"
        CCX q0 q1 q2
        CZ q0 q1
        CCX q0 q1 q2
        CZ q0 q1
    ''')
    assert b + b == km.Circuit('''
        REGISTER r0 "y"
        X q2
        X q2
    ''')


def test_mul():
    assert km.Circuit() * 0 == km.Circuit()
    assert km.Circuit() * 100 == km.Circuit()
    a = km.Circuit('''
        CCX q0 q1 q2
        CZ q0 q1
    ''')
    assert a * -999 == -999 * a == km.Circuit()
    assert a * -1 == -1 * a == km.Circuit()
    assert a * 0 == 0 * a == km.Circuit()
    assert a * 1 == 1 * a == a
    assert a * 2 == 2 * a == km.Circuit('''
        CCX q0 q1 q2
        CZ q0 q1
        CCX q0 q1 q2
        CZ q0 q1
    ''')
    assert a * 3 == 3 * a == km.Circuit('''
        CCX q0 q1 q2
        CZ q0 q1
        CCX q0 q1 q2
        CZ q0 q1
        CCX q0 q1 q2
        CZ q0 q1
    ''')


def test_mul_with_register_data():
    a = km.Circuit('''
        REGISTER r0 "test"
        CCX q0 q1 q2
        CZ q0 q1
    ''')
    assert a * 0 == 0 * a == a * -1 == -1 * a == km.Circuit('''
        REGISTER r0 "test"
    ''')
    assert a * 1 == 1 * a == a
    assert a * 2 == 2 * a == km.Circuit('''
        REGISTER r0 "test"
        CCX q0 q1 q2
        CZ q0 q1
        CCX q0 q1 q2
        CZ q0 q1
    ''')
    assert a * 3 == 3 * a == km.Circuit('''
        REGISTER r0 "test"
        CCX q0 q1 q2
        CZ q0 q1
        CCX q0 q1 q2
        CZ q0 q1
        CCX q0 q1 q2
        CZ q0 q1
    ''')
