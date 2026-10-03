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

import kickmix as km


def test_qid():
    q = km.q(5)

    assert q == km.q(5)
    assert q != km.q(4)
    assert q != km.b(4)
    assert q != object()
    assert q != "q5"
    assert hash(q) == hash(km.q(5))
    assert str(q) == "q5"
    assert km.q(0) != bool(False)

    assert repr(q) == "km.q(5)"
    assert q.id == 5
    assert eval(repr(q), {'km': km}, {}) == q


def test_bid():
    b = km.b(5)

    assert b == km.b(5)
    assert b != km.b(4)
    assert b != km.q(4)
    assert b != object()
    assert b != "q5"
    assert b != "b5"
    assert hash(b) == hash(km.b(5))
    assert str(b) == "b5"
    assert km.b(0) != bool(False)

    assert repr(b) == "km.b(5)"
    assert b.id == 5
    assert eval(repr(b), {'km': km}, {}) == b


def test_xbid():
    b = km.xb(5)

    assert b == km.xb(5)
    assert b != km.xb(4)
    assert b != km.q(4)
    assert b != km.b(4)
    assert b != object()
    assert b != "q5"
    assert b != "b5"
    assert b != "xb5"
    assert hash(b) == hash(km.xb(5))
    assert str(b) == "xb5"
    assert km.xb(0) != bool(False)

    assert repr(b) == "km.xb(5)"
    assert b.id == 5
    assert eval(repr(b), {'km': km}, {}) == b


def test_xbool():
    assert km.xbool(False) != bool(False)
    assert km.xbool(False) == km.xbool(False)
    assert km.xbool(True) == km.xbool(True)
    assert km.xbool(False) != km.xbool(True)
    assert km.xbool(False) != object()
    assert km.xbool(False) != ''
    assert km.xbool(False) != bool(True)
    assert hash(km.xbool(True)) == hash(km.xbool(True))
    assert hash(km.xbool(False)) == hash(km.xbool(False))
    assert str(km.xbool(False)) == 'xbool(False)'
    assert str(km.xbool(True)) == 'xbool(True)'
    assert repr(km.xbool(False)) == 'xbool(False)'
    assert repr(km.xbool(True)) == 'xbool(True)'
    assert eval(repr(km.xbool(False)), {'km': km}, {}) == km.xbool(False)
    assert eval(repr(km.xbool(True)), {'km': km}, {}) == km.xbool(True)
