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
    assert km.q(0) != False

    assert repr(q) == "q(5)"
    assert q.id == 5


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
    assert km.b(0) != False

    assert repr(b) == "b(5)"
    assert b.id == 5


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
    assert km.xb(0) != False

    assert repr(b) == "xb(5)"
    assert b.id == 5


def test_xbool():
    assert km.xbool(False) != False
    assert km.xbool(False) == km.xbool(False)
    assert km.xbool(True) == km.xbool(True)
    assert km.xbool(False) != km.xbool(True)
    assert km.xbool(False) != object()
    assert km.xbool(False) != ''
    assert km.xbool(False) != True
    assert hash(km.xbool(True)) == hash(km.xbool(True))
    assert hash(km.xbool(False)) == hash(km.xbool(False))
    assert str(km.xbool(False)) == 'xbool(False)'
    assert str(km.xbool(True)) == 'xbool(True)'
    assert repr(km.xbool(False)) == 'xbool(False)'
    assert repr(km.xbool(True)) == 'xbool(True)'
