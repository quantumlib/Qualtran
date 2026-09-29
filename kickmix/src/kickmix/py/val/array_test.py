import kickmix as km
import pytest


def test_qcarray_empty():
    assert km.array() == km.array([])


def test_qcarray_bool():
    c = km.array([False, True])
    assert c._UNSTABLE_internal_values() == {'common_type': 0, 'len': 2, 'offset': 0, 'stride': 1}
    assert str(c) == "km.array([False, True])"
    assert c[0] == False
    assert c[1] == True
    assert list(c) == [False, True]


def test_qcarray_bool_values():
    c = km.array([False, True, 0, 1])
    assert c[0] == False
    assert c[1] == True
    assert c[2] == False
    assert c[3] == True

    with pytest.raises(TypeError, match="Expected a q | b | bool"):
        km.array([""])
    with pytest.raises(TypeError, match="Expected a q | b | bool"):
        km.array([object()])
    with pytest.raises(TypeError, match="Expected a q | b | bool"):
        km.array([()])
    with pytest.raises(TypeError, match="Expected a q | b | bool"):
        km.array([2])
    with pytest.raises(TypeError, match="Expected a q | b | bool"):
        km.array([None])


def test_qcarray_qubit():
    c = km.array([km.q(k) for k in range(10)])
    assert c._UNSTABLE_internal_values() == {'common_type': 2, 'len': 10, 'offset': 0, 'stride': 1}
    assert str(c) == "km.array([q0, q1, q2, q3, q4, q5, q6, q7, q8, q9])"
    assert c[3].id == 3
    assert c[3] == km.q(3)


def test_qcarray_bit():
    c = km.array([km.b(k) for k in range(10)])
    assert c._UNSTABLE_internal_values() == {'common_type': 1, 'len': 10, 'offset': 0, 'stride': 1}
    assert str(c) == "km.array([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])"
    assert c[3].id == 3
    assert c[3] == km.b(3)


def test_qcarray_slice():
    vs = [km.q(k) for k in range(100)]
    c = km.array(vs)
    assert list(c) == vs
    assert list(c[5:]) == vs[5:]
    assert list(c[14:]) == vs[14:]
    assert list(c[0::2]) == vs[0::2]
    assert list(c[0:8:2]) == vs[0:8:2]
    assert list(c[3:8:2]) == vs[3:8:2]
    assert list(c[14::2]) == vs[14::2]
    assert list(c[::-1]) == vs[::-1]
    assert list(c[5000:]) == vs[5000:]
    assert list(c[1::2][2::3]) == vs[1::2][2::3]


def test_qcarray_mixed():
    vs = [km.q(k) for k in range(10)] + [False] + [km.b(k) for k in range(12)] + [True]
    c = km.array(vs)
    assert list(c) == vs
    assert c._UNSTABLE_internal_values() == {
        'common_type': 128,
        'len': 24,
        'offset': 0,
        'stride': 1,
    }
    assert list(c[5:]) == vs[5:]
    assert c[5:]._UNSTABLE_internal_values() == {
        'common_type': 128,
        'len': 19,
        'offset': 5,
        'stride': 1,
    }
    assert list(c[14:]) == vs[14:]
    assert c[14:]._UNSTABLE_internal_values() == {
        'common_type': 128,
        'len': 10,
        'offset': 14,
        'stride': 1,
    }
    assert list(c[14::2]) == vs[14::2]
    assert c[14::2]._UNSTABLE_internal_values() == {
        'common_type': 128,
        'len': 5,
        'offset': 14,
        'stride': 2,
    }
    assert list(c[::-1]) == vs[::-1]
    assert c[::-1]._UNSTABLE_internal_values() == {
        'common_type': 128,
        'len': 24,
        'offset': 23,
        'stride': -1,
    }
    assert list(c[5000:]) == vs[5000:]
    assert c[5000:]._UNSTABLE_internal_values() == {
        'common_type': 128,
        'len': 0,
        'offset': 24,
        'stride': 1,
    }

    assert list(c[1::2][2::3]) == vs[1::2][2::3]


def test_qcarray_add():
    assert km.array([]) + km.array([]) == km.array([])
    assert km.array([km.q(0)]) + km.array([]) == km.array([km.q(0)])
    assert km.array([]) + km.array([km.q(0)]) == km.array([km.q(0)])
    assert km.array([km.q(0)]) + km.array([km.b(1)]) == km.array([km.q(0), km.b(1)])

    items1 = []
    items2 = []
    for k in range(99):
        items1.append(km.q(k))
        items1.append(km.b(k))
        items1.append(k % 2)
    for k in range(53):
        items2.append(km.q(k + 100))
        items2.append(km.b(k + 100))
        items2.append(k % 3 == 0)
    assert km.array(items1) + km.array(items2) == km.array(items1 + items2)
