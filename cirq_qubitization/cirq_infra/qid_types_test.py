import cirq_qubitization.cirq_infra as cqi


def test_clean_qubits():
    q = cqi.CleanQubit(1)
    assert q.id == 1
    assert q.dimension == 2
    assert str(q) == '_c1'
    assert repr(q) == 'CleanQubit(1)'


def test_borrow_qubits():
    q = cqi.BorrowableQubit(10)
    assert q.id == 10
    assert q.dimension == 2
    assert str(q) == '_b10'
    assert repr(q) == 'BorrowableQubit(10)'
