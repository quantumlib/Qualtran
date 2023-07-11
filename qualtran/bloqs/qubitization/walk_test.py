import qualtran.testing as qlt_testing


def _make_walk():
    from qualtran.bloqs.qubitization.prepare import BlackBoxPrepare, DummyPrepare
    from qualtran.bloqs.qubitization.select_bloq import BlackBoxSelect, DummySelect
    from qualtran.bloqs.qubitization.walk import Walk

    select = BlackBoxSelect(DummySelect())
    prepare = BlackBoxPrepare(DummyPrepare())

    return Walk(select=select, prepare=prepare)


def test_walk():
    qlt_testing.assert_valid_bloq_decomposition(_make_walk())
