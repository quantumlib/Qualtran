import qualtran.testing as qt_testing


def _make_reflect():
    from qualtran.bloqs.qubitization.prepare import BlackBoxPrepare, DummyPrepare
    from qualtran.bloqs.qubitization.reflect import Reflect

    prepare = BlackBoxPrepare(DummyPrepare())
    return Reflect(prepare=prepare)


def test_reflect():
    qt_testing.assert_valid_bloq_decomposition(_make_reflect())
