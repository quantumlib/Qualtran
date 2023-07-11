import qualtran.testing as qt_testing


def _make_phase_esimation():
    from qualtran.bloqs.qubitization.phase_estimation import PhaseEstimation
    from qualtran.bloqs.qubitization.prepare import BlackBoxPrepare, DummyPrepare
    from qualtran.bloqs.qubitization.select_bloq import BlackBoxSelect, DummySelect
    from qualtran.bloqs.qubitization.walk import Walk

    walk = Walk(select=BlackBoxSelect(DummySelect()), prepare=BlackBoxPrepare(DummyPrepare()))

    return PhaseEstimation(walk, m=4)


def test_phase_estimation():
    qt_testing.assert_valid_bloq_decomposition(_make_phase_esimation())
