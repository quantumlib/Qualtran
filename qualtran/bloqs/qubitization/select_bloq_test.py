from typing import List

import qualtran.testing as qlt_testing


def _make_select():
    from attrs import frozen

    from qualtran import Register, Signature
    from qualtran.bloqs.qubitization.select_bloq import Select

    @frozen
    class MySelect(Select):
        __doc__ = Select.__doc__

        @property
        def control_registers(self) -> List[Register]:
            return []

        @property
        def selection_registers(self) -> List[Register]:
            return list(Signature.build(p=32, q=32, spin=1))

        @property
        def system_register(self) -> Register:
            return Register(name='psi', bitsize=128)

    return MySelect()


def _make_black_box_select():
    from qualtran.bloqs.qubitization.select_bloq import BlackBoxSelect, DummySelect

    return BlackBoxSelect(DummySelect())


def test_black_box_select():
    qlt_testing.assert_valid_bloq_decomposition(_make_black_box_select())
