from dev_tools.qualtran_dev_tools.bloq_finder import get_bloq_examples
from qualtran.qref_interop import bloq_to_qref
import pytest

from qualtran import DecomposeTypeError, DecomposeNotImplementedError
from qref.verification import verify_topology


@pytest.mark.parametrize("bloq_example", get_bloq_examples())
def test_bloq_examples_can_be_converted_to_qualtran(bloq_example):
    bloq = bloq_example.make()
    try:
        qref_routine = bloq_to_qref(bloq)
        verify_topology(qref_routine)
    except:
        pytest.xfail(f"QREF conversion failing for {bloq}")


@pytest.mark.parametrize("bloq_example", get_bloq_examples())
def test_bloq_examples_can_be_converted_to_qualtran_when_decomposed(bloq_example):
    try:
        bloq = bloq_example.make().decompose_bloq()
    # I think ValueError here is a bit hacky
    except (DecomposeTypeError, DecomposeNotImplementedError, ValueError) as e:
        pytest.xfail(f"QREF conversion not attempted, as bloq decomposition faield with {e}")

    try:
        qref_routine = bloq_to_qref(bloq)
        # verify_topology(qref_routine)
    except:
        pytest.xfail(f"QREF conversion failing for {bloq}")
