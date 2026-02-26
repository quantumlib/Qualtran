from qualtran import QUInt
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.mcmt import MultiTargetCNOT
from qualtran.bloqs.arithmetic.addition import Add
from qualtran.quirk_interop.bloq_to_quirk import bloq_to_quirk


def test_bloq_to_quirk():
    bloq_to_quirk(Add(QUInt(5)))
    bloq_to_quirk(MultiTargetCNOT(5))


def test_bloq_to_quirk_on_atomic():
    bloq_to_quirk(Toffoli())
