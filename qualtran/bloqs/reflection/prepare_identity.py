from functools import cached_property
from typing import Dict, Iterable, Sequence, Tuple, TYPE_CHECKING

from attrs import field, frozen

from qualtran import bloq_example, BloqDocSpec, QAny, Register, SoquetT
from qualtran.bloqs.basic_gates import Identity
from qualtran.bloqs.basic_gates.on_each import OnEach
from qualtran.bloqs.select_and_prepare import PrepareOracle
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.symbolics.types import SymbolicInt

if TYPE_CHECKING:
    from qualtran import BloqBuilder, Soquet, SoquetT


def _to_tuple(x: Iterable[SymbolicInt]) -> Sequence[SymbolicInt]:
    """mypy compatible attrs converter for Reflection.cvs and bitsizes"""
    return tuple(x)


@frozen
class PrepareIdentity(PrepareOracle):
    """An identity gate PrepareOracle.

    This is mainly used as an intermediate bloq as input for the
    ReflectionUsingPrepare to produce a reflection about zero.

    Args:
        bitsize: the size of the register.

    Registers:
        x: The register to build the Identity operation on.
    """

    bitsizes: Tuple[SymbolicInt, ...] = field(converter=_to_tuple)

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return tuple(Register(f'x{i}', QAny(b)) for i, b in enumerate(self.bitsizes))

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        return ()

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'Soquet') -> Dict[str, SoquetT]:
        for i, b in enumerate(self.bitsizes):
            label = f'x{i}'
            reg = soqs[label]
            q = bb.add(OnEach(b, Identity()), q=reg)
            soqs[label] = q
        return soqs


@bloq_example(generalizer=ignore_split_join)
def _prepare_identity() -> PrepareIdentity:
    prepare = PrepareIdentity(bitsizes=(10, 4, 1))
    return prepare


_PREPARE_IDENTITY_DOC = BloqDocSpec(
    bloq_cls=PrepareIdentity,
    import_line='from qualtran.bloqs.reflection.prepare_identity import PrepareIdentity',
    examples=(_prepare_identity,),
)
