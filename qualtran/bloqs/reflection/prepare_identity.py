from functools import cached_property
from typing import Any, Dict, Tuple, TYPE_CHECKING

from attrs import frozen

from qualtran import QAny, Register, SoquetT
from qualtran.bloqs.select_and_prepare import PrepareOracle
from qualtran.symbolics.types import SymbolicInt

if TYPE_CHECKING:
    import quimb.tensor as qtn

    from qualtran import BloqBuilder, Soquet, SoquetT


@frozen
class PrepareIdentity(PrepareOracle):
    """An identity gate for reflecting around the zero state.


    Args:
        bitsize: the size of the register.

    Registers:
        x: The register to build the Identity operation on.
    """

    bitsize: SymbolicInt

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (Register('x', QAny(self.bitsize)),)

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        return ()

    def add_my_tensors(
        self,
        tn: 'qtn.TensorNetwork',
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        import quimb.tensor as qtn

        from qualtran._infra.composite_bloq import _flatten_soquet_collection
        from qualtran.simulation.tensor._tensor_data_manipulation import eye_tensor_for_signature

        data = eye_tensor_for_signature(self.signature)
        in_ind = _flatten_soquet_collection(incoming[reg.name] for reg in self.signature.lefts())
        out_ind = _flatten_soquet_collection(outgoing[reg.name] for reg in self.signature.rights())
        tn.add(qtn.Tensor(data=data, inds=out_ind + in_ind, tags=[self.pretty_name(), tag]))

    def build_composite_bloq(self, bb: 'BloqBuilder', x: 'Soquet') -> Dict[str, SoquetT]:
        return {'x': x}
