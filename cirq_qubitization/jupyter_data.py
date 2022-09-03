import dataclasses
import cirq_qubitization
from types import ModuleType
from typing import Callable, List

from cirq_qubitization.gate_with_registers import GateWithRegisters
import cirq_qubitization.jupyter_factory_functions as cq_jd


@dataclasses.dataclass
class GateNbData:
    """Notebook data for a Gate factory.

    Attributes:
        factory: A factory function that produces a gate. Its source code will be rendered
            to the notebook. See the `cirq_qubitization.jupyter_factory_functions` module.
        draw_vertical: Whether to render `vertical=True` in the rendered call to
            `display_gate_and_compilation`
    """
    factory: Callable[[], GateWithRegisters]
    draw_vertical: bool = False

    @property
    def cqid(self):
        """A globally-unique id for this `GateNbData` to identify existing notebook cells."""
        return self.factory.__name__


@dataclasses.dataclass
class NotebookData:
    """Data for rendering a jupyter notebook for a given module.

    Attributes:
        title: The title of the notebook
        module: The module it documents. This is used to render the module docstring
            at the top of the notebook.
        gates: A list of `GateNbData`.
    """
    title: str
    module: ModuleType
    gates: List[GateNbData]


NOTEBOOK_DATA = {
    'qrom': NotebookData(
        title='QROM',
        module=cirq_qubitization.qrom,
        gates=[GateNbData(cq_jd._make_QROM)]
    ),
    'swap_network': NotebookData(
        title='Swap Network',
        module=cirq_qubitization.swap_network,
        gates=[
            GateNbData(cq_jd._make_MultiTargetCSwap),
            GateNbData(cq_jd._make_MultiTargetCSwapApprox),
        ]
    ),
    'generic_subprepare': NotebookData(
        title='Sub-Prepare',
        module=cirq_qubitization.generic_subprepare,
        gates=[GateNbData(cq_jd._make_GenericSubPrepare)]
    )
}
