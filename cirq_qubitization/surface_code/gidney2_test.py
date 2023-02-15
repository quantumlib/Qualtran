from cirq_qubitization.surface_code.gidney2 import (
    MultiLevelFactory,
    SimpleTFactory,
    StateInjectionFactory,
)
from cirq_qubitization.surface_code.magic_state_factory import MagicStateCount


def test():
    stf = SimpleTFactory(d=15)
    stf.distillation_error(n_magic=MagicStateCount(t_count=10))
