import abc
import math
from typing import Callable, Sequence, Tuple

from attrs import frozen

from cirq_qubitization.surface_code.formulae import (
    error_at,
    n_cycles_per_quop,
    physical_qubits_per_tile,
    quop_err,
)
from cirq_qubitization.surface_code.magic_state_factory import (
    MagicStateCount,
    MagicStateFactory,
    SimpleTFactory,
)


@frozen
class StateInjectionFactory(SimpleTFactory):
    """State injection.

    We use 20 state injection units (arranged in a 2*2 spatial grid and 5 attempts in time).
    """

    d: int  # l1d/2

    @property
    def quop_dimensions(self) -> Tuple[float, float, float]:
        """2x2 grid of units at distance `d`.

        For packing purposes, the distance should be half the distance of the level-1
        factory. That means our 2x2 grid of units fits in one level-1 tile.
        """
        return (2, 2, 5)

    def rejection_prob(self, phys_err: float) -> float:
        units = 2 * 2 * 5  # 20 copies
        return 0.5**units

    def error_for_t(self, phys_err: float) -> float:
        # (distillation error)
        # TODO: include topological error? 100 "unit cells"
        # This would be one distance 7 tile for 1 cycle
        distill_err = phys_err
        topo_error = self.quop_volume * quop_err(phys_err=phys_err, d=self.d)
        return distill_err + topo_error


@frozen
class LatticeSurgeryFactory(SimpleTFactory):
    """15-to-1 standard distiller, using dimensions from the first paper."""

    d: int

    @property
    def quop_dimensions(self):
        return (3, 8, 6.5)

    def rejection_prob(self, in_err: float) -> float:
        return 15 * in_err

    def error_for_t(self, in_err: float) -> float:
        # TODO: include topological error

        distill_err = 35 * in_err**3
        topo_error = self.quop_volume * quop_err(in_err, d=self.d)

        return distill_err + topo_error


class MultiLevelFactory(MagicStateFactory):
    def __init__(self, factories: Sequence[SimpleTFactory]):
        self._factories = factories

    def footprint(self) -> int:
        return sum(fac.footprint() for fac in self._factories)

    def n_cycles(self, n_magic: MagicStateCount) -> int:
        n_cycles_per = max(fac.n_cycles_for_t() for fac in self._factories)
        return math.ceil(n_cycles_per * n_magic.all_t_count())

    def distillation_error(self, n_magic: MagicStateCount, phys_err: float) -> float:
        errors = [phys_err]

        for fac in self._factories:
            errors.append(fac.distillation_error(n_magic, phys_err=errors[-1]))

        print(errors)
        return errors[-1]

    def rejection_prob(self, phys_err: float) -> float:
        # TODO: multiple copies
        pass
