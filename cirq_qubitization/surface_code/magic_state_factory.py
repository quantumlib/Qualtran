import abc
import math
from typing import Sequence, Tuple

from attrs import frozen

from cirq_qubitization.surface_code.formulae import n_cycles_per_quop, physical_qubits_per_tile


@frozen
class MagicStateCount:
    """The number of magic states needed for a computation.

    Each `count` excludes the resources needed to perform operations captured in other
    magic state counts.

    Args:
        t_count: The number of T operations that need to be performed.
        ccz_count: The number of Toffoli or CCZ operations that need to be performed.
    """

    t_count: int = 0
    ccz_count: int = 0

    def all_t_count(self) -> int:
        """The T count needed to do all magic operations with T only."""
        return self.t_count + 4 * self.ccz_count


class MagicStateFactory(metaclass=abc.ABCMeta):
    """A cost model for the magic state distillation factory of a surface code compilation.

    A surface code layout is segregated into qubits dedicated to magic state distillation
    and storing the data being processed. The former area is called the magic state distillation
    factory, and we provide its costs here.
    """

    @abc.abstractmethod
    def footprint(self) -> int:
        """The number of physical qubits used by the magic state factory."""

    @abc.abstractmethod
    def n_cycles(self, n_magic: MagicStateCount) -> int:
        """The number of cycles (time) required to produce the requested number of magic states."""

    @abc.abstractmethod
    def distillation_error(self, n_magic: MagicStateCount, phys_err: float) -> float:
        """The total error expected from distilling magic states with a given physical error rate."""

    @abc.abstractmethod
    def rejection_prob(self, phys_err: float) -> float:
        """asdf"""


@frozen
class SimpleTFactory(MagicStateFactory, metaclass=abc.ABCMeta):
    d: int

    @property
    @abc.abstractmethod
    def quop_dimensions(self) -> Tuple[float, float, float]:
        ...

    @property
    def quop_volume(self) -> float:
        w, h, t = self.quop_dimensions
        return w * h * t

    @property
    def quop_area(self) -> float:
        w, h, t = self.quop_dimensions
        return w * h

    @property
    def quop_time(self) -> float:
        w, h, t = self.quop_dimensions
        return t

    def footprint(self) -> int:
        return math.ceil(self.quop_area * physical_qubits_per_tile(d=self.d))

    def n_cycles_for_t(self) -> float:
        return self.quop_time * n_cycles_per_quop(d=self.d)

    def n_cycles(self, n_magic: MagicStateCount) -> int:
        return math.ceil(self.n_cycles_for_t() * n_magic.all_t_count())

    @abc.abstractmethod
    def error_for_t(self, inp_err: float) -> float:
        ...

    def distillation_error(self, n_magic: MagicStateCount, phys_err: float) -> float:
        return self.error_for_t(inp_err=phys_err) * n_magic.all_t_count()


class NFactories(MagicStateFactory):
    def __init__(self, factory: MagicStateFactory, n: int):
        self._factory = factory
        self._n = n

    def footprint(self) -> int:
        """N factories take up N-times as many physical qubits."""
        return self._n * self._factory.footprint()

    def n_cycles(self, n_magic: MagicStateCount) -> int:
        """N factories can produce `n_magic` states in 1/N cycles (approximately)."""
        return math.ceil(self._factory.n_cycles(n_magic) / self._n)

    def distillation_error(self, n_magic: MagicStateCount, phys_err: float) -> float:
        """Total error is unaffected by space/time trade-off."""
        return self._factory.distillation_error(n_magic, phys_err)


class MultiLevelFactory(MagicStateFactory):
    def __init__(self, factories: Sequence[MagicStateFactory], re_use='time'):
        self._factories = factories

        if re_use not in ['space', 'time']:
            raise ValueError("`re_use` must be 'space' or 'time'.")
        self._re_use = re_use

    def footprint(self) -> int:
        if self._re_use == 'space':
            return max(fac.footprint() for fac in self._factories)
        elif self._re_use == 'time':
            return sum(fac.footprint() for fac in self._factories)
        raise ValueError()

    def n_cycles(self, n_magic: MagicStateCount) -> int:
        if self._re_use == 'space':
            return sum(fac.n_cycles(n_magic) for fac in self._factories)
        elif self._re_use == 'time':
            return max(fac.n_cycles(n_magic) for fac in self._factories)
        raise ValueError()

    def distillation_error(self, n_magic: MagicStateCount, phys_err: float) -> float:
        errors = [phys_err]  # level 0 injected at physical error rate

        for fac, level in zip(self._factories, range(1, len(self._factories) + 1)):
            # `in_err` should be interpreted as "input error".
            new_error = fac.distillation_error(n_magic, phys_err=errors[level - 1])
            errors.append(new_error)

        return errors[-1]
