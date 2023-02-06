import abc

from attrs import frozen


@frozen
class MagicStateCount:
    """The number of magic states needed for a computation.

    Each `count` excludes the resources needed to perform operations captured in other
    magic state counts.

    Args:
        t_count: The number of T operations that need to be performed.
        ccz_count: The number of Toffoli or CCZ operations that need to be performed.
    """

    t_count: int
    ccz_count: int

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
        """The physical qubit spacial footprint of the factory."""

    @abc.abstractmethod
    def n_cycles(self, n_magic: MagicStateCount) -> int:
        """The number of cycles (time) required to produce the requested number of magic states."""

    @abc.abstractmethod
    def distillation_error(self, n_magic: MagicStateCount, phys_err: float) -> float:
        """The total error expected from distilling magic states with a given physical error rate."""
