import abc
import math

from attrs import frozen

from cirq_qubitization.surface_code.formulae import error_at, physical_qubits_per_tile


class DataBlock(metaclass=abc.ABCMeta):
    """A cost model for the data block of a surface code compilation.

    A surface code layout is segregated into qubits dedicated to magic state distillation
    and others dedicated to storing the actual data being processed. The latter area is
    called the data block, and we provide its costs here.
    """

    @abc.abstractmethod
    def footprint(self, n_algo_qubits: int) -> int:
        """The number of physical qubits used by the data block.

        Args:
            n_algo_qubits: The number of algorithm qubits whose data must be stored and
                accessed.
        """

    @abc.abstractmethod
    def data_error(self, n_algo_qubits: int, n_cycles: int, phys_err: float) -> float:
        """The error associated with storing data on `n_algo_qubits` for `n_cycles`."""


class TileDataBlock(metaclass=abc.ABCMeta):
    data_d: int = NotImplemented

    @abc.abstractmethod
    def n_tiles(self, n_algo_qubits: int) -> int:
        """Number of logical tiles including overhead."""

    def footprint(self, n_algo_qubits: int) -> int:
        """The number of physical qubits used by the data block."""
        return self.n_tiles(n_algo_qubits) * physical_qubits_per_tile(d=self.data_d)

    def data_error(self, n_algo_qubits: int, n_cycles: int, phys_err: float) -> float:
        """The error associated with storing data on `n_algo_qubits` for `n_cycles`."""
        data_cells = self.n_tiles(n_algo_qubits) * n_cycles
        return data_cells * error_at(phys_err, d=self.data_d)


@frozen
class SimpleDataBlock(TileDataBlock):
    """A simple data block that uses a fixed code distance and routing overhead.

    Args:
        data_d: The code distance `d` for protecting the qubits in the data block.
        routing_overhead: As an approximation, assume a number of routing or auxiliary
            qubits proportional to the number of algorithm qubits.
    """

    data_d: int
    routing_overhead: float = 0.5

    def n_tiles(self, n_algo_qubits: int) -> int:
        """Number of logical tiles including overhead.

        Note: the spreadsheet from the reference had a 50% overhead hardcoded for
        some of the cells using this quantity and variable (but set to 50% as default)
        for others.
        """
        return math.ceil((1 + self.routing_overhead) * n_algo_qubits)
