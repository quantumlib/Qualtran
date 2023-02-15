import math
from typing import Optional, Sequence, Tuple

from attrs import frozen

from cirq_qubitization.surface_code.data_block import DataBlock, TileDataBlock
from cirq_qubitization.surface_code.formulae import error_at, physical_qubits_per_tile
from cirq_qubitization.surface_code.magic_state_factory import MagicStateCount, MagicStateFactory
from cirq_qubitization.surface_code.physical_cost import PhysicalCost


@frozen
class AzureSpaceEffFactory(MagicStateFactory):
    """15-to-1 space efficient factory."""

    d: int
    n_copies: int

    @classmethod
    def from_acceptance_probability(
        cls, *, d: int, phys_err: float, acceptance_probability: float = 0.99
    ):
        t_error = phys_err
        data_error = error_at(phys_err=phys_err, d=d)

        # rejection probability for one factory, from table VI
        reject_prob_1 = 15 * t_error + 356 * data_error
        # prob(all get rejected) = reject_prob_1 ** n
        # Invert with logs:
        n_factory = math.log(1 - acceptance_probability) / math.log(reject_prob_1)
        n_factory = math.ceil(n_factory)

        return cls(d=d, n_copies=n_factory)

    def footprint(self) -> int:
        # Table VI
        tiles = 20
        return tiles * physical_qubits_per_tile(d=self.d) * self.n_copies

    def n_cycles(self, n_magic: MagicStateCount) -> int:
        # Table VI
        return 13 * self.d * n_magic.all_t_count()

    def distillation_error(self, n_magic: MagicStateCount, phys_err: float) -> float:
        # Table VI
        t_error = phys_err
        data_error = error_at(phys_err=phys_err, d=self.d)
        # TODO: the data error scales with n_copies, right?
        per_t = 35 * t_error**2 + 7.1 * data_error * self.n_copies
        return n_magic.all_t_count() * per_t


class MultiLevelFactory(MagicStateFactory):
    def __init__(self, factories: Sequence[MagicStateFactory]):
        self.factories = factories

    def footprint(self) -> int:
        # azure re-uses space; footprint is the max over levels
        return max(fac.footprint() for fac in self.factories)

    def n_cycles(self, n_magic: MagicStateCount) -> int:
        # azure doesn't re-use time; n_cycles add
        per_t = sum(fac.n_cycles(MagicStateCount(t_count=1)) for fac in self.factories)
        return n_magic.all_t_count() * per_t

    def distillation_error(self, n_magic: MagicStateCount, phys_err: float) -> float:
        errors = [phys_err]  # level 0 injected at physical error rate

        for fac, level in zip(self.factories, range(1, len(self.factories) + 1)):
            # `in_err` should be interpreted as "input error".
            new_error = fac.distillation_error(n_magic, phys_err=errors[level - 1])
            errors.append(new_error)

        return errors[-1]


@frozen
class AzureDataBlock(TileDataBlock):
    data_d: int

    def n_tiles(self, n_algo_qubits: int) -> int:
        # equation (D1)
        # use "fast data block" with double-wide algorithmic qubits
        n_algo_tiles = 2 * n_algo_qubits
        # fast data block access hallways, although I don't know where this comes from
        n_ancilla_tiles = math.ceil(math.sqrt(8 * n_algo_qubits)) + 1

        return n_algo_tiles + n_ancilla_tiles


def scratch(m_meas, m_rot, m_t, m_toffoli, n_rot_layers, eps_synth):
    """


    Args:
        m_meas: $M_\mathrm{meas}$ number of Pauli measurements of the input algorithm
        m_rot:  $M_R$ number of single-qubit rotations
        m_t:  $M_T$ number of T gates
        m_toffoli: $M_\mathrm{Tof}$ number of Toffoli gates
        n_rot_layers: $D_R$ the number of non-Clifford layers in which there is at least
            one arbitrary angle rotation (i.e. exclude layers consisting entirely of T/Tof).
        eps_synth: $\eps_\mathrm{syn}$ target error budget for synthesis.

    Returns:

    """
    # logical time steps, minimum (D3)
    # assumes consumption limited computation, i.e. excess T production
    # why is there a factor of 3 for toffli? why not four? as described in fig 10 (d)
    a = 0.53
    b = 5.3
    synth_per_rot = math.ceil(a * math.log2(m_rot / eps_synth) + b)
    min_n_cycles = (m_meas + m_rot + m_t) + synth_per_rot * n_rot_layers + 3 * m_toffoli

    # number of T states $M$, eq (D4)
    # compile tofflis into 4T
    m = synth_per_rot * m_rot + 4 * m_toffoli + m_t

    # runtime:
    # they say choose c >= c_min; Then that sets the number of factories.
    slowdown_factor: float = 10
    c = math.ceil(min_n_cycles * slowdown_factor)  # c is now just a number of cycles
    # it needs to kick back into factory params


def stuff(
    *,
    n_magic: MagicStateCount,
    n_algo_qubits: int,
    phys_err: float = 1e-3,
    error_budget: Optional[float] = 1e-2,
    cycle_time_us: float = 1.0,
    slowdown_factor: float = 10,
    factory: MagicStateFactory = None,
    data_block: DataBlock = None,
):
    d = 29  # TODO

    # TODO: figure out levels
    factory = AzureSpaceEffFactory.from_acceptance_probability(d=d, phys_err=phys_err)
    distillation_error = factory.distillation_error(n_magic, phys_err=phys_err)
    print(f'{distillation_error=}')
    data_block = AzureDataBlock(data_d=d)

    # logical time steps, minimum (D3)
    # Simplified: ignore algorithmic measurements; ignore rotations
    min_n_logirounds = n_magic.t_count + 3 * n_magic.ccz_count

    # number of T states $M$, eq (D4)
    # simplified: ignore rotations
    m = n_magic.all_t_count()

    # runtime:
    # they say choose c >= c_min; Then that sets the number of factories.
    n_cycles = math.ceil(min_n_logirounds * slowdown_factor * d)

    # todo: accept = factory.acceptance_prob()
    n_factories = factory.n_cycles(n_magic) / (0.99 * n_cycles)
    n_factories = math.ceil(n_factories)
    print(f"Using {n_factories=}")

    data_error = data_block.data_error(n_algo_qubits, n_cycles, phys_err=phys_err)

    if n_cycles < factory.n_cycles(MagicStateCount(t_count=1)):
        raise ValueError("The runtime of the algorithm is too short, increase slowdown_factor")

    return PhysicalCost(
        failure_prob=distillation_error + data_error,
        footprint=factory.footprint() + data_block.footprint(n_algo_qubits=n_algo_qubits),
        duration_hr=(cycle_time_us * n_cycles) / (1_000_000 * 60 * 60),
    )


if __name__ == '__main__':
    RES = stuff(n_magic=MagicStateCount(t_count=1e6), n_algo_qubits=100)
    pass
