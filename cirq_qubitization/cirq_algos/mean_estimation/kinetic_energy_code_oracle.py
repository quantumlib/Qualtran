"""Implementation of the kinetic energy code oracle"""
from typing import Sequence

import cirq
import numpy

import cirq_qubitization as cq
from cirq_qubitization.cirq_algos.mean_estimation.sandia_block_encoding import StoppingPowerSystem
from cirq_qubitization.cirq_algos.select_and_prepare import SelectOracle


class KineticEnergyCodeOracle(SelectOracle):
    """Cost to build code oracle

    This code oracle performs the following
    $$
    Y\\vert \\omega \\rangle \\vert 0\\rangle_{b} \\vert 0 \\rangle_{c} = \\vert w\\rangle \\vert y(\\omega) \\rangle_{b} \\vert 0 \\rangle_{c}
    $$

    where $$|\\omega \\rangle$$ is the selection register and $$|y(\\omega)\\rangle_{b}$$ takes on a b-bit
    value of the kinetic energy operator is the target register
    """

    def __init__(self, stopping_system: StoppingPowerSystem):
        super().__init__()
        self.stopping_system = stopping_system
        self._max_ke: float = None
        self._ke_prefactor: float = None
        self._num_bits_kp_max: int = None
        self._num_bits_kmean: int = None
        self._num_bits_cross: int = None

    def control_registers(self) -> cq.Registers:
        return cq.Registers([])

    def selection_registers(self) -> cq.SelectionRegisters:
        return cq.SelectionRegisters.build(
            selection=((3 * self.stopping_system.n_n, 2 ** (3 * self.stopping_system.n_n)))
        )

    def target_registers(self) -> cq.Registers:
        """projectile momentum registers x/y/z all of size n_n qubits"""
        return cq.Registers.build(target=3 * self.stopping_system.n_n)

    def encoded_register(self) -> cq.SelectionRegisters:
        return cq.Registers.build([self.get_encoded_register_size()])

    def get_encoded_register_size(self,) -> int:
        p_xyz_max = 2 ** (
            self.stopping_system.n_n - 1
        )  # minus 1 in exponent is for +- range for 2^nn values
        self._ke_prefactor = (
            2 * numpy.pi**2 / self.stopping_system.Omega ** (2 / 3) / self.stopping_system.mpr
        )

        self._max_ke = self._ke_prefactor * 3 * p_xyz_max**2
        self._num_bits_kp_max = numpy.floor(numpy.log2(self._max_ke) + 1)

        self._num_bits_kmean = (
            numpy.floor(
                numpy.log2(
                    self.stopping_system.get_kmean_in_au() ** 2 / (2 * self.stopping_system.mpr)
                )
            )
            + 1
        )

        self._num_bits_cross = (
            numpy.floor(
                numpy.log2(
                    self.stopping_system.get_kmean_in_au()
                    * (2 * numpy.pi * p_xyz_max / self.stopping_system.Omega ** (1 / 3))
                    / self.stopping_system.mpr
                )
            )
            + 1
        )

        return int(
            numpy.floor(
                numpy.log2(
                    3 * 2 ** max(self._num_bits_kp_max, self._num_bits_kmean, self._num_bits_cross)
                )
            )
            + 1
        )

    def decompose_from_registers(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        # No decompose specified.
        return NotImplemented

    def calculate_toffoli_costs(self) -> int:
        n_n = self.stopping_system.n_n
        # step 1
        ke_of_p = 3 * self.stopping_system.n_n**2 + self.stopping_system.n_n + 1
        # nc = numpy.floor(numpy.log2(2 * numpy.pi**2 / self.stopping_system.mpr / self.stopping_system.Omega**(2/3))) + 1
        # ke_of_p += 2 * self._num_bits_kp_max * nc - self._num_bits_kp_max

        # step 2
        ke_of_cross = (
            4 * (2 * self._num_bits_kmean * n_n - self._num_bits_kmean)
            + 2 * self._num_bits_kmean
            + 1
        )

        # step 3
        ke_mean_constant = 2 * self.get_encoded_register_size() + 1

        return ke_of_p + ke_of_cross + ke_mean_constant

    def _t_complexity_(self) -> cq.TComplexity:
        return cq.TComplexity(t=4 * self.calculate_toffoli_costs())
