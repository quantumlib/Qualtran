"""Implementation of the kinetic energy encoder oracle"""
from functools import cached_property
from typing import Sequence

import cirq
import numpy as np
from attrs import frozen

import cirq_qubitization as cq
from cirq_qubitization.cirq_algos.mean_estimation.sandia_block_encoding import StoppingPowerSystem
from cirq_qubitization.cirq_algos.select_and_prepare import SelectOracle


@frozen
class KineticEnergyEncoder(SelectOracle):
    """Cost to build the encoder oracle

    This code oracle performs the following
    $$
    Y\\vert \\omega \\rangle \\vert 0\\rangle_{b} \\vert 0 \\rangle_{c} =
        \\vert w\\rangle \\vert y(\\omega) \\rangle_{b} \\vert 0 \\rangle_{c}
    $$

    where $$|\\omega \\rangle$$ is the selection register and $$|y(\\omega)\\rangle_{b}$$ takes on a b-bit
    value of the kinetic energy operator is the target register
    """

    stopping_system: StoppingPowerSystem

    def control_registers(self) -> cq.Registers:
        return cq.Registers([])

    def selection_registers(self) -> cq.SelectionRegisters:
        return cq.SelectionRegisters.build(
            selection=(3 * self.stopping_system.n_n, 2 ** (3 * self.stopping_system.n_n))
        )

    def target_registers(self) -> cq.Registers:
        """projectile momentum registers x/y/z all of size n_n qubits"""
        return cq.Registers.build(target=3 * self.stopping_system.n_n)

    def encoded_register(self) -> cq.Registers:
        # TODO: Is this needed? Should this be target register instead?
        return cq.Registers.build(encoded_reg=self.encoded_register_size)

    @cached_property
    def p_xyz_max(self) -> int:
        return 2 ** (
            self.stopping_system.n_n - 1
        )  # minus 1 in exponent is for +- range for 2^nn values

    @cached_property
    def num_bits_kp_max(self) -> int:
        ke_prefactor = (
            2 * np.pi**2 / self.stopping_system.Omega ** (2 / 3) / self.stopping_system.mpr
        )
        max_ke = ke_prefactor * 3 * self.p_xyz_max**2
        return int(np.floor(np.log2(max_ke) + 1))

    @cached_property
    def num_bits_kmean(self) -> int:
        ret = np.log2(self.stopping_system.get_kmean_in_au() ** 2 / (2 * self.stopping_system.mpr))
        return int(np.floor(ret)) + 1

    @cached_property
    def num_bits_cross(self) -> int:
        ret = np.log2(
            self.stopping_system.get_kmean_in_au()
            * (2 * np.pi * self.p_xyz_max / self.stopping_system.Omega ** (1 / 3))
            / self.stopping_system.mpr
        )
        return int(np.floor(ret)) + 1

    @cached_property
    def encoded_register_size(self) -> int:
        ret = np.log2(3 * 2 ** max(self.num_bits_kp_max, self.num_bits_kmean, self.num_bits_cross))
        return int(np.floor(ret)) + 1

    def decompose_from_registers(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        # No decompose specified.
        return NotImplemented

    def calculate_toffoli_costs(self) -> int:
        n_n = self.stopping_system.n_n
        # step 1
        ke_of_p = 3 * self.stopping_system.n_n**2 + self.stopping_system.n_n + 1
        # nc = np.floor(np.log2(2 * np.pi**2 / self.stopping_system.mpr / self.stopping_system.Omega**(2/3))) + 1
        # ke_of_p += 2 * self.num_bits_kp_max * nc - self.num_bits_kp_max

        # step 2
        ke_of_cross = (
            4 * (2 * self.num_bits_kmean * n_n - self.num_bits_kmean) + 2 * self.num_bits_kmean + 1
        )

        # step 3
        ke_mean_constant = 2 * self.encoded_register_size + 1

        return ke_of_p + ke_of_cross + ke_mean_constant

    def _t_complexity_(self) -> cq.TComplexity:
        return cq.TComplexity(t=4 * self.calculate_toffoli_costs())
