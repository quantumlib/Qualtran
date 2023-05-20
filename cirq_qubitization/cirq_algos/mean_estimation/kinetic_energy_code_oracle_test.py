import numpy

from cirq_qubitization.cirq_algos.mean_estimation import (
    kinetic_energy_code_oracle,
    sandia_block_encoding,
)


def test_projectile_ke_encoder_oracle():
    Omega = 2419.682821036357
    num_bits_momenta = 6  # Number of bits in each direction for momenta
    num_bits_nuclei_momenta = 8
    eps_total = 1e-3  # Total allowable error
    num_bits_nu = 6  # extra bits for nu
    num_bits_nuc = 8  # extra bits for nuclear positions
    num_nuclei = 216  # L
    num_elec = 218  # eta
    projectile_mass = 1836 * 4
    projectile_charge = 2  # Helium
    projectile_velocity = 4
    nbr = 20

    sts = sandia_block_encoding.StoppingPowerSystem(
        n_p=num_bits_momenta,
        n_n=num_bits_nuclei_momenta,
        eta=num_elec,
        Omega=Omega,
        eps=eps_total,
        nMc=num_bits_nuc,
        nbr=nbr,
        L=num_nuclei,
        zeta=projectile_charge,
        mpr=projectile_mass,
        kmean=projectile_velocity,
    )

    encoder = kinetic_energy_code_oracle.KineticEnergyCodeOracle(stopping_system=sts)
    assert numpy.isclose(encoder.get_encoded_register_size(), 18)
    assert numpy.isclose(encoder.calculate_toffoli_costs(), 1231)
