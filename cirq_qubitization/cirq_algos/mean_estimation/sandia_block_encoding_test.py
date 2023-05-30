import cirq_qubitization as cq
from cirq_qubitization.cirq_algos.mean_estimation import sandia_block_encoding


def test_sandia_synthesizer():
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

    sandia_synthesizer = sandia_block_encoding.SandiaSynthesizer(
        stopping_system=sts, evolution_precision=1.0e-3, evolution_time=2
    )

    cost = cq.t_complexity(sandia_synthesizer)
    assert 3.3e12 <= cost.t <= 3.4e12
    assert cost.rotations == 0
    assert cost.clifford == 0
