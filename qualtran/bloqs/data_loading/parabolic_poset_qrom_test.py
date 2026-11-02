# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");

import numpy as np
try:
    from qualtran.bloqs.data_loading.parabolic_poset_qrom import (
        ParabolicPosetQROM,
        compute_upper_ideal_rota_mobius_gauge,
        fast_zeta_xor_transform,
        parabolic_levi_lie_dim,
        select_graded_hamming_upper_ideal,
        verify_zero_clamping_half_range_msb_folding,
    )
except ImportError:
    try:
        from qualtran_dropin.parabolic_poset_qrom import (
            ParabolicPosetQROM,
            compute_upper_ideal_rota_mobius_gauge,
            fast_zeta_xor_transform,
            parabolic_levi_lie_dim,
            select_graded_hamming_upper_ideal,
            verify_zero_clamping_half_range_msb_folding,
        )
    except ImportError:
        from parabolic_poset_qrom import (
            ParabolicPosetQROM,
            compute_upper_ideal_rota_mobius_gauge,
            fast_zeta_xor_transform,
            parabolic_levi_lie_dim,
            select_graded_hamming_upper_ideal,
            verify_zero_clamping_half_range_msb_folding,
        )


def test_parabolic_poset_qrom_sub8bit_compression():
    qrom_256 = ParabolicPosetQROM(
        poset_order_ideal=(96, 20, 6, 6),
        levi_flag_splits=(30, 12, 10, 8),
        address_bits=8,
    )
    assert qrom_256.total_active_rows == 256
    assert qrom_256.effective_rank_capacity == 180 >= 143
    assert qrom_256.parabolic_group_dimension == 31920
    assert parabolic_levi_lie_dim(40, 10, 5, 5) == 52860

    qrom_248 = ParabolicPosetQROM(poset_order_ideal=(90, 20, 6, 4), address_bits=8)
    assert qrom_248.total_active_rows == 248
    assert qrom_248.effective_rank_capacity == 172 >= 143

    qrom_247 = ParabolicPosetQROM(poset_order_ideal=(89, 20, 6, 4), address_bits=8)
    assert qrom_247.total_active_rows == 247
    assert qrom_247.effective_rank_capacity == 171 >= 143


def test_upper_ideal_rota_mobius_zero_active_violations():
    rng = np.random.default_rng(42)
    address_bits = 8
    n_active = 248
    data = rng.integers(0, 2, size=(n_active, 8), dtype=np.int64)

    gauged_table, gauged_anf = compute_upper_ideal_rota_mobius_gauge(data, address_bits)
    np.testing.assert_array_equal(gauged_table[:n_active], data)
    np.testing.assert_array_equal(gauged_anf[n_active:], 0)

    zero_padded = np.zeros((256, 8), dtype=np.int64)
    zero_padded[:n_active] = data
    anf_zero = fast_zeta_xor_transform(zero_padded)
    assert int(np.sum(gauged_anf)) < int(np.sum(anf_zero))


def test_graded_hamming_upper_ideal_eliminates_all_high_degree_gates():
    rng = np.random.default_rng(7)
    for address_bits, n_active in [(10, 800), (11, 1220)]:
        n_slots = 1 << address_bits
        ideal_idx = select_graded_hamming_upper_ideal(address_bits, n_active)
        assert ideal_idx.shape == (n_slots - n_active,)

        high_deg_mask = np.array([int(m).bit_count() >= 7 for m in range(n_slots)])
        assert np.all(np.isin(np.where(high_deg_mask)[0], ideal_idx))

        raw_full = rng.integers(0, 2, size=(n_slots, 4), dtype=np.int64)
        _, gauged_anf = compute_upper_ideal_rota_mobius_gauge(
            raw_full, address_bits, upper_ideal_indices=ideal_idx
        )
        assert int(np.sum(gauged_anf[high_deg_mask])) == 0


if __name__ == "__main__":
    test_parabolic_poset_qrom_sub8bit_compression()
    test_upper_ideal_rota_mobius_zero_active_violations()
    test_graded_hamming_upper_ideal_eliminates_all_high_degree_gates()
    print("[PASS] parabolic_poset_qrom_test.py: all 3 test suites passed!")
