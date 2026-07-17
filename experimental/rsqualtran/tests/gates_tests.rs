//! Direct gate operation tests.
//!
//! Verifies correctness of individual quantum gates: X, CNOT, Z, S, T,
//! CZ, CCZ, And/And_dag, and TwoBitCSwap.

use _rsqlt::fastsim::vm::SimState;

// ============================================================================
// X Gate
// ============================================================================

#[test]
fn test_x_gate_flip() {
    let mut s1 = SimState::from_bits(vec![false]);
    _rsqlt::fastsim::gates::apply_x(&mut s1, &[0]);
    assert!(s1.get_bit(0), "X on |0⟩ should give |1⟩");

    let mut s2 = SimState::from_bits(vec![true]);
    _rsqlt::fastsim::gates::apply_x(&mut s2, &[0]);
    assert!(!s2.get_bit(0), "X on |1⟩ should give |0⟩");
}

// ============================================================================
// CNOT Gate
// ============================================================================

#[test]
fn test_cnot_ctrl_zero_no_flip() {
    let mut state = SimState::from_bits(vec![false, true]);
    _rsqlt::fastsim::gates::apply_cnot(&mut state, &[0], &[1]);
    assert!(state.get_bit(1), "CNOT with ctrl=0 should not flip target");
}

#[test]
fn test_cnot_ctrl_one_flips() {
    let mut state = SimState::from_bits(vec![true, true]);
    _rsqlt::fastsim::gates::apply_cnot(&mut state, &[0], &[1]);
    assert!(!state.get_bit(1), "CNOT with ctrl=1 should flip target");
}

#[test]
fn test_cnot_repeated_toggling() {
    let mut state = SimState::from_bits(vec![true, false]);
    for _ in 0..10000 {
        _rsqlt::fastsim::gates::apply_cnot(&mut state, &[0], &[1]);
    }
    assert!(
        !state.get_bit(1),
        "10000 CNOTs (even count) should return target to original"
    );
}

// ============================================================================
// Phase Gates: Z, S, T
// ============================================================================

#[test]
fn test_z_gate_on_one() {
    let mut state = SimState::from_bits(vec![true]);
    _rsqlt::fastsim::gates::apply_z(&mut state, &[0]);
    assert!(
        (state.phase_exponent - 1.0).abs() < 1e-10,
        "Z on |1⟩ should add 1.0, got {}",
        state.phase_exponent
    );
    assert!(state.get_bit(0), "Z should not modify bits");
}

#[test]
fn test_z_gate_on_zero() {
    let mut state = SimState::from_bits(vec![false]);
    _rsqlt::fastsim::gates::apply_z(&mut state, &[0]);
    assert!(state.phase_exponent.abs() < 1e-10, "Z on |0⟩ should add 0");
    assert!(!state.get_bit(0));
}

#[test]
fn test_s_gate_on_one() {
    let mut state = SimState::from_bits(vec![true]);
    _rsqlt::fastsim::gates::apply_s(&mut state, &[0]);
    assert!(
        (state.phase_exponent - 0.5).abs() < 1e-10,
        "S on |1⟩ should add 0.5, got {}",
        state.phase_exponent
    );
    assert!(state.get_bit(0), "S should not modify bits");
}

#[test]
fn test_s_gate_on_zero() {
    let mut state = SimState::from_bits(vec![false]);
    _rsqlt::fastsim::gates::apply_s(&mut state, &[0]);
    assert!(state.phase_exponent.abs() < 1e-10);
}

#[test]
fn test_t_gate_on_one() {
    let mut state = SimState::from_bits(vec![true]);
    _rsqlt::fastsim::gates::apply_t(&mut state, &[0]);
    assert!(
        (state.phase_exponent - 0.25).abs() < 1e-10,
        "T on |1⟩ should add 0.25, got {}",
        state.phase_exponent
    );
    assert!(state.get_bit(0), "T should not modify bits");
}

#[test]
fn test_t_gate_on_zero() {
    let mut state = SimState::from_bits(vec![false]);
    _rsqlt::fastsim::gates::apply_t(&mut state, &[0]);
    assert!(state.phase_exponent.abs() < 1e-10);
}

#[test]
fn test_phase_exponent_gates_dont_modify_bits() {
    let mut state = SimState::from_bits(vec![true, false, true, true]);
    let original_bits = state.bits.clone();

    _rsqlt::fastsim::gates::apply_z(&mut state, &[0]);
    _rsqlt::fastsim::gates::apply_s(&mut state, &[2]);
    _rsqlt::fastsim::gates::apply_t(&mut state, &[3]);

    assert_eq!(
        state.bits, original_bits,
        "Phase gates should never modify bits"
    );
    assert!(
        (state.phase_exponent - 1.75).abs() < 1e-10,
        "Expected phase_exponent 1.75, got {}",
        state.phase_exponent
    );
}

#[test]
fn test_phase_exponent_accumulates() {
    let mut state = SimState::from_bits(vec![true]);
    _rsqlt::fastsim::gates::apply_z(&mut state, &[0]); // +1.0
    _rsqlt::fastsim::gates::apply_s(&mut state, &[0]); // +0.5
    _rsqlt::fastsim::gates::apply_t(&mut state, &[0]); // +0.25
    assert!(
        (state.phase_exponent - 1.75).abs() < 1e-10,
        "Accumulated phase_exponent should be 1.75, got {}",
        state.phase_exponent
    );
}

// ============================================================================
// CZ Gate
// ============================================================================

#[test]
fn test_cz_both_one() {
    let mut state = SimState::from_bits(vec![true, true]);
    _rsqlt::fastsim::gates::apply_cz(&mut state, &[0], &[1]);
    assert!((state.phase_exponent - 1.0).abs() < 1e-10);
    assert!(state.get_bit(0) && state.get_bit(1));
}

#[test]
fn test_cz_one_zero() {
    let mut state = SimState::from_bits(vec![true, false]);
    _rsqlt::fastsim::gates::apply_cz(&mut state, &[0], &[1]);
    assert!(state.phase_exponent.abs() < 1e-10);
}

// ============================================================================
// CCZ Gate
// ============================================================================

#[test]
fn test_ccz_all_one() {
    let mut state = SimState::from_bits(vec![true, true, true]);
    _rsqlt::fastsim::gates::apply_ccz(&mut state, &[0], &[1], &[2]);
    assert!((state.phase_exponent - 1.0).abs() < 1e-10);
}

#[test]
fn test_ccz_partial() {
    for combo in &[
        (true, true, false),
        (true, false, true),
        (false, true, true),
        (true, false, false),
        (false, true, false),
        (false, false, true),
        (false, false, false),
    ] {
        let mut state = SimState::from_bits(vec![combo.0, combo.1, combo.2]);
        _rsqlt::fastsim::gates::apply_ccz(&mut state, &[0], &[1], &[2]);
        assert!(
            state.phase_exponent.abs() < 1e-10,
            "CCZ on ({},{},{}) should give phase_exponent 0, got {}",
            combo.0,
            combo.1,
            combo.2,
            state.phase_exponent
        );
    }
}

// ============================================================================
// And / And_dag
// ============================================================================

#[test]
fn test_and_gate() {
    let mut s1 = SimState::from_bits(vec![true, true]);
    let idx1 = _rsqlt::fastsim::gates::apply_and(&mut s1, &[0, 1], &[true, true]);
    assert!(s1.get_bit(idx1), "1 & 1 = 1");

    let mut s2 = SimState::from_bits(vec![true, false]);
    let idx2 = _rsqlt::fastsim::gates::apply_and(&mut s2, &[0, 1], &[true, true]);
    assert!(!s2.get_bit(idx2), "1 & 0 = 0");
}

#[test]
fn test_and_and_dag_roundtrip() {
    let mut state = SimState::from_bits(vec![true, true]);
    let idx = _rsqlt::fastsim::gates::apply_and(&mut state, &[0, 1], &[true, true]);
    assert_eq!(state.len(), 3);
    assert!(state.get_bit(idx));

    _rsqlt::fastsim::gates::apply_and_dag(&mut state, &[0, 1], idx, &[true, true]);
    assert!(
        !state.get_bit(idx),
        "And_dag should set bit to false (deallocate)"
    );
}

#[test]
#[should_panic(expected = "And_dag assertion failed")]
fn test_and_dag_fails_on_mismatch() {
    let mut state = SimState::from_bits(vec![true, false, true]);
    _rsqlt::fastsim::gates::apply_and_dag(&mut state, &[0, 1], 2, &[true, true]);
}

#[test]
fn test_and_dag_preserves_phase_exponent() {
    let mut state = SimState::from_bits(vec![true, true]);
    state.phase_exponent = 1000000.0;
    let idx = _rsqlt::fastsim::gates::apply_and(&mut state, &[0, 1], &[true, true]);
    assert_eq!(state.len(), 3);
    assert!(state.get_bit(idx));
    _rsqlt::fastsim::gates::apply_and_dag(&mut state, &[0, 1], idx, &[true, true]);
    assert!(!state.get_bit(idx));
    assert!(
        (state.phase_exponent - 1000000.0).abs() < 1e-10,
        "Phase exponent should remain untouched"
    );
}

// ============================================================================
// TwoBitCSwap
// ============================================================================

#[test]
fn test_twobitcswap_all_combinations() {
    for ctrl in [false, true] {
        for x in [false, true] {
            for y in [false, true] {
                let mut state = SimState::from_bits(vec![ctrl, x, y]);
                _rsqlt::fastsim::gates::apply_twobitcswap(&mut state, &[0], &[1], &[2]);

                assert_eq!(state.get_bit(0), ctrl, "ctrl should not change");
                if ctrl {
                    assert_eq!(state.get_bit(1), y, "With ctrl=1, x should become y");
                    assert_eq!(state.get_bit(2), x, "With ctrl=1, y should become x");
                } else {
                    assert_eq!(state.get_bit(1), x, "With ctrl=0, x unchanged");
                    assert_eq!(state.get_bit(2), y, "With ctrl=0, y unchanged");
                }
                assert!(
                    state.phase_exponent.abs() < 1e-10,
                    "TwoBitCSwap should not affect phase exponent"
                );
            }
        }
    }
}

// ============================================================================
// Large qubit index
// ============================================================================

#[test]
fn test_gate_at_max_qubit_index() {
    let mut state = SimState::new(65536);
    state.extend_false(65536);
    _rsqlt::fastsim::gates::apply_x(&mut state, &[65535]);
    _rsqlt::fastsim::gates::apply_z(&mut state, &[65535]);
    assert!(state.get_bit(65535), "Bit 65535 should be flipped to true");
    assert!(
        (state.phase_exponent - 1.0).abs() < 1e-10,
        "Phase exponent should be 1.0"
    );
}
