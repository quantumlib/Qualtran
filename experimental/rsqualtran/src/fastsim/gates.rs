//! Built-in gate implementations for the qlt_fastsim simulator.
//!
//! Each gate operates on a `SimState` (bit vector + phase exponent) given specific bit indices.

use crate::fastsim::vm::SimState;

/// Apply XGate: flip the single qubit.
pub fn apply_x(state: &mut SimState, bits: &[usize]) {
    assert_eq!(bits.len(), 1, "XGate expects exactly 1 bit");
    state.flip_bit(bits[0]);
}

/// Apply CNOT: if ctrl is 1, flip target.
pub fn apply_cnot(state: &mut SimState, ctrl_bits: &[usize], target_bits: &[usize]) {
    assert_eq!(ctrl_bits.len(), 1, "CNOT expects 1 ctrl bit");
    assert_eq!(target_bits.len(), 1, "CNOT expects 1 target bit");
    if state.get_bit(ctrl_bits[0]) {
        state.flip_bit(target_bits[0]);
    }
}

/// Apply Toffoli: if both ctrl bits are 1, flip target.
pub fn apply_toffoli(state: &mut SimState, ctrl_bits: &[usize], target_bits: &[usize]) {
    assert_eq!(ctrl_bits.len(), 2, "Toffoli expects exactly 2 ctrl bits");
    assert_eq!(target_bits.len(), 1, "Toffoli expects 1 target bit");
    if state.get_bit(ctrl_bits[0]) && state.get_bit(ctrl_bits[1]) {
        state.flip_bit(target_bits[0]);
    }
}

/// Apply And (forward): compute (ctrl[0] == cv[0]) && (ctrl[1] == cv[1]) into a newly
/// allocated ancilla bit. The `cv` (control values) parameter specifies which bit value
/// activates each control: `true` means bit=1 activates (default), `false` means bit=0
/// activates.
/// Returns the index of the newly allocated bit.
pub fn apply_and(state: &mut SimState, ctrl_bits: &[usize], cv: &[bool; 2]) -> usize {
    assert_eq!(ctrl_bits.len(), 2, "And expects exactly 2 ctrl bits");
    let result = (state.get_bit(ctrl_bits[0]) == cv[0]) && (state.get_bit(ctrl_bits[1]) == cv[1]);
    state.push_bit(result)
}

/// Apply And_dag (adjoint): verify the target bit equals
/// (ctrl[0] == cv[0]) && (ctrl[1] == cv[1]), then deallocate it.
/// The `cv` (control values) parameter specifies which bit value activates each control:
/// `true` means bit=1 activates (default), `false` means bit=0 activates.
pub fn apply_and_dag(state: &mut SimState, ctrl_bits: &[usize], target_bit: usize, cv: &[bool; 2]) {
    assert_eq!(ctrl_bits.len(), 2, "And_dag expects exactly 2 ctrl bits");
    let expected = (state.get_bit(ctrl_bits[0]) == cv[0]) && (state.get_bit(ctrl_bits[1]) == cv[1]);
    let actual = state.get_bit(target_bit);
    if actual != expected {
        panic!(
            "And_dag assertion failed: target bit {} is {}, expected {} (ctrl[0]={} & ctrl[1]={}, cv={:?})",
            target_bit,
            actual,
            expected,
            state.get_bit(ctrl_bits[0]),
            state.get_bit(ctrl_bits[1]),
            cv
        );
    }
    // Mark as deallocated (set to false). We don't actually remove to keep indices stable.
    state.set_bit(target_bit, false);
}

/// Apply TwoBitCSwap: if ctrl is 1, swap x and y.
pub fn apply_twobitcswap(
    state: &mut SimState,
    ctrl_bits: &[usize],
    x_bits: &[usize],
    y_bits: &[usize],
) {
    assert_eq!(ctrl_bits.len(), 1, "TwoBitCSwap expects 1 ctrl bit");
    assert_eq!(x_bits.len(), 1, "TwoBitCSwap expects 1 x bit");
    assert_eq!(y_bits.len(), 1, "TwoBitCSwap expects 1 y bit");
    if state.get_bit(ctrl_bits[0]) {
        let tmp = state.get_bit(x_bits[0]);
        let y_val = state.get_bit(y_bits[0]);
        state.set_bit(x_bits[0], y_val);
        state.set_bit(y_bits[0], tmp);
    }
}

/// Apply ZGate: add 1.0 to phase exponent if qubit is 1.
pub fn apply_z(state: &mut SimState, bits: &[usize]) {
    assert_eq!(bits.len(), 1, "ZGate expects exactly 1 bit");
    if state.get_bit(bits[0]) {
        state.phase_exponent += 1.0;
    }
}

/// Apply SGate: add 0.5 to phase exponent if qubit is 1.
pub fn apply_s(state: &mut SimState, bits: &[usize]) {
    assert_eq!(bits.len(), 1, "SGate expects exactly 1 bit");
    if state.get_bit(bits[0]) {
        state.phase_exponent += 0.5;
    }
}

/// Apply TGate: add 0.25 to phase exponent if qubit is 1.
pub fn apply_t(state: &mut SimState, bits: &[usize]) {
    assert_eq!(bits.len(), 1, "TGate expects exactly 1 bit");
    if state.get_bit(bits[0]) {
        state.phase_exponent += 0.25;
    }
}

/// Apply CZGate: add 1.0 to phase exponent if both qubits are 1.
pub fn apply_cz(state: &mut SimState, q1_bits: &[usize], q2_bits: &[usize]) {
    assert_eq!(q1_bits.len(), 1, "CZGate expects 1 q1 bit");
    assert_eq!(q2_bits.len(), 1, "CZGate expects 1 q2 bit");
    if state.get_bit(q1_bits[0]) && state.get_bit(q2_bits[0]) {
        state.phase_exponent += 1.0;
    }
}

/// Apply CCZGate: add 1.0 to phase exponent if all three qubits are 1.
pub fn apply_ccz(state: &mut SimState, q1_bits: &[usize], q2_bits: &[usize], q3_bits: &[usize]) {
    assert_eq!(q1_bits.len(), 1, "CCZGate expects 1 q1 bit");
    assert_eq!(q2_bits.len(), 1, "CCZGate expects 1 q2 bit");
    assert_eq!(q3_bits.len(), 1, "CCZGate expects 1 q3 bit");
    if state.get_bit(q1_bits[0]) && state.get_bit(q2_bits[0]) && state.get_bit(q3_bits[0]) {
        state.phase_exponent += 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(bits: Vec<bool>) -> SimState {
        SimState::from_bits(bits)
    }

    #[test]
    fn test_x_gate_flip() {
        let mut state = make_state(vec![false]);
        apply_x(&mut state, &[0]);
        assert!(state.get_bit(0));
        apply_x(&mut state, &[0]);
        assert!(!state.get_bit(0));
    }

    #[test]
    fn test_cnot_ctrl_zero() {
        let mut state = make_state(vec![false, false]);
        apply_cnot(&mut state, &[0], &[1]);
        assert!(!state.get_bit(1)); // ctrl=0, target unchanged
    }

    #[test]
    fn test_cnot_ctrl_one() {
        let mut state = make_state(vec![true, false]);
        apply_cnot(&mut state, &[0], &[1]);
        assert!(state.get_bit(1)); // ctrl=1, target flipped
    }

    #[test]
    fn test_toffoli() {
        // ctrl=[0, 0], target=0 -> unchanged
        let mut state = make_state(vec![false, false, false]);
        apply_toffoli(&mut state, &[0, 1], &[2]);
        assert!(!state.get_bit(2));

        // ctrl=[1, 0], target=0 -> unchanged
        let mut state = make_state(vec![true, false, false]);
        apply_toffoli(&mut state, &[0, 1], &[2]);
        assert!(!state.get_bit(2));

        // ctrl=[1, 1], target=0 -> flipped to 1
        let mut state = make_state(vec![true, true, false]);
        apply_toffoli(&mut state, &[0, 1], &[2]);
        assert!(state.get_bit(2));

        // ctrl=[1, 1], target=1 -> flipped to 0
        let mut state = make_state(vec![true, true, true]);
        apply_toffoli(&mut state, &[0, 1], &[2]);
        assert!(!state.get_bit(2));
    }

    #[test]
    fn test_and_forward() {
        // Both controls true
        let mut state = make_state(vec![true, true]);
        let idx = apply_and(&mut state, &[0, 1], &[true, true]);
        assert_eq!(idx, 2);
        assert!(state.get_bit(2));

        // One control false
        let mut state = make_state(vec![true, false]);
        let idx = apply_and(&mut state, &[0, 1], &[true, true]);
        assert_eq!(idx, 2);
        assert!(!state.get_bit(2));

        // Both controls false
        let mut state = make_state(vec![false, false]);
        let idx = apply_and(&mut state, &[0, 1], &[true, true]);
        assert_eq!(idx, 2);
        assert!(!state.get_bit(2));
    }

    #[test]
    fn test_and_forward_cv_0_0() {
        // Both controls are negative (cv=[false, false]).
        // ctrl=[false, false] -> result = true
        let mut state = make_state(vec![false, false]);
        let idx = apply_and(&mut state, &[0, 1], &[false, false]);
        assert_eq!(idx, 2);
        assert!(state.get_bit(2));

        // ctrl=[true, true] -> result = false
        let mut state = make_state(vec![true, true]);
        let idx = apply_and(&mut state, &[0, 1], &[false, false]);
        assert_eq!(idx, 2);
        assert!(!state.get_bit(2));
    }

    #[test]
    fn test_and_forward_cv_0_1() {
        // Mixed controls (cv=[false, true]).
        // ctrl=[false, true] -> result = true
        let mut state = make_state(vec![false, true]);
        let idx = apply_and(&mut state, &[0, 1], &[false, true]);
        assert_eq!(idx, 2);
        assert!(state.get_bit(2));

        // ctrl=[true, true] -> result = false (ctrl[0]=true != cv[0]=false)
        let mut state = make_state(vec![true, true]);
        let idx = apply_and(&mut state, &[0, 1], &[false, true]);
        assert_eq!(idx, 2);
        assert!(!state.get_bit(2));

        // ctrl=[false, false] -> result = false (ctrl[1]=false != cv[1]=true)
        let mut state = make_state(vec![false, false]);
        let idx = apply_and(&mut state, &[0, 1], &[false, true]);
        assert_eq!(idx, 2);
        assert!(!state.get_bit(2));
    }

    #[test]
    fn test_and_forward_cv_1_0() {
        // Mixed controls (cv=[true, false]).
        // ctrl=[true, false] -> result = true
        let mut state = make_state(vec![true, false]);
        let idx = apply_and(&mut state, &[0, 1], &[true, false]);
        assert_eq!(idx, 2);
        assert!(state.get_bit(2));
    }

    #[test]
    fn test_and_dag() {
        let mut state = make_state(vec![true, true, true]); // ctrl0=1, ctrl1=1, target=1
        apply_and_dag(&mut state, &[0, 1], 2, &[true, true]);
        assert!(!state.get_bit(2)); // deallocated (set to false)
    }

    #[test]
    #[should_panic(expected = "And_dag assertion failed")]
    fn test_and_dag_mismatch() {
        let mut state = make_state(vec![true, false, true]); // ctrl0=1, ctrl1=0, target=1 (should be 0)
        apply_and_dag(&mut state, &[0, 1], 2, &[true, true]);
    }

    #[test]
    fn test_and_dag_cv_0_1() {
        // And_dag with cv=[false, true].
        // ctrl=[false, true] and target=true -> expected=true, should succeed
        let mut state = make_state(vec![false, true, true]);
        apply_and_dag(&mut state, &[0, 1], 2, &[false, true]);
        assert!(!state.get_bit(2)); // deallocated
    }

    #[test]
    #[should_panic(expected = "And_dag assertion failed")]
    fn test_and_dag_cv_0_1_mismatch() {
        // And_dag with cv=[false, true].
        // ctrl=[true, true] and target=true -> expected=false, should panic
        let mut state = make_state(vec![true, true, true]);
        apply_and_dag(&mut state, &[0, 1], 2, &[false, true]);
    }

    #[test]
    fn test_twobitcswap_ctrl_one() {
        let mut state = make_state(vec![true, false, true]); // ctrl=1, x=0, y=1
        apply_twobitcswap(&mut state, &[0], &[1], &[2]);
        assert!(state.get_bit(1)); // x = former y = 1
        assert!(!state.get_bit(2)); // y = former x = 0
    }

    #[test]
    fn test_twobitcswap_ctrl_zero() {
        let mut state = make_state(vec![false, false, true]); // ctrl=0, x=0, y=1
        apply_twobitcswap(&mut state, &[0], &[1], &[2]);
        assert!(!state.get_bit(1)); // unchanged
        assert!(state.get_bit(2)); // unchanged
    }

    #[test]
    fn test_z_gate() {
        let mut state = make_state(vec![true]);
        apply_z(&mut state, &[0]);
        assert!((state.phase_exponent - 1.0).abs() < 1e-10);

        let mut state = make_state(vec![false]);
        apply_z(&mut state, &[0]);
        assert!((state.phase_exponent - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_s_gate() {
        let mut state = make_state(vec![true]);
        apply_s(&mut state, &[0]);
        assert!((state.phase_exponent - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_t_gate() {
        let mut state = make_state(vec![true]);
        apply_t(&mut state, &[0]);
        assert!((state.phase_exponent - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_cz_gate() {
        let mut state = make_state(vec![true, true]);
        apply_cz(&mut state, &[0], &[1]);
        assert!((state.phase_exponent - 1.0).abs() < 1e-10);

        let mut state = make_state(vec![true, false]);
        apply_cz(&mut state, &[0], &[1]);
        assert!((state.phase_exponent - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_ccz_gate() {
        let mut state = make_state(vec![true, true, true]);
        apply_ccz(&mut state, &[0], &[1], &[2]);
        assert!((state.phase_exponent - 1.0).abs() < 1e-10);

        let mut state = make_state(vec![true, true, false]);
        apply_ccz(&mut state, &[0], &[1], &[2]);
        assert!((state.phase_exponent - 0.0).abs() < 1e-10);
    }
}
