//! CLI integration tests for the qlt_fastsim binary.
//!
//! Tests batch execution, output formatting, error reporting,
//! flag parsing, and edge cases through the command-line interface.

use std::io::Write;
use std::process::{Command, Stdio};

// ============================================================================
// 1. Basic batch execution
// ============================================================================

#[test]
fn test_cli_batch_negate() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_qlt_fastsim"))
        .arg("example_qlts/negate.qlt")
        .arg("Negate")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn qlt_fastsim");

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(b"5\n10\n-3\n")
            .expect("Failed to write to stdin");
    }

    let output = child.wait_with_output().expect("Failed to read output");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout, "# phase_exponent x\n0 -5\n0 -10\n0 3\n");
}

#[test]
fn test_cli_batch_cswap() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_qlt_fastsim"))
        .arg("example_qlts/cswap.qlt")
        .arg("CSwap")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn qlt_fastsim");

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(b"1 7 3\n0 7 3\n")
            .expect("Failed to write to stdin");
    }

    let output = child.wait_with_output().expect("Failed to read output");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout, "# phase_exponent ctrl x y\n0 1 3 7\n0 0 7 3\n");
}

// ============================================================================
// 2. Empty stdin
// ============================================================================

#[test]
fn test_cli_empty_stdin() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_qlt_fastsim"))
        .arg("example_qlts/negate.qlt")
        .arg("Negate")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn qlt_fastsim");

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(b"").expect("Failed to write to stdin");
    }

    let output = child.wait_with_output().expect("Failed to read output");
    assert!(output.status.success());
    // Even with no input data, the header line is always emitted.
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout, "# phase_exponent x\n");
}

// ============================================================================
// 3. Whitespace handling
// ============================================================================

#[test]
fn test_cli_excessive_whitespace() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_qlt_fastsim"))
        .arg("example_qlts/cswap.qlt")
        .arg("CSwap")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn qlt_fastsim");

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(b"   1 \t  7 \t 3   \n")
            .expect("Failed to write to stdin");
    }

    let output = child.wait_with_output().expect("Failed to read output");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout, "# phase_exponent ctrl x y\n0 1 3 7\n");
}

// ============================================================================
// 3b. Comment handling in input
// ============================================================================

#[test]
fn test_cli_comment_lines_skipped() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_qlt_fastsim"))
        .arg("example_qlts/cswap.qlt")
        .arg("CSwap")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn qlt_fastsim");

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(b"# This is a comment\n1 7 3\n# Another comment\n0 7 3\n")
            .expect("Failed to write to stdin");
    }

    let output = child.wait_with_output().expect("Failed to read output");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout, "# phase_exponent ctrl x y\n0 1 3 7\n0 0 7 3\n");
}

#[test]
fn test_cli_inline_comments() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_qlt_fastsim"))
        .arg("example_qlts/cswap.qlt")
        .arg("CSwap")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn qlt_fastsim");

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(b"1 7 3  # swap case\n0 7 3  # no-swap case\n")
            .expect("Failed to write to stdin");
    }

    let output = child.wait_with_output().expect("Failed to read output");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout, "# phase_exponent ctrl x y\n0 1 3 7\n0 0 7 3\n");
}

// ============================================================================
// 4. Error: wrong number of input values
// ============================================================================

#[test]
fn test_cli_wrong_arg_count() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_qlt_fastsim"))
        .arg("example_qlts/cswap.qlt")
        .arg("CSwap")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn qlt_fastsim");

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(b"1 7\n").expect("Failed to write to stdin");
    }

    let output = child.wait_with_output().expect("Failed to read output");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Expected 3 input values"));
}

#[test]
fn test_cli_batch_strict_error_handling() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_qlt_fastsim"))
        .arg("example_qlts/negate.qlt")
        .arg("Negate")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn qlt_fastsim");

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(b"5\n5 10\n")
            .expect("Failed to write to stdin");
    }

    let output = child.wait_with_output().expect("Failed to read output");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Expected 1 input values"));
}

// ============================================================================
// 5. Error: overflow value for bit width
// ============================================================================

#[test]
fn test_cli_overflow_ctrl_bit() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_qlt_fastsim"))
        .arg("example_qlts/cswap.qlt")
        .arg("CSwap")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn qlt_fastsim");

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(b"2 7 3\n")
            .expect("Failed to write to stdin");
    }

    let output = child.wait_with_output().expect("Failed to read output");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("exceeds maximum value for bit width"));
}

// ============================================================================
// 6. Flag parsing
// ============================================================================

#[test]
fn test_cli_unknown_flag() {
    let output = Command::new(env!("CARGO_BIN_EXE_qlt_fastsim"))
        .arg("--invalid-flag")
        .arg("example_qlts/negate.qlt")
        .arg("Negate")
        .output()
        .expect("Failed to run qlt_fastsim");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Unknown flag: --invalid-flag"));
}

#[test]
fn test_cli_missing_positional_args() {
    let output = Command::new(env!("CARGO_BIN_EXE_qlt_fastsim"))
        .arg("example_qlts/negate.qlt")
        .output()
        .expect("Failed to run qlt_fastsim");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Usage:"));
}

#[test]
fn test_cli_no_timing_flag() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_qlt_fastsim"))
        .arg("--no-timing")
        .arg("example_qlts/negate.qlt")
        .arg("Negate")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn qlt_fastsim");

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(b"5\n").expect("Failed to write to stdin");
    }
    let output = child.wait_with_output().expect("Failed to read output");
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "# phase_exponent x\n0 -5\n"
    );
    assert!(!String::from_utf8_lossy(&output.stderr).contains("[timing]"));
}

#[test]
fn test_cli_timing_flag() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_qlt_fastsim"))
        .arg("--timing")
        .arg("example_qlts/negate.qlt")
        .arg("Negate")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn qlt_fastsim");

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(b"5\n").expect("Failed to write to stdin");
    }
    let output = child.wait_with_output().expect("Failed to read output");
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("[timing] parse_qlt:"));
    assert!(stderr.contains("[timing] compile:"));
    assert!(stderr.contains("[timing] parse_inputs:"));
    assert!(stderr.contains("[timing] execute:"));
}

// ============================================================================
// 7. Large-width batch execution
// ============================================================================

#[test]
fn test_cli_batch_large_width() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_qlt_fastsim"))
        .arg("example_qlts/negate-large.qlt")
        .arg("Negate")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn qlt_fastsim");

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(b"11111111111111111111\n22222222222222222222\n33333333333333333333\n")
            .expect("Failed to write to stdin");
    }

    let output = child.wait_with_output().expect("Failed to read output");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout, "# phase_exponent x\n0 -11111111111111111111\n0 -22222222222222222222\n0 -33333333333333333333\n");
}

// ============================================================================
// 8. Composite ALU CLI
// ============================================================================

#[test]
fn test_cli_composite_alu() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_qlt_fastsim"))
        .arg("example_qlts/composite_alu.qlt")
        .arg("CompositeALU")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn qlt_fastsim");

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(b"123 456\n")
            .expect("Failed to write to stdin");
    }

    let output = child.wait_with_output().expect("Failed to read output");
    assert!(
        output.status.success(),
        "Failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    // The header should contain phase_exponent and register names.
    let lines: Vec<&str> = stdout.lines().collect();
    assert!(lines[0].starts_with("# phase_exponent"));
    assert!(lines[0].contains("a"));
    assert!(lines[0].contains("b"));
}

#[test]
fn test_cli_composite_alu_known_values() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_qlt_fastsim"))
        .arg("example_qlts/composite_alu.qlt")
        .arg("CompositeALU")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn qlt_fastsim");

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(b"0 1\n1 0\n1 1\n0 0\n")
            .expect("Failed to write to stdin");
    }

    let output = child.wait_with_output().expect("Failed to read output");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        stdout,
        "# phase_exponent a b\n0.5 0 1\n2.5 1 1\n1 1 0\n0 0 0\n"
    );
}

#[test]
fn test_cli_compile_only() {
    let output = Command::new(env!("CARGO_BIN_EXE_qlt_fastsim"))
        .arg("--compile-only")
        .arg("example_qlts/negate.qlt")
        .output()
        .expect("Failed to run qlt_fastsim");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("CompiledModule"));
    assert!(stdout.contains("Subroutine 0"));
    assert!(stdout.contains("Subroutine 1"));
    assert!(stdout.contains("body Impl"));
}
