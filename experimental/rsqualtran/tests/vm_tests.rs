//! VM executor integration tests.
//!
//! Tests the full parse → compile → execute pipeline for negate and cswap,
//! exhaustive correctness verification, involution properties, large-bitwidth
//! execution, error handling, and edge cases.

// ============================================================================
// Helpers
// ============================================================================

/// Helper: parse, compile, and execute negate with given input
fn run_negate_pipeline(input_val: i64) -> (f64, Vec<(String, String, String)>) {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let file_path = std::path::Path::new(&manifest_dir).join("example_qlts/negate.qlt");
    let source = std::fs::read_to_string(&file_path)
        .unwrap_or_else(|e| panic!("Failed to read {:?}: {}", file_path, e));

    let (module, errors) = _rsqlt::parser::parse_l1_module(&source);
    assert!(errors.is_empty(), "Parse errors: {:?}", errors);

    let compiled = _rsqlt::fastsim::compiler::compile(&module).expect("Compilation failed");

    let bits = _rsqlt::fastsim::vm::int_to_bits(input_val, 8);
    let input_values = vec![("x".to_string(), bits)];
    let mut sim = _rsqlt::fastsim::vm::VmSimulator::new(&compiled, "Negate")
        .expect("VmSimulator creation failed");
    sim.execute_run(&input_values).expect("Execution failed");
    let outputs = sim.extract_outputs();
    (sim.phase_exponent(), outputs)
}

/// Helper: parse, compile, and execute cswap with given inputs
fn run_cswap_pipeline(ctrl: i64, x: i64, y: i64) -> (f64, Vec<(String, String, String)>) {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let file_path = std::path::Path::new(&manifest_dir).join("example_qlts/cswap.qlt");
    let source = std::fs::read_to_string(&file_path)
        .unwrap_or_else(|e| panic!("Failed to read {:?}: {}", file_path, e));

    let (module, errors) = _rsqlt::parser::parse_l1_module(&source);
    assert!(errors.is_empty(), "Parse errors: {:?}", errors);

    let compiled = _rsqlt::fastsim::compiler::compile(&module).expect("Compilation failed");

    let input_values = vec![
        (
            "ctrl".to_string(),
            _rsqlt::fastsim::vm::int_to_bits(ctrl, 1),
        ),
        ("x".to_string(), _rsqlt::fastsim::vm::int_to_bits(x, 5)),
        ("y".to_string(), _rsqlt::fastsim::vm::int_to_bits(y, 5)),
    ];
    let mut sim = _rsqlt::fastsim::vm::VmSimulator::new(&compiled, "CSwap")
        .expect("VmSimulator creation failed");
    sim.execute_run(&input_values).expect("Execution failed");
    let outputs = sim.extract_outputs();
    (sim.phase_exponent(), outputs)
}

// ============================================================================
// 1. Basic execution
// ============================================================================

#[test]
fn test_negate_5() {
    let (phase_exponent, outputs) = run_negate_pipeline(5);
    assert_eq!(
        outputs.len(),
        1,
        "Expected 1 output register, got {}",
        outputs.len()
    );
    assert_eq!(outputs[0].0, "x");
    assert_eq!(
        outputs[0].1, "-5",
        "Negate(5) should be -5, got {}",
        outputs[0].1
    );
    assert!(
        phase_exponent.abs() < 1e-10,
        "Phase exponent should be 0, got {}",
        phase_exponent
    );
}

#[test]
fn test_cswap_swap() {
    let (_phase_exponent, outputs) = run_cswap_pipeline(1, 7, 3);
    let ctrl_val = &outputs.iter().find(|o| o.0 == "ctrl").unwrap().1;
    let x_val = &outputs.iter().find(|o| o.0 == "x").unwrap().1;
    let y_val = &outputs.iter().find(|o| o.0 == "y").unwrap().1;
    assert_eq!(ctrl_val, "1");
    assert_eq!(x_val, "3", "CSwap(ctrl=1, x=7, y=3) should give x=3");
    assert_eq!(y_val, "7", "CSwap(ctrl=1, x=7, y=3) should give y=7");
}

#[test]
fn test_cswap_no_swap() {
    let (_phase_exponent, outputs) = run_cswap_pipeline(0, 7, 3);
    let x_val = &outputs.iter().find(|o| o.0 == "x").unwrap().1;
    let y_val = &outputs.iter().find(|o| o.0 == "y").unwrap().1;
    assert_eq!(x_val, "7");
    assert_eq!(y_val, "3");
}

// ============================================================================
// 2. Exhaustive negate (all 256 8-bit inputs)
// ============================================================================

#[test]
fn test_negate_exhaustive_256() {
    for i in 0u16..256 {
        let input = i as i64;
        let (phase_exponent, outputs) = run_negate_pipeline(input);

        let expected_unsigned = ((256u16.wrapping_sub(i)) % 256) as u64;
        let expected_signed = if expected_unsigned >= 128 {
            expected_unsigned as i64 - 256
        } else {
            expected_unsigned as i64
        };

        assert_eq!(
            outputs[0].1,
            expected_signed.to_string(),
            "Negate({}) expected {} but got {}",
            input,
            expected_signed,
            outputs[0].1
        );
        assert!(
            phase_exponent.abs() < 1e-10,
            "Negate({}) phase_exponent should be 0 but got {}",
            input,
            phase_exponent
        );
    }
}

// ============================================================================
// 3. Exhaustive cswap
// ============================================================================

#[test]
fn test_cswap_exhaustive() {
    for ctrl in 0..=1i64 {
        for x in 0..8i64 {
            for y in 0..8i64 {
                let (phase_exponent, outputs) = run_cswap_pipeline(ctrl, x, y);
                let out_ctrl: i64 = outputs
                    .iter()
                    .find(|o| o.0 == "ctrl")
                    .unwrap()
                    .1
                    .parse()
                    .unwrap();
                let out_x: i64 = outputs
                    .iter()
                    .find(|o| o.0 == "x")
                    .unwrap()
                    .1
                    .parse()
                    .unwrap();
                let out_y: i64 = outputs
                    .iter()
                    .find(|o| o.0 == "y")
                    .unwrap()
                    .1
                    .parse()
                    .unwrap();

                assert_eq!(out_ctrl, ctrl);
                if ctrl == 0 {
                    assert_eq!(out_x, x, "CSwap(0,{},{}) x unchanged", x, y);
                    assert_eq!(out_y, y, "CSwap(0,{},{}) y unchanged", x, y);
                } else {
                    assert_eq!(out_x, y, "CSwap(1,{},{}) x should be {}", x, y, y);
                    assert_eq!(out_y, x, "CSwap(1,{},{}) y should be {}", x, y, x);
                }
                assert!(
                    phase_exponent.abs() < 1e-10,
                    "CSwap({},{},{}) phase_exponent should be 0",
                    ctrl,
                    x,
                    y
                );
            }
        }
    }
}

#[test]
fn test_cswap_max_5bit_values() {
    let (_, outputs) = run_cswap_pipeline(1, 31, 0);
    assert_eq!(outputs.iter().find(|o| o.0 == "x").unwrap().1, "0");
    assert_eq!(outputs.iter().find(|o| o.0 == "y").unwrap().1, "31");

    let (_, outputs) = run_cswap_pipeline(1, 0, 31);
    assert_eq!(outputs.iter().find(|o| o.0 == "x").unwrap().1, "31");
    assert_eq!(outputs.iter().find(|o| o.0 == "y").unwrap().1, "0");
}

#[test]
fn test_cswap_identical_values() {
    for val in [0, 1, 15, 31i64] {
        let (_, outputs) = run_cswap_pipeline(1, val, val);
        let x_val: i64 = outputs
            .iter()
            .find(|o| o.0 == "x")
            .unwrap()
            .1
            .parse()
            .unwrap();
        let y_val: i64 = outputs
            .iter()
            .find(|o| o.0 == "y")
            .unwrap()
            .1
            .parse()
            .unwrap();
        assert_eq!(x_val, val);
        assert_eq!(y_val, val);
    }
}

// ============================================================================
// 4. Negate involution: negate(negate(x)) == x
// ============================================================================

#[test]
fn test_negate_involution() {
    for i in 0u16..256 {
        let input = i as i64;
        let (_, first_outputs) = run_negate_pipeline(input);
        let first_result: i64 = first_outputs[0].1.parse().unwrap();

        let unsigned_first = if first_result < 0 {
            (first_result + 256) as i64
        } else {
            first_result
        };
        let (_, second_outputs) = run_negate_pipeline(unsigned_first);
        let second_result: i64 = second_outputs[0].1.parse().unwrap();

        let orig_signed = if i >= 128 { i as i64 - 256 } else { i as i64 };
        assert_eq!(
            second_result, orig_signed,
            "negate(negate({})) = {} != {}",
            input, second_result, orig_signed
        );
    }
}

// ============================================================================
// 5. Negate large (2048-bit)
// ============================================================================

#[test]
fn test_negate_large_2048bit() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    let input_path = std::path::Path::new(&manifest_dir).join("example_qlts/negate-large.in.txt");
    let input_str = std::fs::read_to_string(&input_path)
        .unwrap_or_else(|e| panic!("Failed to read {:?}: {}", input_path, e));
    let input_str = input_str.trim();

    let qlt_path = std::path::Path::new(&manifest_dir).join("example_qlts/negate-large.qlt");
    let source = std::fs::read_to_string(&qlt_path)
        .unwrap_or_else(|e| panic!("Failed to read {:?}: {}", qlt_path, e));

    let (module, errors) = _rsqlt::parser::parse_l1_module(&source);
    assert!(errors.is_empty(), "Parse errors: {:?}", errors);

    let compiled = _rsqlt::fastsim::compiler::compile(&module).expect("Compilation failed");

    let bits = _rsqlt::fastsim::vm::decimal_str_to_bits(input_str, 2048);
    let input_values = vec![("x".to_string(), bits)];

    let mut sim = _rsqlt::fastsim::vm::VmSimulator::new(&compiled, "Negate")
        .expect("VmSimulator creation failed");
    sim.execute_run(&input_values).expect("Execution failed");

    let outputs = sim.extract_outputs();

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].0, "x");

    let expected_output = format!("-{}", input_str);
    assert_eq!(
        outputs[0].1, expected_output,
        "Negate-large output mismatch"
    );

    assert!(
        sim.phase_exponent().abs() < 1e-10,
        "Negate-large phase_exponent should be 0, got {}",
        sim.phase_exponent()
    );
}

#[test]
fn test_negate_large_involution() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let qlt_path = std::path::Path::new(&manifest_dir).join("example_qlts/negate-large.qlt");
    let source = std::fs::read_to_string(&qlt_path).unwrap();
    let (module, _) = _rsqlt::parser::parse_l1_module(&source);
    let compiled = _rsqlt::fastsim::compiler::compile(&module).unwrap();

    let input_str = "1234567890987654321";
    let bits1 = _rsqlt::fastsim::vm::decimal_str_to_bits(input_str, 2048);
    let input_values1 = vec![("x".to_string(), bits1)];
    let mut sim = _rsqlt::fastsim::vm::VmSimulator::new(&compiled, "Negate").unwrap();
    sim.execute_run(&input_values1).unwrap();
    let outputs1 = sim.extract_outputs();
    assert_eq!(outputs1[0].1, format!("-{}", input_str));

    let bits2 = _rsqlt::fastsim::vm::signed_decimal_str_to_bits(&outputs1[0].1, 2048);
    let input_values2 = vec![("x".to_string(), bits2)];
    sim.execute_run(&input_values2).unwrap();
    let outputs2 = sim.extract_outputs();
    assert_eq!(outputs2[0].1, input_str);
}

// ============================================================================
// 6. Error handling
// ============================================================================

#[test]
fn test_missing_entrypoint() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let source = std::fs::read_to_string(
        std::path::Path::new(&manifest_dir).join("example_qlts/negate.qlt"),
    )
    .unwrap();
    let (module, _) = _rsqlt::parser::parse_l1_module(&source);
    let compiled = _rsqlt::fastsim::compiler::compile(&module).unwrap();

    let result = _rsqlt::fastsim::vm::VmSimulator::new(&compiled, "NonExistent");
    assert!(result.is_err(), "Should error on missing entrypoint");
}

#[test]
fn test_empty_entrypoint() {
    let source = "# Qualtran-L1\n# 1.0.0\nqdef Valid [ x: QInt(8) ] { return [x=x] }\n";
    let (module, _) = _rsqlt::parser::parse_l1_module(source);
    let compiled = _rsqlt::fastsim::compiler::compile(&module).unwrap();
    let res = _rsqlt::fastsim::vm::VmSimulator::new(&compiled, "");
    assert!(res.is_err(), "Empty entrypoint should return Err");
}

#[test]
#[should_panic(expected = "was never interned during compilation")]
fn test_unknown_input_register() {
    let source = "# Qualtran-L1\n# 1.0.0\nqdef Simple [ x: QInt(8) ] { return [x=x] }\n";
    let (module, errors) = _rsqlt::parser::parse_l1_module(source);
    assert!(errors.is_empty());
    let compiled = _rsqlt::fastsim::compiler::compile(&module).unwrap();
    let input_values = vec![(
        "unknown_reg".to_string(),
        _rsqlt::fastsim::vm::decimal_str_to_bits("0", 8),
    )];
    let mut sim = _rsqlt::fastsim::vm::VmSimulator::new(&compiled, "Simple").unwrap();
    let _ = sim.execute_run(&input_values);
}

#[test]
fn test_bitwidth_mismatch_cast() {
    let source = "# Qualtran-L1\n# 1.0.0\nqdef BadCast [ x: QInt(5) ] {\n reg = Split(QInt(8)) [reg=x]\n reg2 = Join(QInt(5)) [reg=reg]\n return [x=reg2]\n}\nqcast Split(QInt(8)) [reg: QInt(8) -> QBit[8]]\nqcast Join(QInt(5)) [reg: QBit[5] -> QInt(5)]\n";
    let (module, _) = _rsqlt::parser::parse_l1_module(source);
    let compiled = _rsqlt::fastsim::compiler::compile(&module).unwrap();
    let input_values = vec![("x".to_string(), vec![false; 5])];
    let mut sim = _rsqlt::fastsim::vm::VmSimulator::new(&compiled, "BadCast").unwrap();
    let result = sim.execute_run(&input_values);
    assert!(result.is_err(), "Bitwidth mismatch should produce an error");
    let err = result.unwrap_err();
    assert!(
        err.contains("Split: expected 8 bits but found 5"),
        "Error: {}",
        err
    );
}

#[test]
#[should_panic(expected = "distribute_elements")]
fn test_invalid_bit_index_access() {
    let source = "# Qualtran-L1\n# 1.0.0\nqdef BadIdx [ x: QInt(2) ] {\n reg = Split(QInt(2)) [reg=x]\n reg2 = Join(QInt(2)) [reg=[reg[5], reg[0]]]\n return [x=reg2]\n}\nqcast Split(QInt(2)) [reg: QInt(2) -> QBit[2]]\nqcast Join(QInt(2)) [reg: QBit[2] -> QInt(2)]\n";
    let (module, _) = _rsqlt::parser::parse_l1_module(source);
    let compiled = _rsqlt::fastsim::compiler::compile(&module).unwrap();
    let input_values = vec![("x".to_string(), vec![false, false])];
    let mut sim = _rsqlt::fastsim::vm::VmSimulator::new(&compiled, "BadIdx").unwrap();
    let _ = sim.execute_run(&input_values);
}

// ============================================================================
// 7. Output format verification
// ============================================================================

#[test]
fn test_output_format_negate() {
    let (phase_exponent, outputs) = run_negate_pipeline(5);
    let mut parts: Vec<String> = Vec::new();
    for (name, value_str, _dtype) in &outputs {
        parts.push(format!("{}={}", name, value_str));
    }
    if phase_exponent == 0.0 {
        parts.push("phase_exponent=0".to_string());
    } else {
        parts.push(format!("phase_exponent={}", phase_exponent));
    }
    let output_line = parts.join(" ");
    assert_eq!(output_line, "x=-5 phase_exponent=0");
}

#[test]
fn test_output_format_cswap() {
    let (phase_exponent, outputs) = run_cswap_pipeline(1, 7, 3);
    let mut parts: Vec<String> = Vec::new();
    for (name, value_str, _dtype) in &outputs {
        parts.push(format!("{}={}", name, value_str));
    }
    if phase_exponent == 0.0 {
        parts.push("phase_exponent=0".to_string());
    } else {
        parts.push(format!("phase_exponent={}", phase_exponent));
    }
    let output_line = parts.join(" ");
    assert_eq!(output_line, "ctrl=1 x=3 y=7 phase_exponent=0");
}

#[test]
fn test_output_no_trailing_whitespace() {
    let (_, outputs) = run_negate_pipeline(0);
    let mut parts: Vec<String> = Vec::new();
    for (name, value_str, _dtype) in &outputs {
        parts.push(format!("{}={}", name, value_str));
    }
    parts.push("phase_exponent=0".to_string());
    let output_line = parts.join(" ");
    assert!(!output_line.starts_with(' '), "Output starts with space");
    assert!(!output_line.ends_with(' '), "Output ends with space");
    assert!(!output_line.contains("  "), "Output contains double spaces");
}

// ============================================================================
// 8. QUInt vs QInt formatting
// ============================================================================

#[test]
fn test_quint_vs_qint_formatting() {
    let source = "# Qualtran-L1\n# 1.0.0\nqdef Formatting [ x: QUInt(8), y: QInt(8) ] {\n return [x=x, y=y]\n}\n";
    let (module, _) = _rsqlt::parser::parse_l1_module(source);
    let compiled = _rsqlt::fastsim::compiler::compile(&module).unwrap();

    let bits = vec![true; 8];
    let input_values = vec![("x".to_string(), bits.clone()), ("y".to_string(), bits)];
    let mut sim = _rsqlt::fastsim::vm::VmSimulator::new(&compiled, "Formatting").unwrap();
    sim.execute_run(&input_values).unwrap();
    let outputs = sim.extract_outputs();
    assert_eq!(outputs[0].1, "255"); // QUInt
    assert_eq!(outputs[1].1, "-1"); // QInt
}

// ============================================================================
// 9. Identifier resolution edge cases
// ============================================================================

#[test]
fn test_bitref_resolution() {
    let source = "# Qualtran-L1\n# 1.0.0\nqdef BitRefTest [ x: QInt(2) ] {\n reg = Split(QInt(2)) [reg=x]\n reg2 = Join(QInt(2)) [reg=[reg[1], reg[0]]]\n return [x=reg2]\n}\nqcast Split(QInt(2)) [reg: QInt(2) -> QBit[2]]\nqcast Join(QInt(2)) [reg: QBit[2] -> QInt(2)]\n";
    let (module, _) = _rsqlt::parser::parse_l1_module(source);
    let compiled = _rsqlt::fastsim::compiler::compile(&module).unwrap();
    let input_values = vec![("x".to_string(), vec![false, true])];
    let mut sim = _rsqlt::fastsim::vm::VmSimulator::new(&compiled, "BitRefTest").unwrap();
    sim.execute_run(&input_values).unwrap();
    let outputs = sim.extract_outputs();
    assert_eq!(outputs[0].1, "-2"); // vec![true, false] is -2 in QInt(2)
}

#[test]
fn test_case_sensitivity() {
    let source = "# Qualtran-L1\n# 1.0.0\nqdef Case [ my_reg: QInt(8), MY_REG: QInt(8) ] { return [my_reg=my_reg, MY_REG=MY_REG] }\n";
    let (module, _) = _rsqlt::parser::parse_l1_module(source);
    let compiled = _rsqlt::fastsim::compiler::compile(&module).unwrap();
    let input_values = vec![
        (
            "my_reg".to_string(),
            _rsqlt::fastsim::vm::decimal_str_to_bits("10", 8),
        ),
        (
            "MY_REG".to_string(),
            _rsqlt::fastsim::vm::decimal_str_to_bits("20", 8),
        ),
    ];
    let mut sim = _rsqlt::fastsim::vm::VmSimulator::new(&compiled, "Case").unwrap();
    sim.execute_run(&input_values).unwrap();
    let outputs = sim.extract_outputs();
    assert_eq!(outputs[0].1, "10");
    assert_eq!(outputs[1].1, "20");
}

#[test]
fn test_argmapping_resolution() {
    let source = "# Qualtran-L1\n# 1.0.0\nqdef Caller [ a: QInt(8) ] {\n b = Callee [x=a]\n return [a=b]\n}\nqdef Callee [ x: QInt(8) ] {\n return [x=x]\n}\n";
    let (module, _) = _rsqlt::parser::parse_l1_module(source);
    let compiled = _rsqlt::fastsim::compiler::compile(&module).unwrap();
    let input_values = vec![(
        "a".to_string(),
        _rsqlt::fastsim::vm::decimal_str_to_bits("5", 8),
    )];
    let mut sim = _rsqlt::fastsim::vm::VmSimulator::new(&compiled, "Caller").unwrap();
    sim.execute_run(&input_values).unwrap();
    let outputs = sim.extract_outputs();
    assert_eq!(outputs[0].1, "5");
}

#[test]
fn test_long_register_name() {
    let long_name = "a".repeat(256);
    let source = format!(
        "# Qualtran-L1\n# 1.0.0\nqdef LongReg [ {}: QInt(8) ] {{\n return [{}={}]\n}}\n",
        long_name, long_name, long_name
    );
    let (module, _) = _rsqlt::parser::parse_l1_module(&source);
    let compiled = _rsqlt::fastsim::compiler::compile(&module).unwrap();
    let input_values = vec![(long_name.clone(), vec![false; 8])];
    let mut sim = _rsqlt::fastsim::vm::VmSimulator::new(&compiled, "LongReg").unwrap();
    sim.execute_run(&input_values).unwrap();
    let outputs = sim.extract_outputs();
    assert_eq!(outputs[0].0, long_name);
    assert_eq!(outputs[0].1, "0");
}

// ============================================================================
// 10. Composite ALU execution
// ============================================================================

#[test]
fn test_composite_alu_execution() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let qlt_path = std::path::Path::new(&manifest_dir).join("example_qlts/composite_alu.qlt");
    let source = std::fs::read_to_string(&qlt_path).unwrap();
    let (module, errors) = _rsqlt::parser::parse_l1_module(&source);
    assert!(errors.is_empty(), "Parse errors: {:?}", errors);
    let compiled = _rsqlt::fastsim::compiler::compile(&module).unwrap();

    let bits_a = _rsqlt::fastsim::vm::decimal_str_to_bits("123", 72);
    let bits_b = _rsqlt::fastsim::vm::decimal_str_to_bits("456", 72);
    let input_values = vec![("a".to_string(), bits_a), ("b".to_string(), bits_b)];
    let mut sim = _rsqlt::fastsim::vm::VmSimulator::new(&compiled, "CompositeALU").unwrap();
    sim.execute_run(&input_values).unwrap();
    let outputs = sim.extract_outputs();
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].0, "a");
    assert_eq!(outputs[1].0, "b");
}

// ============================================================================
// 11. ND array (multi-dimensional index) execution
// ============================================================================

/// Rust translation of the Python reference implementation for TestND3Grid.
///
/// ```python
/// def _reference_nd3_grid(cube, aux):
///     c = cube.copy()
///     a = int(aux)
///     c[0, 0, 0] ^= 1
///     c[0, 1, 0] ^= c[0, 0, 1]
///     c[1, 0, 0] ^= c[0, 1, 1]
///     c[1, 1, 0] ^= c[1, 0, 0] & c[1, 0, 1]
///     c[1, 1, 1] ^= c[1, 1, 0]
///     a ^= c[1, 1, 1]
///     return c, a
/// ```
///
/// `cube` is a row-major flat array of 8 bools for shape [2, 2, 2].
/// Row-major flat indices:
///   [0,0,0]=0  [0,0,1]=1  [0,1,0]=2  [0,1,1]=3
///   [1,0,0]=4  [1,0,1]=5  [1,1,0]=6  [1,1,1]=7
fn reference_nd3_grid(cube: &[bool; 8], aux: bool) -> ([bool; 8], bool) {
    let mut c = *cube;
    let mut a = aux;

    c[0] ^= true; // c[0,0,0] ^= 1
    c[2] ^= c[1]; // c[0,1,0] ^= c[0,0,1]
    c[4] ^= c[3]; // c[1,0,0] ^= c[0,1,1]
    c[6] ^= c[4] & c[5]; // c[1,1,0] ^= c[1,0,0] & c[1,0,1]
    c[7] ^= c[6]; // c[1,1,1] ^= c[1,1,0]
    a ^= c[7]; // a ^= c[1,1,1]

    (c, a)
}

/// Convert a flat bool slice to an unsigned integer (big-endian / MSB-first).
fn bits_to_uint(bits: &[bool]) -> u64 {
    let mut val: u64 = 0;
    for &b in bits {
        val = (val << 1) | (b as u64);
    }
    val
}

/// Helper: parse, compile, and execute TestND3Grid with given inputs.
fn run_nd3grid_pipeline(
    cube_bits: Vec<bool>,
    aux_bit: bool,
) -> (f64, Vec<(String, String, String)>) {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let file_path = std::path::Path::new(&manifest_dir).join("example_qlts/testnd3grid.qlt");
    let source = std::fs::read_to_string(&file_path)
        .unwrap_or_else(|e| panic!("Failed to read {:?}: {}", file_path, e));

    let (module, errors) = _rsqlt::parser::parse_l1_module(&source);
    assert!(errors.is_empty(), "Parse errors: {:?}", errors);

    let compiled =
        _rsqlt::fastsim::compiler::compile(&module).expect("Compilation failed for testnd3grid");

    let input_values = vec![
        ("cube".to_string(), cube_bits),
        ("aux".to_string(), vec![aux_bit]),
    ];
    let mut sim = _rsqlt::fastsim::vm::VmSimulator::new(&compiled, "TestND3Grid")
        .expect("VmSimulator creation failed for TestND3Grid");
    sim.execute_run(&input_values)
        .expect("Execution failed for TestND3Grid");
    let outputs = sim.extract_outputs();
    (sim.phase_exponent(), outputs)
}

/// Exhaustive test: run all 2^9 = 512 inputs through both the simulation
/// and the Rust reference, and verify they agree.
#[test]
fn test_nd3grid_exhaustive_vs_reference() {
    for input_val in 0u32..512 {
        // Decode 9 bits: cube[8] | aux[1], MSB-first.
        let mut cube = [false; 8];
        for i in 0..8 {
            cube[i] = (input_val >> (8 - i)) & 1 != 0;
        }
        let aux = input_val & 1 != 0;

        // Reference computation.
        let (exp_cube, exp_aux) = reference_nd3_grid(&cube, aux);

        // Simulation.
        let (_phase, outputs) = run_nd3grid_pipeline(cube.to_vec(), aux);

        let cube_out = &outputs.iter().find(|o| o.0 == "cube").unwrap().1;
        let aux_out = &outputs.iter().find(|o| o.0 == "aux").unwrap().1;

        let exp_cube_val = bits_to_uint(&exp_cube);
        let exp_aux_val = exp_aux as u64;

        assert_eq!(
            cube_out,
            &exp_cube_val.to_string(),
            "input={:#011b}: cube mismatch (got {}, expected {})",
            input_val,
            cube_out,
            exp_cube_val
        );
        assert_eq!(
            aux_out,
            &exp_aux_val.to_string(),
            "input={:#011b}: aux mismatch (got {}, expected {})",
            input_val,
            aux_out,
            exp_aux_val
        );
    }
}

// ============================================================================
// Multi-register partition (`_PartitionBase`) casts
// ============================================================================

/// A multi-register partition splits one lumped register into several parts.
/// x = 200 = 0b11001000, big-endian bits [1,1,0,0,1,0,0,0].
/// lo (first 3 bits) = [1,1,0] = 6; hi (next 5 bits) = [0,1,0,0,0] = 8.
#[test]
fn test_partition_scatter() {
    let source = "# Qualtran-L1\n# 1.0.0\n\
qdef DoPartition [ x: QUInt(8) -> |, lo: | -> QUInt(3), hi: | -> QUInt(5) ] {\n\
 lo, hi = Partition [x=x]\n\
 return [lo=lo, hi=hi]\n\
}\n\
qcast Partition [x: QUInt(8) -> |, lo: | -> QUInt(3), hi: | -> QUInt(5)]\n";
    let (module, errors) = _rsqlt::parser::parse_l1_module(source);
    assert!(errors.is_empty(), "Parse errors: {:?}", errors);
    let compiled = _rsqlt::fastsim::compiler::compile(&module).expect("Compilation failed");

    let input_values = vec![("x".to_string(), _rsqlt::fastsim::vm::int_to_bits(200, 8))];
    let mut sim = _rsqlt::fastsim::vm::VmSimulator::new(&compiled, "DoPartition").unwrap();
    sim.execute_run(&input_values).expect("Execution failed");
    let outputs = sim.extract_outputs();

    let lo = &outputs.iter().find(|o| o.0 == "lo").unwrap().1;
    let hi = &outputs.iter().find(|o| o.0 == "hi").unwrap().1;
    assert_eq!(lo, "6", "lo mismatch");
    assert_eq!(hi, "8", "hi mismatch");
}

/// Partition then merge (the adjoint) must round-trip to the identity.
#[test]
fn test_partition_merge_roundtrip() {
    let source = "# Qualtran-L1\n# 1.0.0\n\
qdef RoundTrip [ x: QUInt(8) ] {\n\
 lo, hi = Partition [x=x]\n\
 y = Merge [lo=lo, hi=hi]\n\
 return [x=y]\n\
}\n\
qcast Partition [x: QUInt(8) -> |, lo: | -> QUInt(3), hi: | -> QUInt(5)]\n\
qcast Merge [x: | -> QUInt(8), lo: QUInt(3) -> |, hi: QUInt(5) -> |]\n";
    let (module, errors) = _rsqlt::parser::parse_l1_module(source);
    assert!(errors.is_empty(), "Parse errors: {:?}", errors);
    let compiled = _rsqlt::fastsim::compiler::compile(&module).expect("Compilation failed");

    let mut sim = _rsqlt::fastsim::vm::VmSimulator::new(&compiled, "RoundTrip").unwrap();
    for v in 0..256i64 {
        let input_values = vec![("x".to_string(), _rsqlt::fastsim::vm::int_to_bits(v, 8))];
        sim.execute_run(&input_values).expect("Execution failed");
        let outputs = sim.extract_outputs();
        let x_out = &outputs.iter().find(|o| o.0 == "x").unwrap().1;
        assert_eq!(x_out, &v.to_string(), "round-trip mismatch for x={}", v);
    }
}
