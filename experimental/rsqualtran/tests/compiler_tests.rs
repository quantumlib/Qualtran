//! Compiler output verification tests.
//!
//! Tests that the compiler produces correct subroutine structures, register
//! signatures, extern gate classifications, cast operations, and handles
//! edge cases like missing targets and unknown types.

// ============================================================================
// 1. Subroutine existence and separation
// ============================================================================

#[test]
fn test_compiler_produces_distinct_subroutines() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    let neg_source = std::fs::read_to_string(
        std::path::Path::new(&manifest_dir).join("example_qlts/negate.qlt"),
    )
    .unwrap();
    let csw_source =
        std::fs::read_to_string(std::path::Path::new(&manifest_dir).join("example_qlts/cswap.qlt"))
            .unwrap();

    let (neg_mod, _) = _rsqlt::parser::parse_l1_module(&neg_source);
    let (csw_mod, _) = _rsqlt::parser::parse_l1_module(&csw_source);

    let neg_compiled = _rsqlt::fastsim::compiler::compile(&neg_mod).unwrap();
    let csw_compiled = _rsqlt::fastsim::compiler::compile(&csw_mod).unwrap();

    assert!(neg_compiled.has_subroutine("Negate"));
    assert!(!neg_compiled.has_subroutine("CSwap"));

    assert!(csw_compiled.has_subroutine("CSwap"));
    assert!(!csw_compiled.has_subroutine("Negate"));
}

// ============================================================================
// 2. Register signature verification
// ============================================================================

#[test]
fn test_negate_signature() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let source = std::fs::read_to_string(
        std::path::Path::new(&manifest_dir).join("example_qlts/negate.qlt"),
    )
    .unwrap();
    let (module, _) = _rsqlt::parser::parse_l1_module(&source);
    let compiled = _rsqlt::fastsim::compiler::compile(&module).unwrap();

    let negate_sub = compiled.get_subroutine("Negate").unwrap();
    assert_eq!(negate_sub.signature.len(), 1);
    assert_eq!(
        compiled.intern_table.resolve(negate_sub.signature[0].name),
        "x"
    );
    assert_eq!(negate_sub.signature[0].n_bits, 8);
    assert_eq!(
        compiled
            .intern_table
            .resolve(negate_sub.signature[0].dtype_name),
        "QInt"
    );
    assert_eq!(
        negate_sub.signature[0].direction,
        _rsqlt::fastsim::compiler::RegisterDirection::Thru
    );
}

#[test]
fn test_cswap_signature() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let source =
        std::fs::read_to_string(std::path::Path::new(&manifest_dir).join("example_qlts/cswap.qlt"))
            .unwrap();
    let (module, _) = _rsqlt::parser::parse_l1_module(&source);
    let compiled = _rsqlt::fastsim::compiler::compile(&module).unwrap();

    let cswap_sub = compiled.get_subroutine("CSwap").unwrap();
    assert_eq!(cswap_sub.signature.len(), 3);

    let ctrl_reg = cswap_sub
        .signature
        .iter()
        .find(|r| compiled.intern_table.resolve(r.name) == "ctrl")
        .unwrap();
    assert_eq!(ctrl_reg.n_bits, 1);

    let x_reg = cswap_sub
        .signature
        .iter()
        .find(|r| compiled.intern_table.resolve(r.name) == "x")
        .unwrap();
    assert_eq!(x_reg.n_bits, 5);

    let y_reg = cswap_sub
        .signature
        .iter()
        .find(|r| compiled.intern_table.resolve(r.name) == "y")
        .unwrap();
    assert_eq!(y_reg.n_bits, 5);
}

// ============================================================================
// 3. Extern gate classification
// ============================================================================

#[test]
fn test_extern_gate_classification() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let source = std::fs::read_to_string(
        std::path::Path::new(&manifest_dir).join("example_qlts/negate.qlt"),
    )
    .unwrap();
    let (module, _) = _rsqlt::parser::parse_l1_module(&source);
    let compiled = _rsqlt::fastsim::compiler::compile(&module).unwrap();

    let x_sub = compiled.get_subroutine("X").unwrap();
    match &x_sub.body {
        _rsqlt::fastsim::compiler::SubroutineBody::Extern(gate) => {
            assert!(matches!(
                gate,
                _rsqlt::fastsim::compiler::ExternGate::XGate { .. }
            ));
        }
        other => panic!("X should be Extern, got {:?}", other),
    }

    let and_sub = compiled.get_subroutine("And").unwrap();
    match &and_sub.body {
        _rsqlt::fastsim::compiler::SubroutineBody::Extern(gate) => {
            assert!(matches!(
                gate,
                _rsqlt::fastsim::compiler::ExternGate::And { .. }
            ));
        }
        other => panic!("And should be Extern, got {:?}", other),
    }

    let and_dag_sub = compiled.get_subroutine("And_dag").unwrap();
    match &and_dag_sub.body {
        _rsqlt::fastsim::compiler::SubroutineBody::Extern(gate) => {
            assert!(matches!(
                gate,
                _rsqlt::fastsim::compiler::ExternGate::AndDag { .. }
            ));
        }
        other => panic!("And_dag should be Extern, got {:?}", other),
    }

    let cnot_sub = compiled.get_subroutine("CNOT").unwrap();
    match &cnot_sub.body {
        _rsqlt::fastsim::compiler::SubroutineBody::Extern(gate) => {
            assert!(matches!(
                gate,
                _rsqlt::fastsim::compiler::ExternGate::CNOT { .. }
            ));
        }
        other => panic!("CNOT should be Extern, got {:?}", other),
    }
}

// ============================================================================
// 4. Cast operation classification
// ============================================================================

#[test]
fn test_cast_operations() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let source = std::fs::read_to_string(
        std::path::Path::new(&manifest_dir).join("example_qlts/negate.qlt"),
    )
    .unwrap();
    let (module, _) = _rsqlt::parser::parse_l1_module(&source);
    let compiled = _rsqlt::fastsim::compiler::compile(&module).unwrap();

    let split_sub = compiled.get_subroutine("Split(QInt(8))").unwrap();
    match &split_sub.body {
        _rsqlt::fastsim::compiler::SubroutineBody::Cast(op) => match op {
            _rsqlt::fastsim::compiler::CastOp::Split { total_bits, .. } => {
                assert_eq!(*total_bits, 8);
            }
            other => panic!("Split(QInt(8)) should be Split, got {:?}", other),
        },
        other => panic!("Split(QInt(8)) should be Cast, got {:?}", other),
    }

    let join_sub = compiled.get_subroutine("Join(QInt(8))").unwrap();
    match &join_sub.body {
        _rsqlt::fastsim::compiler::SubroutineBody::Cast(op) => match op {
            _rsqlt::fastsim::compiler::CastOp::Join { total_bits, .. } => {
                assert_eq!(*total_bits, 8);
            }
            other => panic!("Join(QInt(8)) should be Join, got {:?}", other),
        },
        other => panic!("Join(QInt(8)) should be Cast, got {:?}", other),
    }
}

// ============================================================================
// 5. Compiler edge cases
// ============================================================================

#[test]
fn test_compiler_empty_qdef() {
    let source = "# Qualtran-L1\n# 1.0.0\nqdef Empty [ x: QInt(8) ] {\n return [x=x]\n}\n";
    let (module, errors) = _rsqlt::parser::parse_l1_module(source);
    assert!(errors.is_empty(), "Parse errors: {:?}", errors);
    let compiled = _rsqlt::fastsim::compiler::compile(&module).expect("Compilation failed");
    assert!(compiled.has_subroutine("Empty"));
}

#[test]
fn test_compiler_max_registers() {
    let mut regs = Vec::new();
    let mut rets = Vec::new();
    for i in 0..64 {
        regs.push(format!("r{}: QBit", i));
        rets.push(format!("r{}=r{}", i, i));
    }
    let source = format!(
        "# Qualtran-L1\n# 1.0.0\nqdef MaxRegs [\n {}\n] {{\n return [{}]\n}}\n",
        regs.join(",\n"),
        rets.join(", ")
    );
    let (module, errors) = _rsqlt::parser::parse_l1_module(&source);
    assert!(errors.is_empty(), "Parse errors: {:?}", errors);
    let compiled = _rsqlt::fastsim::compiler::compile(&module).expect("Compilation failed");
    let sub = compiled.get_subroutine("MaxRegs").unwrap();
    assert_eq!(sub.signature.len(), 64);
}

#[test]
fn test_compiler_extreme_bitwidth() {
    let source = "# Qualtran-L1\n# 1.0.0\nqdef Massive [ x: QInt(65536) ] {\n return [x=x]\n}\n";
    let (module, errors) = _rsqlt::parser::parse_l1_module(source);
    assert!(errors.is_empty());
    let compiled = _rsqlt::fastsim::compiler::compile(&module).expect("Compilation failed");
    let sub = compiled.get_subroutine("Massive").unwrap();
    assert_eq!(sub.signature[0].n_bits, 65536);
}

#[test]
fn test_compiler_nested_oneach() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let file_path = std::path::Path::new(&manifest_dir).join("example_qlts/negate.qlt");
    let source = std::fs::read_to_string(&file_path).unwrap();
    let (module, errors) = _rsqlt::parser::parse_l1_module(&source);
    assert!(errors.is_empty());
    let compiled = _rsqlt::fastsim::compiler::compile(&module).expect("Compilation failed");
    assert!(compiled.has_subroutine("X(oneach=8)"));
}

// ============================================================================
// 6. Compiler error handling
// ============================================================================

#[test]
fn test_unrecognized_extern_gate() {
    let source = r#"
# Qualtran-L1
# 1.0.0

extern qdef FakeGate
from some.unknown.FakeGate
[q: QBit]
"#;
    let (module, errors) = _rsqlt::parser::parse_l1_module(source);
    assert!(errors.is_empty());
    let result = _rsqlt::fastsim::compiler::compile(&module);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Unrecognized extern gate"));
}

#[test]
fn test_call_nonexistent_subroutine() {
    let source = "# Qualtran-L1\n# 1.0.0\nqdef Foo [ x: QInt(8) ] {\n y = NonExistent [p=x]\n return [x=y]\n}\n";
    let (module, _) = _rsqlt::parser::parse_l1_module(source);
    let res = _rsqlt::fastsim::compiler::compile(&module);
    assert!(res.is_err());
    assert!(res
        .unwrap_err()
        .contains("call target 'NonExistent' not found"));
}

#[test]
fn test_unknown_dtype() {
    let source = "# Qualtran-L1\n# 1.0.0\nqdef BadType [ x: UnknownType ] { return [x=x] }\n";
    let (module, _) = _rsqlt::parser::parse_l1_module(source);
    let res = _rsqlt::fastsim::compiler::compile(&module);
    assert!(res.is_err());
    assert!(res
        .unwrap_err()
        .contains("Unknown quantum data type: UnknownType"));
}

#[test]
fn test_mismatched_cast_bitwidths() {
    let source = "# Qualtran-L1\n# 1.0.0\nqdef Foo [ x: QInt(8) ] { return [x=x] }\nqcast BadCast [reg: QInt(8) -> QBit[5]]\n";
    let (module, errors) = _rsqlt::parser::parse_l1_module(source);
    assert!(errors.is_empty(), "Parse errors: {:?}", errors);
    let res = _rsqlt::fastsim::compiler::compile(&module);
    assert!(
        res.is_err(),
        "Expected compilation error for mismatched cast bitwidths"
    );
    assert!(res
        .unwrap_err()
        .contains("Both sides of a cast must have the same total bit count"));
}

#[test]
#[should_panic(expected = "not populated in call frame")]
fn test_missing_return_variable() {
    let source =
        "# Qualtran-L1\n# 1.0.0\nqdef MissingVar [ x: QInt(8) ] {\n return [x=missing_var]\n}\n";
    let (module, _) = _rsqlt::parser::parse_l1_module(source);
    let compiled = _rsqlt::fastsim::compiler::compile(&module).unwrap();
    let input_values = vec![("x".to_string(), vec![false; 8])];
    let mut sim = _rsqlt::fastsim::vm::VmSimulator::new(&compiled, "MissingVar").unwrap();
    let _ = sim.execute_run(&input_values);
}

// ============================================================================
// 7. Advanced dtype support
// ============================================================================

#[test]
fn test_advanced_qdatatypes() {
    let source = "# Qualtran-L1\n# 1.0.0\nqdef Advanced [ a: QMontgomeryUInt(16), b: QFxp(32) ] { return [a=a, b=b] }\n";
    let (module, errors) = _rsqlt::parser::parse_l1_module(source);
    assert!(errors.is_empty(), "Parse errors: {:?}", errors);
    let compiled = _rsqlt::fastsim::compiler::compile(&module).expect("Compilation failed");
    let sub = compiled
        .get_subroutine("Advanced")
        .expect("Subroutine not found");
    assert_eq!(sub.signature[0].n_bits, 16);
    assert_eq!(sub.signature[1].n_bits, 32);
}

// ============================================================================
// 8. Stress: many unique symbols
// ============================================================================

#[test]
fn test_stress_thousands_unique_symbols() {
    let mut regs = Vec::new();
    let mut rets = Vec::new();
    for i in 0..1000 {
        regs.push(format!("v{}: QBit", i));
        rets.push(format!("v{}=v{}", i, i));
    }
    let source = format!(
        "# Qualtran-L1\n# 1.0.0\nqdef Stress [\n {}\n] {{\n return [{}]\n}}\n",
        regs.join(",\n"),
        rets.join(", ")
    );
    let (module, errors) = _rsqlt::parser::parse_l1_module(&source);
    assert!(errors.is_empty());
    let compiled = _rsqlt::fastsim::compiler::compile(&module).unwrap();
    let sub = compiled.get_subroutine("Stress").unwrap();
    assert_eq!(sub.signature.len(), 1000);
}

#[test]
fn test_extremely_long_subroutine_key() {
    let long_key = "k".repeat(10000);
    let source = format!(
        "# Qualtran-L1\n# 1.0.0\nqdef {} [ x: QInt(8) ] {{ return [x=x] }}\n",
        long_key
    );
    let (module, _) = _rsqlt::parser::parse_l1_module(&source);
    let compiled = _rsqlt::fastsim::compiler::compile(&module).unwrap();
    assert!(compiled.has_subroutine(&long_key));
}

#[test]
fn test_compiler_zero_shape_dimension() {
    let dtype = _rsqlt::nodes::QDTypeNode {
        dtype: _rsqlt::nodes::CObjectNode {
            name: "QBit".to_string(),
            cargs: vec![],
            span: _rsqlt::nodes::Span::default(),
        },
        shape: Some(vec![0, 2]),
        span: _rsqlt::nodes::Span::default(),
    };
    let result = _rsqlt::fastsim::compiler::compute_dtype_bits(&dtype);
    assert_eq!(
        result,
        Err("Shape dimensions must be greater than 0, got 0".to_string())
    );
}

#[test]
fn test_compiler_negative_shape_dimension() {
    let dtype = _rsqlt::nodes::QDTypeNode {
        dtype: _rsqlt::nodes::CObjectNode {
            name: "QBit".to_string(),
            cargs: vec![],
            span: _rsqlt::nodes::Span::default(),
        },
        shape: Some(vec![-1, 2]),
        span: _rsqlt::nodes::Span::default(),
    };
    let result = _rsqlt::fastsim::compiler::compute_dtype_bits(&dtype);
    assert_eq!(
        result,
        Err("Shape dimensions must be greater than 0, got -1".to_string())
    );
}

#[test]
fn test_unknown_dtype_signedness() {
    let err = _rsqlt::fastsim::compiler::is_signed_dtype("UnknownType");
    assert_eq!(
        err,
        Err("Unknown quantum data type for signedness classification: UnknownType".to_string())
    );
}


