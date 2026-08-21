// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Integration tests for the QLT parser.
//!
//! Verifies that example .qlt files parse without errors and produce
//! the expected AST structure (qdef, qcast, extern counts and names).

#[test]
fn test_parse_cswap() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let file_path = std::path::Path::new(&manifest_dir).join("example_qlts/cswap.qlt");
    let code = std::fs::read_to_string(file_path).expect("Failed to read example_qlts/cswap.qlt");

    let (module, errors) = _rsqlt::parser::parse_l1_module(&code);
    assert!(errors.is_empty(), "Parser reported errors: {:?}", errors);

    assert!(!module.qdefs.is_empty(), "Module should not be empty");

    let cswap = module.qdefs.iter().find(|d| match d {
        _rsqlt::nodes::QDefNode::Impl(n) => n.bloq_key == "CSwap",
        _rsqlt::nodes::QDefNode::Extern(n) => n.bloq_key == "CSwap",
        _rsqlt::nodes::QDefNode::Cast(_) => false,
    });
    assert!(cswap.is_some(), "CSwap definition not found");

    let cast_count = module
        .qdefs
        .iter()
        .filter(|d| matches!(d, _rsqlt::nodes::QDefNode::Cast(_)))
        .count();
    assert!(
        cast_count >= 2,
        "Expected at least 2 qcast entries (Split, Join), found {}",
        cast_count
    );
}

#[test]
fn test_parse_negate() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let file_path = std::path::Path::new(&manifest_dir).join("example_qlts/negate.qlt");
    let code = std::fs::read_to_string(file_path).expect("Failed to read example_qlts/negate.qlt");

    let (module, errors) = _rsqlt::parser::parse_l1_module(&code);
    assert!(errors.is_empty(), "Parser reported errors: {:?}", errors);

    assert!(!module.qdefs.is_empty(), "Module should not be empty");

    let negate = module.qdefs.iter().find(|d| match d {
        _rsqlt::nodes::QDefNode::Impl(n) => n.bloq_key == "Negate",
        _ => false,
    });
    assert!(negate.is_some(), "Negate definition not found");

    let cast_count = module
        .qdefs
        .iter()
        .filter(|d| matches!(d, _rsqlt::nodes::QDefNode::Cast(_)))
        .count();
    assert!(
        cast_count >= 2,
        "Expected at least 2 qcast entries, found {}",
        cast_count
    );

    let extern_count = module
        .qdefs
        .iter()
        .filter(|d| matches!(d, _rsqlt::nodes::QDefNode::Extern(_)))
        .count();
    assert!(
        extern_count >= 5,
        "Expected at least 5 extern qdefs, found {}",
        extern_count
    );
}

#[test]
fn test_parse_composite_alu() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let file_path = std::path::Path::new(&manifest_dir).join("example_qlts/composite_alu.qlt");
    let code =
        std::fs::read_to_string(file_path).expect("Failed to read example_qlts/composite_alu.qlt");

    let (module, errors) = _rsqlt::parser::parse_l1_module(&code);
    assert!(errors.is_empty(), "Parser reported errors: {:?}", errors);

    assert!(!module.qdefs.is_empty(), "Module should not be empty");

    let composite_alu = module.qdefs.iter().find(|d| match d {
        _rsqlt::nodes::QDefNode::Impl(n) => n.bloq_key == "CompositeALU",
        _ => false,
    });
    assert!(composite_alu.is_some(), "CompositeALU definition not found");
}

#[test]
fn test_parse_negate_large() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let file_path = std::path::Path::new(&manifest_dir).join("example_qlts/negate-large.qlt");
    let code =
        std::fs::read_to_string(file_path).expect("Failed to read example_qlts/negate-large.qlt");

    let (module, errors) = _rsqlt::parser::parse_l1_module(&code);
    assert!(errors.is_empty(), "Parser reported errors: {:?}", errors);

    assert!(!module.qdefs.is_empty(), "Module should not be empty");

    let negate = module.qdefs.iter().find(|d| match d {
        _rsqlt::nodes::QDefNode::Impl(n) => n.bloq_key == "Negate",
        _ => false,
    });
    assert!(
        negate.is_some(),
        "Negate definition not found in negate-large.qlt"
    );
}
