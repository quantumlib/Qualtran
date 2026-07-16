use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::fastsim::{
    compiler::{self, CompiledModule, CompiledSubroutine},
    vm::{self, PyVmSimulator, SimState},
};
use crate::nodes::{
    AliasAssignmentNode, CArgNode, CObjectNode, L1Module, LiteralNode, QArgNode, QArgValueNode,
    QCallNode, QCastNode, QDTypeNode, QDefExternNode, QDefImplNode, QReturnNode, QSignatureEntry,
    Span, TupleNode,
};
use crate::parser;

/// Parse a Qualtran-L1 string into an L1Module AST.
#[pyfunction]
fn parse_l1_module(code: String) -> PyResult<L1Module> {
    let (module, errors) = parser::parse_l1_module(&code);
    if !errors.is_empty() {
        let msg = errors
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        return Err(PyValueError::new_err(msg));
    }
    Ok(module)
}

#[pyfunction]
fn compile(module: &L1Module) -> PyResult<CompiledModule> {
    compiler::compile(module).map_err(PyValueError::new_err)
}

#[pyfunction]
fn int_to_bits(value: i64, n_bits: usize) -> Vec<bool> {
    vm::int_to_bits(value, n_bits)
}

#[pyfunction]
fn bits_to_uint(bits: Vec<bool>) -> u64 {
    vm::bits_to_uint(&bits)
}

#[pyfunction]
fn bits_to_int(bits: Vec<bool>) -> i64 {
    vm::bits_to_int(&bits)
}

#[pyfunction]
fn decimal_str_to_bits(s: &str, n_bits: usize) -> Vec<bool> {
    vm::decimal_str_to_bits(s, n_bits)
}

#[pyfunction]
fn signed_decimal_str_to_bits(s: &str, n_bits: usize) -> Vec<bool> {
    vm::signed_decimal_str_to_bits(s, n_bits)
}

#[pyfunction]
fn bits_to_uint_str(bits: Vec<bool>) -> String {
    vm::bits_to_uint_str(&bits)
}

#[pyfunction]
fn bits_to_int_str(bits: Vec<bool>) -> String {
    vm::bits_to_int_str(&bits)
}

#[pymodule]
fn _rsqlt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Span>()?;
    m.add_class::<LiteralNode>()?;
    m.add_class::<TupleNode>()?;
    m.add_class::<CArgNode>()?;
    m.add_class::<CObjectNode>()?;
    m.add_class::<QDTypeNode>()?;
    m.add_class::<QSignatureEntry>()?;
    m.add_class::<AliasAssignmentNode>()?;
    m.add_class::<QArgValueNode>()?;
    m.add_class::<QArgNode>()?;
    m.add_class::<QCallNode>()?;
    m.add_class::<QReturnNode>()?;
    m.add_class::<QDefImplNode>()?;
    m.add_class::<QDefExternNode>()?;
    m.add_class::<QCastNode>()?;
    m.add_class::<L1Module>()?;

    m.add_class::<SimState>()?;
    m.add_class::<PyVmSimulator>()?;
    m.add_class::<CompiledModule>()?;
    m.add_class::<CompiledSubroutine>()?;

    m.add_function(wrap_pyfunction!(parse_l1_module, m)?)?;

    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(int_to_bits, m)?)?;
    m.add_function(wrap_pyfunction!(bits_to_uint, m)?)?;
    m.add_function(wrap_pyfunction!(bits_to_int, m)?)?;
    m.add_function(wrap_pyfunction!(decimal_str_to_bits, m)?)?;
    m.add_function(wrap_pyfunction!(signed_decimal_str_to_bits, m)?)?;
    m.add_function(wrap_pyfunction!(bits_to_uint_str, m)?)?;
    m.add_function(wrap_pyfunction!(bits_to_int_str, m)?)?;

    Ok(())
}
