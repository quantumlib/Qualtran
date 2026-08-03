#[cfg(feature = "py")]
use pyo3::prelude::*;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};

use std::fmt::{self};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "wasm", derive(ts_rs::TS))]
#[cfg_attr(feature = "wasm", ts(export))]
#[cfg_attr(feature = "py", pyo3::pyclass(get_all, set_all, from_py_object))]
pub struct Span {
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize,
}

#[cfg(feature = "py")]
#[pymethods]
impl Span {
    #[new]
    #[pyo3(signature = (start_line=0, start_col=0, end_line=0, end_col=0))]
    fn new(start_line: usize, start_col: usize, end_line: usize, end_col: usize) -> Self {
        Span {
            start_line,
            start_col,
            end_line,
            end_col,
        }
    }
}

// ============================================================================
//  1. CValue Hierarchy
// ============================================================================

/// Represents the `Union[int, float, str]` for LiteralNode
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "wasm", derive(ts_rs::TS))]
#[cfg_attr(feature = "wasm", ts(export))]
#[cfg_attr(feature = "py", derive(FromPyObject))]
pub enum LiteralVal {
    Int(i64),
    Float(f64),
    String(String),
}

#[cfg(feature = "py")]
impl<'py> IntoPyObject<'py> for LiteralVal {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            LiteralVal::Int(v) => Ok(v.into_pyobject(py).unwrap().into_any()),
            LiteralVal::Float(v) => Ok(v.into_pyobject(py).unwrap().into_any()),
            LiteralVal::String(v) => Ok(v.into_pyobject(py).unwrap().into_any()),
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "wasm", derive(ts_rs::TS))]
#[cfg_attr(feature = "wasm", ts(export))]
#[cfg_attr(feature = "py", pyo3::pyclass(get_all, set_all, from_py_object))]
pub struct LiteralNode {
    pub value: LiteralVal,
    pub span: Span,
}

impl fmt::Display for LiteralNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.value {
            LiteralVal::Int(x) => write!(f, "{}", x),
            LiteralVal::Float(x) => write!(f, "{}", x),
            LiteralVal::String(x) => write!(f, "{}", x),
        }
    }
}

#[cfg(feature = "py")]
#[pymethods]
impl LiteralNode {
    #[new]
    #[pyo3(signature = (value, span=None))]
    fn new(value: LiteralVal, span: Option<Span>) -> Self {
        LiteralNode {
            value,
            span: span.unwrap_or_default(),
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "wasm", derive(ts_rs::TS))]
#[cfg_attr(feature = "wasm", ts(export))]
#[cfg_attr(feature = "py", pyo3::pyclass(get_all, set_all, from_py_object))]
pub struct TupleNode {
    pub items: Vec<CValueNode>,
    pub span: Span,
}

#[cfg(feature = "py")]
#[pymethods]
impl TupleNode {
    #[new]
    #[pyo3(signature = (items, span=None))]
    fn new(items: Vec<CValueNode>, span: Option<Span>) -> Self {
        TupleNode {
            items,
            span: span.unwrap_or_default(),
        }
    }
}

impl fmt::Display for TupleNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "wasm", derive(ts_rs::TS))]
#[cfg_attr(feature = "wasm", ts(export))]
#[cfg_attr(feature = "py", pyo3::pyclass(get_all, set_all, from_py_object))]
pub struct CObjectNode {
    pub name: String,
    pub cargs: Vec<CArgNode>,
    pub span: Span,
}

impl fmt::Display for CObjectNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)?;
        if !self.cargs.is_empty() {
            let joined = self
                .cargs
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            write!(f, "({})", joined)?;
        }
        Ok(())
    }
}

#[cfg(feature = "py")]
#[pymethods]
impl CObjectNode {
    #[new]
    #[pyo3(signature = (name, cargs, span=None))]
    fn new(name: String, cargs: Vec<CArgNode>, span: Option<Span>) -> Self {
        CObjectNode {
            name,
            cargs,
            span: span.unwrap_or_default(),
        }
    }

    fn __str__(&self) -> String {
        self.to_string()
    }
}

/// The Rust equivalent of the `CValueNode` abstract base class.
/// We use an Enum to hold the possible variants.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "wasm", derive(ts_rs::TS))]
#[cfg_attr(feature = "wasm", ts(export))]
#[cfg_attr(feature = "py", derive(FromPyObject))]
pub enum CValueNode {
    Literal(LiteralNode),
    Tuple(TupleNode),
    CObject(CObjectNode),
}

impl CValueNode {
    pub fn span(&self) -> Span {
        match self {
            CValueNode::Literal(n) => n.span,
            CValueNode::Tuple(n) => n.span,
            CValueNode::CObject(n) => n.span,
        }
    }
}

#[cfg(feature = "py")]
impl<'py> IntoPyObject<'py> for CValueNode {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            CValueNode::Literal(n) => n.into_pyobject(py).map(|b| b.into_any()),
            CValueNode::Tuple(n) => n.into_pyobject(py).map(|b| b.into_any()),
            CValueNode::CObject(n) => n.into_pyobject(py).map(|b| b.into_any()),
        }
    }
}

impl fmt::Display for CValueNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CValueNode::Literal(n) => write!(f, "{}", n),
            CValueNode::Tuple(n) => write!(f, "{}", n),
            CValueNode::CObject(n) => write!(f, "{}", n),
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "wasm", derive(ts_rs::TS))]
#[cfg_attr(feature = "wasm", ts(export))]
#[cfg_attr(feature = "py", pyo3::pyclass(get_all, set_all, from_py_object))]
pub struct CArgNode {
    pub key: Option<String>,
    pub value: CValueNode,
    pub span: Span,
}

impl fmt::Display for CArgNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.key {
            Some(k) => write!(f, "{}={}", k, self.value),
            None => write!(f, "{}", self.value),
        }
    }
}

#[cfg(feature = "py")]
#[pymethods]
impl CArgNode {
    #[new]
    #[pyo3(signature = (value, key=None, span=None))]
    fn new(value: CValueNode, key: Option<String>, span: Option<Span>) -> Self {
        CArgNode {
            key,
            value,
            span: span.unwrap_or_default(),
        }
    }
}

// ============================================================================
//  2. Quantum Types & Signatures
// ============================================================================

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "wasm", derive(ts_rs::TS))]
#[cfg_attr(feature = "wasm", ts(export))]
#[cfg_attr(feature = "py", pyo3::pyclass(get_all, set_all, from_py_object))]
pub struct QDTypeNode {
    pub dtype: CObjectNode,
    pub shape: Option<Vec<i64>>,
    pub span: Span,
}

#[cfg(feature = "py")]
#[pymethods]
impl QDTypeNode {
    #[new]
    #[pyo3(signature = (dtype, shape=None, span=None))]
    fn new(dtype: CObjectNode, shape: Option<Vec<i64>>, span: Option<Span>) -> Self {
        QDTypeNode {
            dtype,
            shape,
            span: span.unwrap_or_default(),
        }
    }
}

/// Represents `Union[QDTypeNode, Tuple[Optional[QDTypeNode], Optional[QDTypeNode]]]`
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "wasm", derive(ts_rs::TS))]
#[cfg_attr(feature = "wasm", ts(export))]
#[cfg_attr(feature = "py", derive(FromPyObject))]
pub enum SignatureDType {
    Single(QDTypeNode),
    // Maps to (left, right) tuple in Python
    Pair((Option<QDTypeNode>, Option<QDTypeNode>)),
}

#[cfg(feature = "py")]
impl<'py> IntoPyObject<'py> for SignatureDType {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            SignatureDType::Single(n) => n.into_pyobject(py).map(|b| b.into_any()),
            SignatureDType::Pair(tuple) => tuple.into_pyobject(py).map(|b| b.into_any()),
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "wasm", derive(ts_rs::TS))]
#[cfg_attr(feature = "wasm", ts(export))]
#[cfg_attr(feature = "py", pyo3::pyclass(get_all, set_all, from_py_object))]
pub struct QSignatureEntry {
    pub name: String,
    pub dtype: SignatureDType,
    pub span: Span,
}

#[cfg(feature = "py")]
#[pymethods]
impl QSignatureEntry {
    #[new]
    #[pyo3(signature = (name, dtype, span=None))]
    fn new(name: String, dtype: SignatureDType, span: Option<Span>) -> Self {
        QSignatureEntry {
            name,
            dtype,
            span: span.unwrap_or_default(),
        }
    }
}

// ============================================================================
//  3. Arguments (QArgs)
// ============================================================================

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "wasm", derive(ts_rs::TS))]
#[cfg_attr(feature = "wasm", ts(export))]
#[cfg_attr(feature = "py", pyo3::pyclass(get_all, set_all, from_py_object))]
pub struct QArgValueNode {
    pub name: String,
    pub idx: Vec<i64>,
    pub span: Span,
}

#[cfg(feature = "py")]
#[pymethods]
impl QArgValueNode {
    #[new]
    #[pyo3(signature = (name, idx, span=None))]
    fn new(name: String, idx: Vec<i64>, span: Option<Span>) -> Self {
        QArgValueNode {
            name,
            idx,
            span: span.unwrap_or_default(),
        }
    }
}

/// Recursive Type: `NestedQArgValue`
/// We use Box to allow the recursive definition size.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "wasm", derive(ts_rs::TS))]
#[cfg_attr(feature = "wasm", ts(export))]
#[cfg_attr(feature = "py", derive(FromPyObject))]
pub enum NestedQArgValue {
    Leaf(QArgValueNode),
    List(Vec<NestedQArgValue>),
}

#[cfg(feature = "py")]
impl<'py> IntoPyObject<'py> for NestedQArgValue {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            NestedQArgValue::Leaf(n) => n.into_pyobject(py).map(|b| b.into_any()),
            NestedQArgValue::List(l) => l.into_pyobject(py).map(|b| b.into_any()),
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "wasm", derive(ts_rs::TS))]
#[cfg_attr(feature = "wasm", ts(export))]
#[cfg_attr(feature = "py", pyo3::pyclass(get_all, set_all, from_py_object))]
pub struct QArgNode {
    pub key: String,
    pub value: NestedQArgValue,
    pub span: Span,
}

#[cfg(feature = "py")]
#[pymethods]
impl QArgNode {
    #[new]
    #[pyo3(signature = (key, value, span=None))]
    fn new(key: String, value: NestedQArgValue, span: Option<Span>) -> Self {
        QArgNode {
            key,
            value,
            span: span.unwrap_or_default(),
        }
    }
}

// ============================================================================
//  4. Statements
// ============================================================================

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "wasm", derive(ts_rs::TS))]
#[cfg_attr(feature = "wasm", ts(export))]
#[cfg_attr(feature = "py", pyo3::pyclass(get_all, set_all, from_py_object))]
pub struct AliasAssignmentNode {
    pub alias: String,
    pub bloq_key: String,
    pub span: Span,
}

#[cfg(feature = "py")]
#[pymethods]
impl AliasAssignmentNode {
    #[new]
    #[pyo3(signature = (alias, bloq_key, span=None))]
    fn new(alias: String, bloq_key: String, span: Option<Span>) -> Self {
        AliasAssignmentNode {
            alias,
            bloq_key,
            span: span.unwrap_or_default(),
        }
    }

    fn __str__(&self) -> String {
        format!("[AA] {} = {}", self.alias, self.bloq_key)
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "wasm", derive(ts_rs::TS))]
#[cfg_attr(feature = "wasm", ts(export))]
#[cfg_attr(feature = "py", pyo3::pyclass(get_all, set_all, from_py_object))]
pub struct QCallNode {
    pub bloq_key: String,
    pub lvalues: Vec<String>,
    pub qargs: Vec<QArgNode>,
    pub span: Span,
}

#[cfg(feature = "py")]
#[pymethods]
impl QCallNode {
    #[new]
    #[pyo3(signature = (bloq_key, lvalues, qargs, span=None))]
    fn new(
        bloq_key: String,
        lvalues: Vec<String>,
        qargs: Vec<QArgNode>,
        span: Option<Span>,
    ) -> Self {
        QCallNode {
            bloq_key,
            lvalues,
            qargs,
            span: span.unwrap_or_default(),
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "wasm", derive(ts_rs::TS))]
#[cfg_attr(feature = "wasm", ts(export))]
#[cfg_attr(feature = "py", pyo3::pyclass(get_all, set_all, from_py_object))]
pub struct QReturnNode {
    pub ret_mapping: Vec<QArgNode>,
    pub span: Span,
}

#[cfg(feature = "py")]
#[pymethods]
impl QReturnNode {
    #[new]
    #[pyo3(signature = (ret_mapping, span=None))]
    fn new(ret_mapping: Vec<QArgNode>, span: Option<Span>) -> Self {
        QReturnNode {
            ret_mapping,
            span: span.unwrap_or_default(),
        }
    }
}

/// Abstract Base Class `StatementNode`
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "wasm", derive(ts_rs::TS))]
#[cfg_attr(feature = "wasm", ts(export))]
#[cfg_attr(feature = "py", derive(FromPyObject))]
pub enum StatementNode {
    Alias(AliasAssignmentNode),
    Call(QCallNode),
    Return(QReturnNode),
}

impl StatementNode {
    pub fn span(&self) -> Span {
        match self {
            StatementNode::Alias(n) => n.span,
            StatementNode::Call(n) => n.span,
            StatementNode::Return(n) => n.span,
        }
    }
}

#[cfg(feature = "py")]
impl<'py> IntoPyObject<'py> for StatementNode {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            StatementNode::Alias(n) => n.into_pyobject(py).map(|b| b.into_any()),
            StatementNode::Call(n) => n.into_pyobject(py).map(|b| b.into_any()),
            StatementNode::Return(n) => n.into_pyobject(py).map(|b| b.into_any()),
        }
    }
}

// ============================================================================
//  5. QDefs & Modules
// ============================================================================

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "wasm", derive(ts_rs::TS))]
#[cfg_attr(feature = "wasm", ts(export))]
#[cfg_attr(feature = "py", pyo3::pyclass(get_all, set_all, from_py_object))]
pub struct QDefImplNode {
    pub bloq_key: String,
    pub qsignature: Vec<QSignatureEntry>,
    pub body: Vec<StatementNode>,
    pub cobject_from: Option<CObjectNode>,
    pub span: Span,
}

#[cfg(feature = "py")]
#[pymethods]
impl QDefImplNode {
    #[new]
    #[pyo3(signature = (bloq_key, qsignature, body, cobject_from=None, span=None))]
    fn new(
        bloq_key: String,
        qsignature: Vec<QSignatureEntry>,
        body: Vec<StatementNode>,
        cobject_from: Option<CObjectNode>,
        span: Option<Span>,
    ) -> Self {
        QDefImplNode {
            bloq_key,
            qsignature,
            body,
            cobject_from,
            span: span.unwrap_or_default(),
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "wasm", derive(ts_rs::TS))]
#[cfg_attr(feature = "wasm", ts(export))]
#[cfg_attr(feature = "py", pyo3::pyclass(get_all, set_all, from_py_object))]
pub struct QDefExternNode {
    pub bloq_key: String,
    pub qsignature: Vec<QSignatureEntry>,
    pub cobject_from: CObjectNode,
    pub span: Span,
}

#[cfg(feature = "py")]
#[pymethods]
impl QDefExternNode {
    #[new]
    #[pyo3(signature = (bloq_key, qsignature, cobject_from, span=None))]
    fn new(
        bloq_key: String,
        qsignature: Vec<QSignatureEntry>,
        cobject_from: CObjectNode,
        span: Option<Span>,
    ) -> Self {
        QDefExternNode {
            bloq_key,
            qsignature,
            cobject_from,
            span: span.unwrap_or_default(),
        }
    }
}

/// A qcast declaring a casting (bookkeeping) operation.
///
/// A `qcast` is syntactic sugar for bookkeeping bloqs such as Split, Join,
/// Partition, Destructure, and Restructure. Unlike `extern qdef`, it does
/// *not* have a `from` clause — the casting is fully determined by the
/// quantum signature.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "wasm", derive(ts_rs::TS))]
#[cfg_attr(feature = "wasm", ts(export))]
#[cfg_attr(feature = "py", pyo3::pyclass(get_all, set_all, from_py_object))]
pub struct QCastNode {
    pub bloq_key: String,
    pub qsignature: Vec<QSignatureEntry>,
    pub span: Span,
}

#[cfg(feature = "py")]
#[pymethods]
impl QCastNode {
    #[new]
    #[pyo3(signature = (bloq_key, qsignature, span=None))]
    fn new(bloq_key: String, qsignature: Vec<QSignatureEntry>, span: Option<Span>) -> Self {
        QCastNode {
            bloq_key,
            qsignature,
            span: span.unwrap_or_default(),
        }
    }
}

/// Abstract Base Class `QDefNode`
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "wasm", derive(ts_rs::TS))]
#[cfg_attr(feature = "wasm", ts(export))]
#[cfg_attr(feature = "py", derive(FromPyObject))]
pub enum QDefNode {
    Impl(QDefImplNode),
    Extern(QDefExternNode),
    Cast(QCastNode),
}

impl QDefNode {
    pub fn span(&self) -> Span {
        match self {
            QDefNode::Impl(n) => n.span,
            QDefNode::Extern(n) => n.span,
            QDefNode::Cast(n) => n.span,
        }
    }
}

#[cfg(feature = "py")]
impl<'py> IntoPyObject<'py> for QDefNode {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            QDefNode::Impl(n) => n.into_pyobject(py).map(|b| b.into_any()),
            QDefNode::Extern(n) => n.into_pyobject(py).map(|b| b.into_any()),
            QDefNode::Cast(n) => n.into_pyobject(py).map(|b| b.into_any()),
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "wasm", derive(ts_rs::TS))]
#[cfg_attr(feature = "wasm", ts(export))]
#[cfg_attr(feature = "py", pyo3::pyclass(get_all, set_all, from_py_object))]
pub struct L1Module {
    pub qdefs: Vec<QDefNode>,
}

#[cfg(feature = "py")]
#[pymethods]
impl L1Module {
    #[new]
    fn new(qdefs: Vec<QDefNode>) -> Self {
        L1Module { qdefs }
    }
}
