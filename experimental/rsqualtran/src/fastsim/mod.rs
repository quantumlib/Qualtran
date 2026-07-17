//! qlt_fastsim: A bytecode compiler and VM executor for simulating
//! quantum programs in the Qualtran IR (.qlt) format on classical basis-state inputs.
//!
//! # Architecture
//! - `compiler` — Translates L1Module AST into compiled subroutines
//! - `vm` — Executes compiled subroutines on a simulation state (bits + phase exponent)
//! - `gates` — Built-in gate implementations (X, CNOT, And, diagonal/phase gates, etc.)

pub mod compiler;
pub mod decompiler;
pub mod gates;
pub mod vm;
