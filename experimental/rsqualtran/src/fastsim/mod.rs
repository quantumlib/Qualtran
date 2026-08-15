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
