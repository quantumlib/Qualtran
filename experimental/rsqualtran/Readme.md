# rsqualtran

A Rust implementation of the Qualtran-L1 parser and classical simulator, with Python and
WebAssembly frontends.

The core library provides:

- **Parser** — Tokenizes and parses `.qlt` (Qualtran IR) source files into a typed AST
  (`L1Module`). The AST represents quantum definitions (`qdef`, `extern qdef`, `qcast`),
  their signatures, and their bodies (calls, aliases, returns, ...).

- **Visitor** — A trait-based AST visitor (`L1Visitor`) with default walker functions for
  every node type, allowing analyses and transforms over parsed modules.

- **Fastsim** — A bytecode compiler and stack-based VM for fast classical simulation of
  quantum programs. The compiler translates the parsed AST into subroutines containing
  bytecode instructions. The VM executes these against basis-state inputs — tracking a bit
  vector and an accumulated phase exponent. Supported gates include classical gates (X,
  CNOT, And, ...), diagonal/phase gates (Z, S, T, ...), and bookkeeping operations
  (Split, Join, Allocate, Free, ...).


## qlt_fastsim CLI

`qlt_fastsim` is a standalone binary for running fastsim from the command line.

```
qlt_fastsim [--timing] <file.qlt> <entrypoint>
```

Input is read from stdin — one line per simulation, with space-separated integer register
values in signature order. Outputs `register=value` pairs plus the accumulated `phase_exponent`
(the value `x` in `exp(iπx)`).

Multiple input lines produce one output line each (batch mode).

```bash
echo "5" | cargo run --bin qlt_fastsim --no-default-features -- example_qlts/negate.qlt Negate
# x=-5 phase_exponent=0
```

The `example_qlts/` directory contains sample `.qlt` programs and shell scripts for
exercising the simulator.

## Building and developing

### Prerequisites

- **Rust**: [Install Rust](https://www.rust-lang.org/tools/install)
- **Python 3.12+** with `venv` (only for Python bindings)
- **Node.js 18+** and **wasm-pack** (only for WebAssembly)

### Rust-only

Build and test the core library (parser + fastsim) without Python or WASM dependencies:

```bash
cargo build --no-default-features
cargo test --no-default-features
```

### Python bindings

Uses [Maturin](https://www.maturin.rs/) + PyO3 to expose the parser, AST nodes, and a
`BloqBuilder` implementation as a Python extension module.

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop
```

### WebAssembly

The parser can be compiled to WASM for use in web applications. The `javascript/`
directory contains a Vue.js app that demonstrates the WASM parser.

```bash
cd javascript
npm install
npm run dev  # builds WASM automatically, then starts the dev server
```
