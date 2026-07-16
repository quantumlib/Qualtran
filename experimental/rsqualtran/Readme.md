# rsqualtran

A Rust backend for Qualtran programs that can be used standalone via QLT IR or via Python bindings.

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

Uses [Maturin](https://www.maturin.rs/) + PyO3 to expose Python bindings, see
`python/rsqualtran/__init__.py` for the public API.

```bash
pip install maturin
maturin develop
```
