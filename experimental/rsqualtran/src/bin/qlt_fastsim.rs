//! CLI binary for the qlt_fastsim quantum simulator.
//!
//! Usage: qlt_fastsim [--timing|--no-timing] <filename.qlt> <entrypoint>
//!
//! Reads input from stdin: one line per simulation, space-separated integers
//! in signature order. Lines support '#' comments (everything from '#' to end
//! of line is ignored). Blank and comment-only lines are skipped.
//!
//! Outputs a header line starting with '# phase_exponent <reg_name> ...' followed
//! by space-separated values, one line per simulation. Phase exponent is always
//! the first data column.
//!
//! When --timing is enabled, performance timers for each phase are printed to
//! stderr. Input parsing and execution are batched (BATCH_SIZE lines at a time)
//! so their timings can be reported per-batch and per-line.

use std::io::{self, BufRead, BufWriter, Write};
use std::time::{Duration, Instant};

/// Format a Duration with an appropriate unit and 5 significant figures.
///
/// Picks the largest unit (s, ms, µs, ns) that keeps the leading digit ≥ 1,
/// then prints exactly enough decimal places for 5 significant figures total.
fn format_duration(d: Duration) -> String {
    let nanos = d.as_nanos() as f64;
    let (value, unit) = if nanos < 1_000.0 {
        (nanos, "ns")
    } else if nanos < 1_000_000.0 {
        (nanos / 1_000.0, "µs")
    } else if nanos < 1_000_000_000.0 {
        (nanos / 1_000_000.0, "ms")
    } else {
        (nanos / 1_000_000_000.0, " s")
    };

    // Number of digits before the decimal point (at least 1).
    let digits_before = if value >= 1.0 {
        (value.log10().floor() as usize) + 1
    } else {
        1
    };
    let decimal_places = 5usize.saturating_sub(digits_before);
    format!("{:.prec$} {}", value, unit, prec = decimal_places)
}

const BATCH_SIZE: usize = 1000;

/// Parsed input for a single simulation line.
struct ParsedInput {
    input_values: Vec<(String, Vec<bool>)>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse optional flags and positional arguments.
    let mut timing = false;
    let mut compile_only = false;
    let mut positional: Vec<String> = Vec::new();

    for arg in &args[1..] {
        match arg.as_str() {
            "--timing" => timing = true,
            "--no-timing" => timing = false,
            "--compile-only" => compile_only = true,
            _ => {
                if arg.starts_with("--") {
                    eprintln!("Unknown flag: {}", arg);
                    std::process::exit(1);
                }
                positional.push(arg.clone());
            }
        }
    }

    if (compile_only && positional.is_empty()) || (!compile_only && positional.len() < 2) {
        eprintln!(
            "Usage: {} [--timing|--no-timing] [--compile-only] <filename.qlt> [entrypoint]",
            args[0]
        );
        std::process::exit(1);
    }

    let filename = &positional[0];
    let entrypoint = if !compile_only {
        positional[1].as_str()
    } else {
        ""
    };

    // ── Phase 1: Parse the .qlt file ──────────────────────────────────
    let t_parse_qlt = Instant::now();

    let source = match std::fs::read_to_string(filename) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", filename, e);
            std::process::exit(1);
        }
    };

    let (module, errors) = _rsqlt::parser::parse_l1_module(&source);
    if !errors.is_empty() {
        for err in &errors {
            eprintln!("Parse error: {}", err);
        }
        std::process::exit(1);
    }

    let elapsed_parse_qlt = t_parse_qlt.elapsed();

    // ── Phase 2: Compile to bytecode ──────────────────────────────────
    let t_compile = Instant::now();

    let compiled = match _rsqlt::fastsim::compiler::compile(&module) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Compilation error: {}", e);
            std::process::exit(1);
        }
    };

    let elapsed_compile = t_compile.elapsed();

    if compile_only {
        if timing {
            eprintln!(
                "[timing] parse_qlt:    {}",
                format_duration(elapsed_parse_qlt)
            );
            eprintln!(
                "[timing] compile:      {}",
                format_duration(elapsed_compile)
            );
        }
        match _rsqlt::fastsim::decompiler::decompile(&compiled) {
            Ok(assembly) => {
                let mut stdout = io::stdout().lock();
                if let Err(e) = write!(stdout, "{}", assembly) {
                    eprintln!("Error writing stdout: {}", e);
                    std::process::exit(1);
                }
                if let Err(e) = stdout.flush() {
                    eprintln!("Error flushing stdout: {}", e);
                    std::process::exit(1);
                }
                std::process::exit(0);
            }
            Err(e) => {
                eprintln!("Decompilation error: {}", e);
                std::process::exit(1);
            }
        }
    }

    // Find the entrypoint subroutine to determine its signature.
    let sub = if let Some(sub) = compiled.get_subroutine(entrypoint) {
        sub
    } else {
        let with_parens = format!("{}()", entrypoint);
        if let Some(sub) = compiled.get_subroutine(&with_parens) {
            sub
        } else {
            eprintln!(
                "Entrypoint '{}' not found. Available: {:?}",
                entrypoint,
                compiled
                    .subroutines
                    .iter()
                    .map(|s| compiled.intern_table.resolve(s.bloq_key))
                    .collect::<Vec<_>>()
            );
            std::process::exit(1);
        }
    };

    // Determine input registers (Thru or LeftOnly direction).
    let input_regs: Vec<_> = sub
        .signature
        .iter()
        .filter(|r| {
            r.direction == _rsqlt::fastsim::compiler::RegisterDirection::Thru
                || r.direction == _rsqlt::fastsim::compiler::RegisterDirection::LeftOnly
        })
        .collect();

    // Determine output register names (Thru, RightOnly, or Cast — everything
    // except LeftOnly, matching extract_outputs).
    let output_reg_names: Vec<&str> = sub
        .signature
        .iter()
        .filter(|r| r.direction != _rsqlt::fastsim::compiler::RegisterDirection::LeftOnly)
        .map(|r| compiled.intern_table.resolve(r.name))
        .collect();

    if timing {
        eprintln!(
            "[timing] parse_qlt:    {}",
            format_duration(elapsed_parse_qlt)
        );
        eprintln!(
            "[timing] compile:      {}",
            format_duration(elapsed_compile)
        );
    }

    // ── Phases 3 & 4: Batch input parsing + execution ─────────────────
    let mut simulator = match _rsqlt::fastsim::vm::VmSimulator::new(
        &compiled,
        compiled.intern_table.resolve(sub.bloq_key),
    ) {
        Ok(sim) => sim,
        Err(e) => {
            eprintln!("Simulator initialization error: {}", e);
            std::process::exit(1);
        }
    };

    // ── Emit header line ──────────────────────────────────────────────
    {
        let mut stdout = io::stdout().lock();
        let mut header_parts: Vec<&str> = Vec::with_capacity(1 + output_reg_names.len());
        header_parts.push("phase_exponent");
        header_parts.extend_from_slice(&output_reg_names);
        if let Err(e) = writeln!(stdout, "# {}", header_parts.join(" ")) {
            eprintln!("Error writing header: {}", e);
            std::process::exit(1);
        }
        if let Err(e) = stdout.flush() {
            eprintln!("Error flushing header: {}", e);
            std::process::exit(1);
        }
    }

    let stdin = io::stdin();
    let mut lines_iter = stdin.lock().lines();

    loop {
        // ── Phase 3: Parse a batch of input lines ─────────────────────
        let t_parse_inputs = Instant::now();

        let mut batch: Vec<ParsedInput> = Vec::with_capacity(BATCH_SIZE);
        let mut done = false;

        while batch.len() < BATCH_SIZE {
            match lines_iter.next() {
                None => {
                    done = true;
                    break;
                }
                Some(line_result) => {
                    let line = match line_result {
                        Ok(l) => l,
                        Err(e) => {
                            eprintln!("Error reading stdin: {}", e);
                            std::process::exit(1);
                        }
                    };

                    // Strip comments: everything from '#' to end of line.
                    let without_comment = match line.find('#') {
                        Some(pos) => &line[..pos],
                        None => &line,
                    };
                    let trimmed = without_comment.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    let values: Vec<&str> = trimmed.split_whitespace().collect();

                    if values.len() != input_regs.len() {
                        eprintln!(
                            "Expected {} input values ({}) but got {}",
                            input_regs.len(),
                            input_regs
                                .iter()
                                .map(|r| compiled.intern_table.resolve(r.name))
                                .collect::<Vec<_>>()
                                .join(", "),
                            values.len()
                        );
                        std::process::exit(1);
                    }

                    let input_values: Vec<(String, Vec<bool>)> = input_regs
                        .iter()
                        .zip(values.iter())
                        .map(|(reg, &val)| {
                            let bits = if reg.is_signed(&compiled.intern_table) {
                                _rsqlt::fastsim::vm::signed_decimal_str_to_bits(val, reg.n_bits)
                            } else {
                                _rsqlt::fastsim::vm::decimal_str_to_bits(val, reg.n_bits)
                            };
                            (compiled.intern_table.resolve(reg.name).to_string(), bits)
                        })
                        .collect();

                    batch.push(ParsedInput { input_values });
                }
            }
        }

        let n_lines = batch.len();

        // Nothing left to process.
        if n_lines == 0 {
            break;
        }

        let elapsed_parse_inputs = t_parse_inputs.elapsed();

        // ── Phase 4: Execute the batch ────────────────────────────────
        let t_execute = Instant::now();
        let mut results = Vec::with_capacity(batch.len());

        for parsed in &batch {
            if let Err(e) = simulator.execute_run(&parsed.input_values) {
                eprintln!("Execution error: {}", e);
                std::process::exit(1);
            }
            results.push((simulator.extract_outputs(), simulator.phase_exponent()));
        }

        let elapsed_execute = t_execute.elapsed();

        // ── Phase 5: Print outputs ────────────────────────────────────
        let t_print = Instant::now();
        let mut stdout = BufWriter::new(io::stdout().lock());

        for (outputs, phase_exponent) in results {
            let mut parts: Vec<String> = Vec::with_capacity(1 + outputs.len());

            // Phase exponent first (the value x in exp(iπx)).
            if phase_exponent == 0.0 {
                parts.push("0".to_string());
            } else if phase_exponent == phase_exponent.floor() && phase_exponent.abs() < 1e15 {
                parts.push(format!("{}", phase_exponent as i64));
            } else {
                parts.push(format!("{}", phase_exponent));
            }

            // Then output register values.
            for (_name, value_str, _dtype) in &outputs {
                parts.push(value_str.clone());
            }

            if let Err(e) = writeln!(stdout, "{}", parts.join(" ")) {
                eprintln!("Error writing stdout: {}", e);
                std::process::exit(1);
            }
        }

        if let Err(e) = stdout.flush() {
            eprintln!("Error flushing stdout: {}", e);
            std::process::exit(1);
        }

        let elapsed_print_outputs = t_print.elapsed();

        if timing {
            eprintln!(
                "[timing] parse_inputs: {}    total, {}/line ({} lines)",
                format_duration(elapsed_parse_inputs),
                format_duration(elapsed_parse_inputs / n_lines as u32),
                n_lines,
            );
            eprintln!(
                "[timing] execute:      {}    total, {}/line ({} lines)",
                format_duration(elapsed_execute),
                format_duration(elapsed_execute / n_lines as u32),
                n_lines,
            );
            eprintln!(
                "[timing] print_outputs: {}    total, {}/line ({} lines)",
                format_duration(elapsed_print_outputs),
                format_duration(elapsed_print_outputs / n_lines as u32),
                n_lines,
            );
        }

        if done {
            break;
        }
    }
}
