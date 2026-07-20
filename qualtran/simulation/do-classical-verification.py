#!/usr/bin/env python
#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Structural induction verification for arithmetic and modular bloqs.

Verifies bloqs by cross-checking each bloq's reference implementation
(`on_classical_vals`) against recursive decomposition simulation.
The BFS traversal automatically discovers and verifies all sub-bloqs in the
decomposition tree.

Test cases are registered per bloq class and collected into a registry with
short, friendly names. The CLI supports selecting specific bloqs by name,
or using `--all` with optional `--exclude`.

Environment Variables:
    QLT_USE_FASTSIM: Controls the classical simulation engine:
        - Set to '1', 'true', or 'yes' to explicitly use the Rust `rsqualtran` fastsim engine.
        - Set to '0', 'false', or 'no' to explicitly use Python simulation (`simulate_via_subbloq_classical_vals`).
        - If unset, fastsim is used automatically when `rsqualtran` is installed, falling back to Python otherwise.

Usage:
    # Verify specific bloqs:
    python -m qualtran.simulation.do-classical-verification product square

    # Verify all registered bloqs using Python simulation engine explicitly:
    QLT_USE_FASTSIM=0 python -m qualtran.simulation.do-classical-verification --all

    # Verify all except one:
    python -m qualtran.simulation.do-classical-verification --all --exclude square
"""

import argparse
import sys

import numpy as np

from qualtran.simulation.verification import (
    _should_use_fastsim,
    ClassicalSimTestCase,
    print_verification_summary,
    TestCaseProvider,
    validate_test_cases,
    verify_structural_induction,
)

# ---------------------------------------------------------------------------
# Registry: maps short lowercase names to test-case provider functions.
#
# To add a new bloq, import its provider function and add an entry here.
# ---------------------------------------------------------------------------


def _build_registry() -> dict[str, TestCaseProvider]:
    """Build the registry of test-case providers."""
    # fmt: off
    # isort: off
    from qualtran.bloqs.arithmetic.addition import (
        _get_add_k_classical_sim_test_cases,
    )
    from qualtran.bloqs.arithmetic.bitwise import (
        _get_bitwise_not_classical_sim_test_cases,
        _get_xor_classical_sim_test_cases,
        _get_xork_classical_sim_test_cases,
    )
    from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
        _get_controlled_add_or_subtract_classical_sim_test_cases,
    )
    from qualtran.bloqs.arithmetic.negate import (
        _get_negate_classical_sim_test_cases,
    )
    from qualtran.bloqs.basic_gates.on_each import (
        _get_on_each_classical_sim_test_cases,
    )
    # isort: on
    # fmt: on

    return {
        'add_k': _get_add_k_classical_sim_test_cases,
        'bitwise_not': _get_bitwise_not_classical_sim_test_cases,
        'controlled_add_or_subtract': _get_controlled_add_or_subtract_classical_sim_test_cases,
        'negate': _get_negate_classical_sim_test_cases,
        'on_each': _get_on_each_classical_sim_test_cases,
        'xor': _get_xor_classical_sim_test_cases,
        'xork': _get_xork_classical_sim_test_cases,
        'xor_k': _get_xork_classical_sim_test_cases,
    }


def main() -> None:
    registry = _build_registry()
    available = sorted(registry)

    epilog = (
        "environment variables:\n"
        "  QLT_USE_FASTSIM    '1' to force Rust fastsim, '0' to force Python simulation.\n"
        "                     If unset, uses fastsim when available (`rsqualtran` installed).\n\n"
        "available bloqs:\n  " + "\n  ".join(available)
    )
    parser = argparse.ArgumentParser(
        description="Verify bloqs via structural induction.\n\n"
        "Simulation behavior can be controlled via the QLT_USE_FASTSIM environment variable.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument(
        "names", nargs="*", metavar="NAME", help="Bloq names to verify (e.g. 'product', 'square')."
    )
    parser.add_argument(
        "--all", action="store_true", dest="run_all", help="Verify all registered bloqs."
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        metavar="NAME",
        help="Bloq names to exclude (only meaningful with --all).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1_000,
        help="Number of random quantum input samples per bloq. "
        "If this exceeds the total domain size, exhaustive enumeration is used.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # --- Resolve which bloqs to verify. ---
    if args.run_all:
        selected = [n for n in available if n not in args.exclude]
        # Validate --exclude names.
        bad_excludes = [n for n in args.exclude if n not in registry]
        if bad_excludes:
            parser.error(
                f"Unknown bloq name(s) in --exclude: {bad_excludes}. " f"Available: {available}"
            )
    elif args.names:
        selected = args.names
        # Validate positional names.
        bad_names = [n for n in selected if n not in registry]
        if bad_names:
            parser.error(f"Unknown bloq name(s): {bad_names}. " f"Available: {available}")
        if args.exclude:
            parser.error("--exclude can only be used with --all.")
    else:
        parser.error("Specify bloq names to verify, or use --all. " f"Available: {available}")

    if not selected:
        print("Nothing to verify (all bloqs excluded).", file=sys.stderr)
        return

    # --- Collect and validate test cases. ---
    all_test_cases: list[ClassicalSimTestCase] = []
    for name in selected:
        provider = registry[name]
        cases = provider()
        validate_test_cases(name, cases)
        all_test_cases.extend(cases)

    # --- Run verification. ---
    print("=" * 70)
    print("Structural Induction Verification")
    print("=" * 70)
    print(f"  bloqs:     {', '.join(selected)}")
    print(f"  engine:    {'QLTFastsim (Rust)' if _should_use_fastsim() else 'Python'}")
    print(f"  n_samples: {args.n_samples}")
    print(f"  seed:      {args.seed}")
    print()

    print(f"Loaded {len(all_test_cases)} root test cases:")
    for tc in all_test_cases:
        print(f"  - {tc.name}")
    print()

    rng = np.random.default_rng(args.seed)
    results = verify_structural_induction(
        root_cases=all_test_cases, n_samples=args.n_samples, rng=rng
    )

    print_verification_summary(results)


if __name__ == "__main__":
    main()
