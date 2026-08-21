#  Copyright 2026 Google LLC
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
"""Build an on-disk library of Qualtran-L1 (`.qlt`) files from every `BloqExample`.

This is a thin driver around `qualtran.l1.build_library_entry`. It discovers every
`BloqExample` via `qualtran_dev_tools.bloq_finder`, constructs each one, and hands
the resulting `(bloq, name)` pair to the library, which compiles, reloads, and
executes it, filing the `.qlt` file under `<output_dir>/lib/` (on success) or
`<output_dir>/partial/` (on a soft failure); see `qualtran.l1._library`.

The dividing line: everything that understands `BloqExample` lives here; the
per-bloq compile/verify flow lives in `qualtran.l1`.

Each example lands in exactly one `BuildOutcome`. In addition to the outcomes the
library reports, this driver assigns:

 - `construct_failed`: `BloqExample.make()` raised (before the bloq ever reaches
   the library).

Usage:

```
python dev_tools/build-l1-library.py /tmp/qlt                 # build everything
python dev_tools/build-l1-library.py /tmp/qlt --limit 20      # first 20 examples
python dev_tools/build-l1-library.py /tmp/qlt --regenerate    # overwrite existing files
python dev_tools/build-l1-library.py /tmp/qlt --report report.md  # write a markdown report
```
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List

from qualtran_dev_tools.bloq_finder import get_bloq_examples

from qualtran.l1 import build_library_entry, BuildOutcome, L1BuildResult

# Regex that matches 'symb' or 'symbolic' as a standalone word in snake_case names.
#
# Word-boundary matching for snake_case identifier names:
#  - `(?<![a-zA-Z0-9])` asserts the preceding character is not an alphanumeric character.
#  - `(?:symbolic|symb)` matches either 'symbolic' or 'symb'.
#  - `(?![a-zA-Z0-9])` asserts the following character is not an alphanumeric character.
#
# This matches identifier words like `bloq_ex_symb`, `bloq_ex_symb_small`,
# `bloq_ex_symbolic`, and `symbolic_qft`, but does NOT match words where
# 'symb' is a substring of a larger word, such as `bloq_ex_symbiotic` or `bloq_symbiotic_ex`.
SYMBOLIC_NAME_PATTERN = re.compile(
    r'(?<![a-zA-Z0-9])(?:symbolic|symb)(?![a-zA-Z0-9])', re.IGNORECASE
)


def build_all(
    out_dir: Path,
    *,
    limit: int | None = None,
    timeout: float | None = None,
    regenerate: bool = False,
    skip_symbolic: bool = True,
) -> List[L1BuildResult]:
    """Build a library entry for every (or the first `limit`) `BloqExample`.

    Args:
        out_dir: Root directory into which `.qlt` files are written.
        limit: If given, only process the first `limit` examples.
        timeout: Per-example time budget in seconds passed to `build_library_entry`.
        regenerate: Whether to overwrite existing `.qlt` files.
        skip_symbolic: Whether to skip examples whose name matches `SYMBOLIC_NAME_PATTERN`.

    Returns:
        A list of `L1BuildResult`, one per example, in discovery order.
    """
    examples = get_bloq_examples()
    if limit is not None:
        examples = examples[:limit]

    results: List[L1BuildResult] = []
    total = len(examples)
    for i, be in enumerate(examples):
        print(f'[{i + 1:4d}/{total}] {be.name} ...', end='', flush=True)
        if skip_symbolic and SYMBOLIC_NAME_PATTERN.search(be.name):
            result = L1BuildResult(
                be.name, BuildOutcome.SKIPPED, 'Skipped symbolic example'
            )
        else:
            try:
                bloq = be.make()
            except Exception as e:  # pylint: disable=broad-except
                summary = f'{type(e).__name__}: {e}'.replace('\n', ' ')[:300]
                result = L1BuildResult(
                    be.name, BuildOutcome.CONSTRUCT_FAILED, summary, traceback.format_exc()
                )
            else:
                result = build_library_entry(
                    bloq, be.name, out_dir, timeout=timeout, regenerate=regenerate
                )
        results.append(result)
        print(f' {result.outcome.value}')
    return results


def print_summary(results: List[L1BuildResult]) -> None:
    """Print a categorized summary of build results to stdout."""
    by_outcome: Dict[BuildOutcome, List[L1BuildResult]] = {o: [] for o in BuildOutcome}
    for r in results:
        by_outcome[r.outcome].append(r)

    print('\n' + '=' * 72)
    print(f'Qualtran-L1 library build: {len(results)} BloqExample(s)')
    print('=' * 72)
    for outcome in BuildOutcome:
        print(f'  {outcome.value:26s}: {len(by_outcome[outcome]):4d}')
    print('=' * 72)

    # Detail every non-success, non-skipped outcome.
    for outcome in BuildOutcome:
        if outcome in (BuildOutcome.SUCCESS, BuildOutcome.SKIPPED) or not by_outcome[outcome]:
            continue
        print(f'\n### {outcome.value} ({len(by_outcome[outcome])})')
        for r in by_outcome[outcome]:
            print(f'  - {r.name}: {r.detail}')


def write_report(results: List[L1BuildResult], path: Path) -> None:
    """Write a markdown report of the build to `path`."""
    by_outcome: Dict[BuildOutcome, List[L1BuildResult]] = {o: [] for o in BuildOutcome}
    for r in results:
        by_outcome[r.outcome].append(r)

    lines: List[str] = []
    lines.append('# Qualtran-L1 Library Build Report\n')
    lines.append(f'Built **{len(results)}** `BloqExample`s through '
                 'compile → reload → execute.\n')
    lines.append('## Summary\n')
    lines.append('| Outcome | Count |')
    lines.append('| --- | ---: |')
    for outcome in BuildOutcome:
        lines.append(f'| `{outcome.value}` | {len(by_outcome[outcome])} |')
    lines.append('')

    for outcome in BuildOutcome:
        if not by_outcome[outcome]:
            continue
        lines.append(f'## {outcome.value} ({len(by_outcome[outcome])})\n')
        lines.append('| Example | Detail |')
        lines.append('| --- | --- |')
        for r in by_outcome[outcome]:
            detail = r.detail.replace('|', '\\|')
            lines.append(f'| `{r.name}` | {detail} |')
        lines.append('')

    path.write_text('\n'.join(lines))
    print(f'\nWrote markdown report to {path}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output_dir', type=Path,
                        help='Root directory for the generated .qlt files (created if needed). '
                             'Files are compiled into <output_dir>/partial/ and moved to '
                             '<output_dir>/lib/ on success, organized by fully-qualified bloq '
                             'class, e.g. lib/qualtran/bloqs/arithmetic/Add/add_small.qlt.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only process the first N examples (for a quick smoke test).')
    parser.add_argument('--regenerate', action='store_true',
                        help='Recompile and overwrite .qlt files that already exist. By default, '
                             'existing files are reused (the compile step is skipped) and only '
                             'reloaded and executed.')
    parser.add_argument('--timeout', type=float, default=60.0,
                        help='Per-example time budget in seconds; 0 disables it. Default: 60.')
    parser.add_argument('--include-symbolic', action='store_true',
                        help='Include symbolic bloq examples (by default, examples with '
                             '"symb" or "symbolic" in their name are skipped).')
    parser.add_argument('--report', type=Path, default=None,
                        help='Also write a markdown report to this path.')
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Writing .qlt files to {out_dir}')
    results = build_all(
        out_dir,
        limit=args.limit,
        timeout=args.timeout,
        regenerate=args.regenerate,
        skip_symbolic=not args.include_symbolic,
    )

    print_summary(results)
    if args.report is not None:
        write_report(results, args.report)

    # Exit nonzero if anything failed to build cleanly (execution problems are
    # informational and do not affect the exit code).
    n_hard_failures = sum(1 for r in results if r.outcome.is_hard_failure)
    sys.exit(1 if n_hard_failures else 0)


if __name__ == '__main__':
    main()
