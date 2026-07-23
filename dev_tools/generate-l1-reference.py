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
"""Generate `.qlt` reference files for the Qualtran-L1 example bloqs.

For each example in `qualtran.l1.L1_EXAMPLES` this:

 1. Constructs the bloq and validates it (`validate_bloq`).
 2. Compiles it to `.qlt` text and validates that it round-trips
    (`check_artifacts`); the file is only written if the round trip is clean.
 3. Writes (or, with `--check`, verifies) the committed reference file under
    `qualtran/l1/examples/`.

Usage:

```
python dev_tools/generate-l1-reference.py           # (re)write
python dev_tools/generate-l1-reference.py --check   # verify only
```
"""

import argparse
import sys
from typing import List

from qualtran_dev_tools.write_if_different import WriteIfDifferent

from qualtran.l1 import (
    check_artifacts,
    compile_bloq_to_l1,
    get_l1_examples,
    L1Example,
    validate_bloq,
)


def render_reference(example: L1Example) -> str:
    """Construct, validate, and round-trip-check an example's `.qlt` code.

    Args:
        example: The example to render.

    Returns:
        The validated `.qlt` text.

    Raises:
        AssertionError: If the bloq fails validation or its round trip.
    """
    bloq = example.make()
    validate_bloq(bloq)

    artifacts = compile_bloq_to_l1(bloq)
    problems = check_artifacts(artifacts)
    if problems:
        joined = '\n  - '.join(problems)
        raise AssertionError(f"Example {example.name!r} does not round-trip cleanly:\n  - {joined}")
    return artifacts.l1_code


def generate(check: bool = False, include_slow: bool = True) -> List[str]:
    """Generate or verify all reference files.

    Args:
        check: If `True`, do not write; instead return a list of files that are
            missing or out of date.
        include_slow: Whether to include examples marked `slow`.

    Returns:
        A list of human-readable descriptions of stale/missing files (always empty
        unless `check=True`).
    """
    stale: List[str] = []
    for example in get_l1_examples(include_slow=include_slow):
        code = render_reference(example)
        path = example.reference_path()

        if check:
            existing = path.read_text() if path.is_file() else None
            if existing != code:
                reason = 'missing' if existing is None else 'out of date'
                stale.append(f'{path} ({reason})')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with WriteIfDifferent(path) as f:
                f.write(code)

    return stale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--check',
        action='store_true',
        help='Verify committed reference files are up to date instead of writing them.',
    )
    parser.add_argument('--no-slow', action='store_true', help='Skip examples marked as slow.')
    args = parser.parse_args()

    stale = generate(check=args.check, include_slow=not args.no_slow)
    if args.check and stale:
        print('The following Qualtran-L1 reference files are stale:', file=sys.stderr)
        for s in stale:
            print(f'  - {s}', file=sys.stderr)
        print('Regenerate them with: python dev_tools/generate-l1-reference.py', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
