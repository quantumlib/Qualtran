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
"""Build an on-disk library of verified Qualtran-L1 (`.qlt`) files from bloqs.

Given a `(bloq, name)` pair and a root directory, `build_library_entry` runs the
full L1 pipeline for that bloq:

 1. Compiles it to a `.qlt` file (`qualtran.l1.dump_l1`).
 2. Loads that `.qlt` file back into bloqs (`qualtran.l1.load_module`).
 3. Executes the root bloq through the
    `StandardQualtranArchitectureAgnosticVirtualMachine`.

The `.qlt` file is compiled into `<root>/partial/<rel>` and, on a fully
successful execution, moved to `<root>/lib/<rel>`. Hard failures remove the file;
soft failures leave it under `partial/`. Here `<rel>` mirrors the bloq's
fully-qualified class name (`Bloq._class_name_in_pkg_()`) with `name` as the
filename stem, e.g. `qualtran/bloqs/arithmetic/Add/add_small.qlt`. So `lib/` ends
up holding only the bloqs that round-trip and execute cleanly.

The outcome of building one entry is a `BuildOutcome`:

 - `SUCCESS`: compiled, reloaded, and executed with no reported problems.
 - `SKIPPED`: skipped by caller filtering or configuration.
 - `COMPILE_FAILED`: `dump_l1` raised.
 - `RELOAD_FAILED`: `load_module` raised, the root key was lost, or the root
   reloaded as a `_PlaceholderBloq` (an extern that could not be re-linked).
 - `EXECUTION_CATASTROPHIC`: the VM raised an uncaught exception.
 - `EXECUTION_WITH_PROBLEMS`: the VM ran to completion but recorded one or more
   `Problem`s (e.g. unsupported atomic bloqs).
 - `TIMEOUT`: the entry exceeded its per-entry time budget (`timeout=`).
 - `CONSTRUCT_FAILED`: reserved for callers that construct bloqs before handing
   them here (e.g. from a `BloqExample`); `build_library_entry` never returns it.

This module works purely with `Bloq` objects and a caller-provided `name`. The
`(bloq, name)` corpus itself -- e.g. discovering `BloqExample`s -- is the concern
of driver scripts, not this library.
"""

from __future__ import annotations

import contextlib
import enum
import io
import signal
import traceback
from pathlib import Path
from typing import Dict, Iterator, Optional, Type, TYPE_CHECKING, Union

import attrs

from ._parse_eval import load_module
from ._to_l1 import dump_l1
from ._vm import StandardQualtranArchitectureAgnosticVirtualMachine

if TYPE_CHECKING:
    import qualtran as qlt


class BuildOutcome(enum.Enum):
    """The outcome of building one library entry, ordered from best to worst.

    The declaration order is significant: iterating `BuildOutcome` yields members
    from best (`SUCCESS`) to worst (`CONSTRUCT_FAILED`), which is a convenient
    order for summaries and reports. Each member's value is the short,
    human-readable label used in output.
    """

    SUCCESS = 'success'
    EXECUTION_WITH_PROBLEMS = 'execution_with_problems'
    SKIPPED = 'skipped'
    EXECUTION_CATASTROPHIC = 'execution_catastrophic'
    TIMEOUT = 'timeout'
    RELOAD_FAILED = 'reload_failed'
    COMPILE_FAILED = 'compile_failed'
    CONSTRUCT_FAILED = 'construct_failed'

    @property
    def is_hard_failure(self) -> bool:
        """Whether this outcome is a hard failure (e.g. should fail a build).

        Soft outcomes (`SUCCESS`, `EXECUTION_WITH_PROBLEMS`, `SKIPPED`) are not hard
        failures; everything else is.
        """
        return self in _HARD_FAILURES


# Outcomes that represent a hard failure.
_HARD_FAILURES = frozenset(
    {
        BuildOutcome.CONSTRUCT_FAILED,
        BuildOutcome.COMPILE_FAILED,
        BuildOutcome.RELOAD_FAILED,
        BuildOutcome.EXECUTION_CATASTROPHIC,
        BuildOutcome.TIMEOUT,
    }
)


@attrs.frozen
class L1BuildResult:
    """The result of building a single library entry.

    Attributes:
        name: The caller-provided name for the bloq.
        outcome: The `BuildOutcome`.
        detail: A short human-readable description (error summary or stats).
        traceback: The full traceback, if an exception was caught.
        qlt_path: Path to the final `.qlt` file (under `lib/` for a success,
            under `partial/` for a soft failure), or `None` if no file remains.
        n_atoms: Number of ISA operations executed (if execution ran).
        n_calls: Number of subroutine calls (if execution ran).
        n_problems: Number of problems the VM reported (if execution ran).
    """

    name: str
    outcome: BuildOutcome
    detail: str = ''
    traceback: Optional[str] = None
    qlt_path: Optional[str] = None
    n_atoms: Optional[int] = None
    n_calls: Optional[int] = None
    n_problems: Optional[int] = None


class _BuildTimeout(BaseException):
    """Raised (via SIGALRM) when building one entry exceeds its time budget.

    Subclasses `BaseException` (not `Exception`) so that the broad per-stage
    `except Exception` handlers do not swallow it; it propagates to the dedicated
    timeout handler instead.
    """


@contextlib.contextmanager
def _time_limit(seconds: Optional[float]) -> Iterator[None]:
    """Context manager that raises `_BuildTimeout` after `seconds` (Unix only).

    Uses `SIGALRM`, so this interrupts pure-Python work (deep recursion, large
    decomposition walks) but cannot preempt long-running C extension calls that
    do not return to the interpreter.

    Args:
        seconds: The time budget in seconds. If `None` or non-positive, no limit
            is imposed.
    """
    if not seconds or seconds <= 0:
        yield
        return

    def _handler(signum, frame):  # noqa: ANN001
        raise _BuildTimeout()

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _short_exc(e: BaseException) -> str:
    """A one-line summary of an exception."""
    return f'{type(e).__name__}: {e}'.replace('\n', ' ')[:300]


def library_qlt_path(
    bloq_or_cls: Union['qlt.Bloq', Type['qlt.Bloq']], name: str, root: Path, *, subdir: str = 'lib'
) -> Path:
    """Compute the fully-qualified `.qlt` path for a named bloq in a library.

    The bloq's fully-qualified class name (`Bloq._class_name_in_pkg_()`, e.g.
    `qualtran.bloqs.arithmetic.Add`) becomes a subdirectory path and `name`
    becomes the filename stem, e.g.
    `<root>/<subdir>/qualtran/bloqs/arithmetic/Add/add_small.qlt`.

    Args:
        bloq_or_cls: A `Bloq` instance or subclass. `_class_name_in_pkg_` is a
            classmethod, so either works.
        name: The caller-provided name (filename stem). Should be unique among
            entries that share a bloq class.
        root: The library root directory.
        subdir: The tree under `root` to place the file in (`lib` or `partial`).

    Returns:
        The path to the `.qlt` file.
    """
    class_in_pkg = bloq_or_cls._class_name_in_pkg_()
    return root.joinpath(subdir, *class_in_pkg.split('.'), f'{name}.qlt')


def _finalize_placement(
    result: L1BuildResult, partial_path: Path, lib_path: Path
) -> L1BuildResult:
    """Move/remove the working `.qlt` file based on the outcome.

    Establishes the invariant:

     - `SUCCESS`: the file lives under `lib/`.
     - soft failure (`EXECUTION_WITH_PROBLEMS`): the file stays under `partial/`.
     - hard failure (see `BuildOutcome.is_hard_failure`): the file is removed
       from both `lib/` and `partial/` (if anything was written at all).

    Args:
        result: The outcome of building the entry. Files are compiled into
            `partial_path`, but a reused file may already live at `lib_path`.
        partial_path: The `partial/` location for this entry.
        lib_path: The `lib/` location for this entry.

    Returns:
        A copy of `result` with `qlt_path` pointing at the final location (or
        `None` if the file was removed / never written).
    """
    if result.outcome is BuildOutcome.SUCCESS:
        lib_path.parent.mkdir(parents=True, exist_ok=True)
        if partial_path.exists():
            partial_path.replace(lib_path)
        final: Optional[Path] = lib_path
    elif result.outcome.is_hard_failure:
        partial_path.unlink(missing_ok=True)
        lib_path.unlink(missing_ok=True)
        final = None
    else:  # Soft failure: keep it under partial/.
        if lib_path.exists() and not partial_path.exists():
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            lib_path.replace(partial_path)
        final = partial_path if partial_path.exists() else None

    return attrs.evolve(result, qlt_path=(str(final) if final is not None else None))


def build_library_entry(
    bloq: 'qlt.Bloq',
    name: str,
    root: Path,
    *,
    timeout: Optional[float] = None,
    regenerate: bool = False,
    extern_only_from: bool = False,
) -> L1BuildResult:
    """Compile, reload, and execute one bloq, filing it into an on-disk library.

    The `.qlt` file is compiled into `<root>/partial/<rel>`. On success it is
    moved to `<root>/lib/<rel>`; on a hard failure it is removed; on a soft
    failure it is left under `partial/` (see `_finalize_placement`). Here `<rel>`
    is the fully-qualified path from `library_qlt_path`.

    Args:
        bloq: The (already-constructed) bloq to build. The library never
            constructs bloqs itself.
        name: A name for this bloq, unique among entries sharing its class. Used
            as the `.qlt` filename stem and as the root bloq's key in the file.
        root: The library root directory (containing `partial/` and `lib/`).
        timeout: Per-entry time budget in seconds (via `SIGALRM`, Unix-only).
            `None` or `0` disables it. On expiry the partial file is cleaned up
            and the outcome is `TIMEOUT`.
        regenerate: If `True`, always recompile (overwriting `partial/`). If
            `False` (default) and a `lib/` file already exists, the compile step
            is skipped and that file is reused for loading/execution.
        extern_only_from: Passed through to `dump_l1`.

    Returns:
        An `L1BuildResult` describing the outcome. `build_library_entry` never
        returns `BuildOutcome.CONSTRUCT_FAILED`.
    """
    from ._eval import _PlaceholderBloq

    partial_path = library_qlt_path(bloq, name, root, subdir='partial')
    lib_path = library_qlt_path(bloq, name, root, subdir='lib')
    # Mutable holder so the timeout handler can report which stage was running.
    stage = 'setup'

    def _inner() -> L1BuildResult:
        nonlocal stage

        # 1. Compile into partial/, unless a successfully-built lib/ file already
        #    exists and can be reused.
        if lib_path.exists() and not regenerate:
            stage = 'reuse'
            source_path = lib_path
        else:
            source_path = partial_path
            stage = 'compile'
            try:
                buf = io.StringIO()
                # Assign the root a deterministic, known key (`name`) so it can be
                # looked up unambiguously after reloading.
                dump_l1(bloq, buf, root_bloq_key=name, extern_only_from=extern_only_from)
                partial_path.parent.mkdir(parents=True, exist_ok=True)
                partial_path.write_text(buf.getvalue())
            except Exception as e:  # pylint: disable=broad-except
                return L1BuildResult(
                    name, BuildOutcome.COMPILE_FAILED, _short_exc(e), traceback.format_exc()
                )

        # 2. Reload from the .qlt file.
        stage = 'reload'
        try:
            l1_text = source_path.read_text()
            loaded: Dict[str, object] = load_module(l1_text)  # type: ignore[assignment]
        except Exception as e:  # pylint: disable=broad-except
            return L1BuildResult(
                name, BuildOutcome.RELOAD_FAILED, _short_exc(e), traceback.format_exc(),
                qlt_path=str(source_path)
            )

        if not loaded:
            return L1BuildResult(
                name, BuildOutcome.RELOAD_FAILED, 'load_module returned no bloqs',
                qlt_path=str(source_path)
            )

        # Select the root bloq by its known key. Fall back to the first
        # (definition-order) entry -- which is the root -- only for older files
        # that were generated without an assigned key.
        if name in loaded:
            root_bloq = loaded[name]
        else:
            root_bloq = loaded[next(iter(loaded))]

        if isinstance(root_bloq, _PlaceholderBloq):
            return L1BuildResult(
                name,
                BuildOutcome.RELOAD_FAILED,
                f'root {name!r} reloaded as a placeholder (extern failed to re-link)',
                qlt_path=str(source_path),
            )

        # 3. Execute through the VM.
        stage = 'execute'
        vm = StandardQualtranArchitectureAgnosticVirtualMachine()
        try:
            vm.execute(root_bloq)  # type: ignore[arg-type]
        except Exception as e:  # pylint: disable=broad-except
            return L1BuildResult(
                name,
                BuildOutcome.EXECUTION_CATASTROPHIC,
                _short_exc(e),
                traceback.format_exc(),
                qlt_path=str(source_path),
                n_atoms=vm.n_atoms,
                n_calls=vm.n_calls,
            )

        if vm.problems:
            summaries = sorted({p.get_summary().strip() for p in vm.problems})
            detail = f'{len(vm.problems)} problem(s): ' + '; '.join(summaries[:3])
            return L1BuildResult(
                name,
                BuildOutcome.EXECUTION_WITH_PROBLEMS,
                detail,
                qlt_path=str(source_path),
                n_atoms=vm.n_atoms,
                n_calls=vm.n_calls,
                n_problems=len(vm.problems),
            )

        return L1BuildResult(
            name,
            BuildOutcome.SUCCESS,
            f'{vm.n_atoms} ISA ops through {vm.n_calls} calls',
            qlt_path=str(source_path),
            n_atoms=vm.n_atoms,
            n_calls=vm.n_calls,
            n_problems=0,
        )

    try:
        with _time_limit(timeout):
            result = _inner()
    except _BuildTimeout:
        result = L1BuildResult(name, BuildOutcome.TIMEOUT, f'exceeded {timeout}s during {stage}')

    return _finalize_placement(result, partial_path, lib_path)
