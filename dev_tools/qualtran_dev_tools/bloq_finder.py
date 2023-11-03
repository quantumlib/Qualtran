#  Copyright 2023 Google LLC
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
import importlib
import inspect
import subprocess
from pathlib import Path
from typing import Callable, Iterable, List, Tuple, Type

from qualtran import Bloq, BloqExample

from .git_tools import get_git_root


def _get_paths(bloqs_root: Path, filter_func: Callable[[Path], bool]) -> List[Path]:
    """Get *.py files based on `filter_func`."""
    cp = subprocess.run(
        ['git', 'ls-files', '*.py'], capture_output=True, universal_newlines=True, cwd=bloqs_root
    )
    outs = cp.stdout.splitlines()
    paths = [Path(out) for out in outs]

    paths = [path for path in paths if filter_func(path)]
    return paths


def get_bloq_module_paths(bloqs_root: Path) -> List[Path]:
    """Get *.py files for non-test, non-init modules under `bloqs_root`."""

    def is_module_path(path: Path) -> bool:
        if path.name.endswith('_test.py'):
            return False

        if path.name == '__init__.py':
            return False

        return True

    return _get_paths(bloqs_root, is_module_path)


def get_bloq_test_module_paths(bloqs_root: Path) -> List[Path]:
    """Get *_test.py files under `bloqs_root`."""

    def is_test_module_path(path: Path) -> bool:
        if not path.name.endswith('_test.py'):
            return False

        return True

    return _get_paths(bloqs_root, is_test_module_path)


def _bloq_modpath_to_modname(path: Path) -> str:
    """Get the canonical, full module name given a module path."""
    return 'qualtran.bloqs.' + str(path)[: -len('.py')].replace('/', '.')


def modpath_to_bloqs(path: Path) -> Iterable[Type[Bloq]]:
    """Given a module path, return all the `Bloq` classes defined within."""
    modname = _bloq_modpath_to_modname(path)
    mod = importlib.import_module(modname)
    for name, cls in inspect.getmembers(mod, inspect.isclass):
        if cls.__module__ != modname:
            # Perhaps from an import
            continue

        if not issubclass(cls, Bloq):
            continue

        if cls.__name__.startswith('_'):
            continue

        yield cls


def modpath_to_bloq_exs(path: Path) -> Iterable[Tuple[str, str, BloqExample]]:
    """Given a module path, return all the `BloqExample`s defined within."""
    modname = _bloq_modpath_to_modname(path)
    mod = importlib.import_module(modname)

    for name, obj in inspect.getmembers(mod, lambda x: isinstance(x, BloqExample)):
        yield modname, name, obj


def get_bloq_classes() -> List[Type[Bloq]]:
    reporoot = get_git_root()
    bloqs_root = reporoot / 'qualtran/bloqs'
    paths = get_bloq_module_paths(bloqs_root)
    bloq_clss: List[Type[Bloq]] = []
    for path in paths:
        bloq_clss.extend(modpath_to_bloqs(path))
    return bloq_clss


def get_bloq_examples() -> List[BloqExample]:
    reporoot = get_git_root()
    bloqs_root = reporoot / 'qualtran/bloqs'
    paths = get_bloq_module_paths(bloqs_root)

    bexamples: List[BloqExample] = []
    for path in paths:
        for modname, name, be in modpath_to_bloq_exs(path):
            bexamples.append(be)

    return bexamples
