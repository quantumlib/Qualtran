#  Copyright 2024 Google LLC
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
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, MutableMapping, Optional, Protocol, Set, Tuple

from mdit_py_plugins.anchors.index import slugify

from ._page import Page

_CACHE: MutableMapping[str, Dict[str, Optional[str]]] = defaultdict(dict)


class Writable(Protocol):
    def write(self, s: str): ...


class LinkingWriter:
    """Support `.write`, but insert markdown links based on a regex.

    Ordinarily, if you use `f.write(s)`, it will look through the string `s` and insert markdown
    links. To avoid this, use `f.write_nl(s)`.

    This uses the "references" style of markdown link where the inline syntax is [text][href]
    and the mapping of href to URLs is added at the bottom of the document. Users of this
    writer must call `write_link_targets` at the end of their document to include this
    crucial part.
    """

    def __init__(
        self,
        f: Writable,
        *,
        link_aliases: Dict[str, str],
        link_d: Dict[str, Tuple[Page, Optional[str]]],
        refdoc_relpath: Path,
    ):
        self._f: Writable = f
        self._linked: Set[str] = set()
        self._link_aliases = link_aliases
        self._link_d = link_d
        self._reference_relpath = refdoc_relpath  # TODO: continue refactor -> refdoc_relpath

    def resolve_dotpath(self, dotpath: str) -> Optional[str]:
        maybe_cached = _CACHE[str(self._reference_relpath)].get(dotpath, None)
        if maybe_cached is not None:
            return maybe_cached

        try:
            page, subname = self._link_d[self._link_aliases[dotpath]]
        except KeyError:
            return None

        slash_path = page.out_path(out_dir=self._reference_relpath)
        if subname is None:
            pp = f'/{slash_path}'
            _CACHE[str(self._reference_relpath)][dotpath] = pp
            return pp
        else:
            # https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#anchor-slug-structure
            slug = slugify(subname)
            pp = f'/{slash_path}#{slug}'
            _CACHE[str(self._reference_relpath)][dotpath] = pp
            return pp

    def repl(self, ma: re.Match) -> str:
        # Used in `re.sub`
        l = ma.group(1)
        if self.resolve_dotpath(l) is None:
            return f'`{l}`'
        else:
            self._linked.add(l)
            return f'[`{l}`]'

    def linkify(self, s: str) -> str:
        return re.sub(r'`([\w\.]+)`', self.repl, s)

    def write(self, s: str):
        self._f.write(self.linkify(s))

    def write_nl(self, s: str):
        self._f.write(s)

    def write_link_targets(self):
        self._f.write('\n')
        for dotpath in self._linked:
            self._f.write(f'[`{dotpath}`]: {self.resolve_dotpath(dotpath)}\n')
