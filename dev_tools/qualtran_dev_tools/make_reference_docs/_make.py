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
import ast
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, cast, Dict, List, Optional, Sequence, Set, Tuple, Union

import attrs
import griffe
from griffe import GriffeLoader, Kind

from ._render_context import RenderContext

# Begin monkeypatch:
# the `Visitor` will use certain decorators to apply "labels" to the AST nodes, which
# causes properties to be parsed as Attributes instead of Functions. By removing these
# decorator-to-label mappings, they are kept as Function. We handle decorators for functions
# in this script how we want.
import griffe._internal.agents.visitor
del griffe._internal.agents.visitor.builtin_decorators['property']
del griffe._internal.agents.visitor.stdlib_decorators['functools.cached_property']

from ._page import MajorClassPage, MemberType, ModulePage, ModulePageMember, Page
from ._pages.render_major_class import render_major_class
from ._pages.render_module import render_module
from ._pages.render_toc import render_toc

MAJOR_CLASSES = [
    'qualtran.Bloq',
    'qualtran.CompositeBloq',
    'qualtran.BloqBuilder',
    'qualtran.Signature',
    'qualtran.GateWithRegisters',
    'qualtran.dtype.QCDType',
    'qualtran.resource_counting.QECGatesCost',
    'qualtran.resource_counting.GateCounts',
    'qualtran.resource_counting.QubitCount',
    'qualtran.resource_counting.CostKey',
    'qualtran.resource_counting.SuccessProb',
    'qualtran.resource_counting.BloqCount',
    'qualtran.simulation.classical_sim.ClassicalSimState',
    'qualtran.simulation.classical_sim.PhasedClassicalSimState',
]

SKIP_MODULES = [
    'qualtran.conftest',
    'qualtran.testing_test',
    'qualtran.protos',
    'qualtran.serialization.resolver_dict',
    'qualtran.bloqs',
]

MINIMAL_SKIP_CANONPATHS = [
    'qualtran.surface_code',
    'qualtran.serialization',
    'qualtran.cirq_interop',
    'qualtran.qref_interop',
    'qualtran.linalg',
    'qualtran._infra.gate_with_registers.GateWithRegisters',
    'qualtran.resource_counting.t_counts_from_sigma',
]
MINIMAL = True

DEFINED_IN_CONTAINER_EXCEPTIONS = ['qualtran.dtype', 'qualtran.exception', 'qualtran.cirq_interop']


def _get_all_aliases(obj: Union[griffe.Object, griffe.Alias]) -> Set[str]:
    """Get all the valid aliases for `obj`."""

    # First, try to use the aliases griffe has found
    if len(obj.aliases) == 0:
        all_aliases = set()
    else:
        all_aliases = set(obj.aliases.keys())

    # make sure to include the current path and the canonical path, which don't always
    # get included in griffe's list of aliases.
    all_aliases.add(obj.path)
    all_aliases.add(obj.canonical_path)

    defined_container = '.'.join(obj.canonical_path.split('.')[:-1])

    def is_valid_alias(alias: str) -> bool:
        if '._' in alias or alias.startswith('_'):
            # private
            return False

        # The alias must be in the same lineage as the place where the object is
        # actually defined
        alias_container = '.'.join(alias.split('.')[:-1])
        if not defined_container.startswith(alias_container):
            # With some exceptions
            if alias_container in DEFINED_IN_CONTAINER_EXCEPTIONS:
                return True
            return False

        # Doens't violate any rules
        return True

    valid_aliases = set(a for a in all_aliases if is_valid_alias(a))
    if len(valid_aliases) == 0:
        warnings.warn(f"No valid aliases for {obj} found in {all_aliases}")
    return valid_aliases


def _get_preferred_dotpath(all_aliases: Set[str]):
    """From a list of candidates, select the best "dotpath".

    A "dotpath" is a string path of the form "foo.bar.X", i.e. with dots. It is in contrast
    to a file path with slashes.
    """

    def prefered_path_key(alias: str):
        return -len(alias.split('.'))

    pref_path = sorted(all_aliases, key=prefered_path_key)[0]
    return pref_path


@attrs.mutable
class _PackageWalker:
    """A helper class that uses Griffe to walk through a package and organize the objects it finds.

    The public entry point is via the function `get_pages`.
    """

    seen: Set[str] = attrs.field(factory=set, kw_only=True)
    """A set of seen canonical dotpaths. Griffe uses canonical dotpaths to uniquely
    identify things, so we do too for `seen`. Afterwards, the doc system will use only our
    preferred path."""

    pages_d: Dict[str, Page] = attrs.field(factory=lambda: defaultdict(ModulePage), kw_only=True)
    """Mapping of preferred dotpath to `Page`."""

    aliases_d: Dict[str, str] = attrs.field(factory=dict, kw_only=True)
    """Mapping from each alias to the preferred dotpath (many-to-one)."""

    link_d: Dict[str, Tuple[Page, Optional[str]]] = attrs.field(factory=dict, kw_only=True)
    """Mapping from preferred dotpath to a doc location."""

    def _walk_table_of_contents(self, obj: Union[griffe.Alias, griffe.Object]):
        """DFS through all the objects.

        First, we recurse on the objects we care about.

        Then, we determine the "preferred dotpath" for each object. Todo .. doc

        Finally, we sort the object into either a MajorClassPage or ModulePage.
        """
        if obj.canonical_path in SKIP_MODULES:
            return

        if MINIMAL and obj.canonical_path in MINIMAL_SKIP_CANONPATHS:
            return

        if obj.is_module:
            for name, obj2 in obj.members.items():
                if obj2.kind is Kind.ALIAS:
                    # print(obj2.path)
                    continue

                if obj2.canonical_path in self.seen:
                    # Already found it
                    continue

                if obj2.is_private:
                    continue

                if obj2.is_special:
                    continue

                if obj2.is_module and obj2.name.endswith('_test'):
                    # print(obj2.path)
                    continue

                if obj2.is_module and obj2.name.endswith('_pb2'):
                    # print(obj2.path)
                    continue

                self._walk_table_of_contents(obj2)

        aliases = _get_all_aliases(obj)

        if len(aliases) == 0:
            # Warning issued in _get_all_aliases
            return

        pref_path = _get_preferred_dotpath(aliases)
        pref_parent = '.'.join(pref_path.split('.')[:-1])
        for alias in aliases:
            assert alias not in self.aliases_d
            self.aliases_d[alias] = pref_path

        # First, make sure our parent knows about us
        if obj.is_module:
            membtyp = MemberType.MODULE
        elif pref_path in MAJOR_CLASSES:
            membtyp = MemberType.MAJOR
        else:
            membtyp = MemberType.MINOR

        parent_page = self.pages_d[pref_parent]
        assert isinstance(parent_page, ModulePage), f"Parent isn't a module: {parent_page}"
        parent_page.members.append(ModulePageMember(cast(griffe.Object, obj), pref_path, membtyp))

        if obj.is_module:
            # Module
            obj = cast(griffe.Module, obj)
            self.pages_d[pref_path].obj = obj
            self.pages_d[pref_path].section = '.'.join(pref_path.split('.')[:2])
            self.pages_d[pref_path].pref_path = pref_path
            self.link_d[pref_path] = (self.pages_d[pref_path], None)
            self.seen.add(obj.canonical_path)

        elif pref_path in MAJOR_CLASSES:
            # Major class
            obj = cast(griffe.Class, obj)
            page = MajorClassPage(
                obj=obj, section='.'.join(pref_parent.split('.')[:2]), pref_path=pref_path
            )
            assert pref_path not in self.pages_d
            self.pages_d[pref_path] = page
            self.link_d[pref_path] = (page, None)
            self.seen.add(obj.canonical_path)

        else:
            # Minor member
            self.seen.add(obj.canonical_path)
            self.link_d[pref_path] = (parent_page, obj.name)


def get_pages(
    root_mod: griffe.Module,
) -> Tuple[List[Page], Dict[str, str], Dict[str, Tuple[Page, Optional[str]]]]:
    """Walk down from `root_mod`."""
    assert root_mod.is_module
    assert root_mod.is_init_module
    assert root_mod.parent is None

    pw = _PackageWalker()
    pw._walk_table_of_contents(root_mod)

    pages_d = pw.pages_d
    prefpaths = list(pages_d.keys())
    for prefpath in prefpaths:
        page = pages_d[prefpath]
        if page.obj is None:
            warnings.warn(f"Orphan {prefpath!r}")
            del pages_d[prefpath]

    return list(pages_d.values()), pw.aliases_d, pw.link_d


def walk_and_configure(
    rootmod_dotpath: str,
    refdoc_relpath: Path,
    top_sections: Sequence[str],
    fake_sections: Sequence[str],
    addtl_linkable: Dict[str, str],
) -> Tuple[List[Page], RenderContext]:
    """Organize things or whatever."""

    # Incantation to get `griffe` set up.
    extensions = griffe.load_extensions(AttrsExtension())
    loader = GriffeLoader(extensions=extensions)
    root_mod = cast(griffe.Module, loader.load(rootmod_dotpath))
    unresolved, _ = loader.resolve_aliases()
    assert len(unresolved) == 0

    # Walk the tree into our weird internal data structurs.
    pages, aliases_d, link_d = get_pages(root_mod=root_mod)

    # Merge user-provided and auto-found section titles.
    sections = list(top_sections) + sorted(
        set(mp.section_not_none for mp in pages) - set(top_sections) - set(fake_sections)
    )

    # Set up dictionary of "linkable" items.
    linkable_to_prefpath = dict(aliases_d)
    linkable_to_prefpath |= addtl_linkable

    # Return pages and a `RenderContext`.
    render_context = RenderContext(
        refdoc_relpath=refdoc_relpath,
        sections=sections,
        aliases_d=aliases_d,
        link_d=link_d,
        linkable_to_prefpath=linkable_to_prefpath,
    )
    return pages, render_context


class AttrsExtension(griffe.Extension):
    """Hack to try to get attrs annotations to show up in parameter docs."""

    def _is_attrs_class(self, cls: griffe.Class) -> bool:
        if 'attrs.frozen' in [d.callable_path for d in cls.decorators]:
            return True
        return False

    def on_class_members(
        self,
        *,
        node: Union[ast.AST, griffe.ObjectNode],
        cls: griffe.Class,
        agent: Union[griffe.Visitor, griffe.Inspector],
        **kwargs: Any,
    ) -> None:
        if self._is_attrs_class(cls):
            cls.members['__init__'] = griffe.Function(
                name='__init__',
                parameters=griffe.Parameters(
                    *[
                        griffe.Parameter(name=name.lstrip('_'), annotation=pp.annotation)
                        for name, pp in cls.attributes.items()
                    ]
                ),
            )


def make_reference_docs(
    repo_root_path: Path,
    refdoc_relpath: Path = Path("reference"),
    rootmod_dotpath: str = "qualtran",
):
    """Make the reference documentation.

    The steps are

        1. Walk the package structure with `get_pages`.
        2. Write each page; and write the table of contentes (TOC).

    There are multiple 'types' of 'path's we must consider when rendering the documentation.

    Filesystem paths are a series of slash (/) delimited segments and point to a file on disk. They
    can be *relative* to another directory.

    Python import paths are a seriies of dot (.) delimited segments and point to a Python object.
    The same object can be referred to by multiple "dotpaths". The one corresponding to the
    containing file path is the "canonical dotpath". The one that we want users to use
    during import statements (i.e. re-exported from a private module to a public package
    via `__init__.py` files) are "preferred dotpath"s.
    """

    top_sections = [
        'qualtran',
        'qualtran.resource_counting',
        'qualtran.simulation',
        'qualtran.drawing',
        'qualtran.symbolics',
        'interop',
    ]
    fake_sections = [
        # Grouped in `qualtran`.
        'qualtran.dtype',
        'qualtran.exception',
        # Grouped in "interop"
        'qualtran.qref_interop',
        'qualtran.cirq_interop',
    ]
    addtl_linkable = {
        'Bloq': 'qualtran.Bloq',
        'CompositeBloq': 'qualtran.CompositeBloq',
        'Soquet': 'qualtran.Soquet',
    }

    def _pages_sort_key(p: Page):
        assert p.pref_path is not None, f'Uninitialized {p}'
        path_parts = p.pref_path.split('.')
        return path_parts

    def _page_in_section(p: Page, section: str):
        if p.section == section:
            return True
        if section == 'qualtran' and p.section in ['qualtran.dtype', 'qualtran.exception']:
            # Group
            return True
        if section == 'interop' and p.section in ['qualtran.qref_interop', 'qualtran.cirq_interop']:
            return True
        return False

    pages, render_context = walk_and_configure(
        rootmod_dotpath=rootmod_dotpath,
        refdoc_relpath=refdoc_relpath,
        top_sections=top_sections,
        fake_sections=fake_sections,
        addtl_linkable=addtl_linkable,
    )

    refdoc_path = repo_root_path / 'docs' / refdoc_relpath

    render_toc(
        out_dir=refdoc_path,
        pages=pages,
        page_in_section=_page_in_section,
        pages_sort_key=_pages_sort_key,
        render_context=render_context,
    )
    for page in pages:
        if isinstance(page, MajorClassPage):
            render_major_class(base_dir=refdoc_path, page=page, render_context=render_context)
        elif isinstance(page, ModulePage):
            render_module(out_dir=refdoc_path, page=page, render_context=render_context)
        else:
            raise ValueError(page)
