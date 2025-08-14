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


from pathlib import Path
from typing import cast, Dict, Sequence

import griffe
from griffe import Kind

from ...write_if_different import WriteIfDifferent
from .._components.aliases import get_aliases_str
from .._components.render_docstring import split_docstring, write_docstring_parts
from .._components.render_signature import write_function_signature
from .._linking_writer import LinkingWriter
from .._page import MemberType, ModulePage, ModulePageMember
from .._render_context import RenderContext


def write_module_other_member(f: LinkingWriter, obj, pref_path, aliases_d):
    first_line, rest = split_docstring(obj)
    f.write_nl(f'### `{obj.name}`\n')
    f.write_nl(get_aliases_str(pref_path, aliases_d=aliases_d))
    f.write_nl('\n\n')
    f.write(first_line)
    f.write('\n\n')
    if obj.kind is Kind.FUNCTION:
        write_function_signature(f, cast(griffe.Function, obj))
    write_docstring_parts(f, rest, level=1)

    submemb: griffe.Object
    doc_submembers = {
        (submemb_name, submemb)
        for submemb_name, submemb in obj.members.items()
        if (not submemb.is_special) and submemb.has_docstring
    }
    if doc_submembers:
        f.write('#### Members\n')
        for submemb_name, submemb in doc_submembers:
            subdesc, _ = split_docstring(submemb)
            f.write_nl(f'`{submemb_name}`\n')
            f.write(f': {subdesc}\n\n')

    if obj.inherited_members:
        f.write('**All Members:** ')
        f.write(
            ', '.join(
                f'`{submemb_name}`'
                for submemb_name, submemb in obj.all_members.items()
                if (not submemb.is_special) and submemb.is_public
            )
        )
        f.write('\n')


def write_module(
    f: LinkingWriter,
    obj: griffe.Module,
    pref_path: str,
    members: Sequence[ModulePageMember],
    aliases_d: Dict[str, str],
):
    # Title
    title_dotpath = '.'.join(pref_path.split('.')[-2:])
    f.write_nl(f'# `{title_dotpath}`\n\n')

    # Overview text for the current module
    first_line, rest = split_docstring(obj)
    f.write(first_line)
    f.write('\n\n')
    if rest:
        f.write('## Overview\n')  # Note: must include <h2> before <h3> appears.
        write_docstring_parts(f, rest, level=0)
        f.write('\n\n')

    # Submodules in the current module
    module_lines = []
    for obj, pref_path in (
        (m.obj, m.pref_dotpath) for m in members if m.mytype == MemberType.MODULE
    ):
        summ, _ = split_docstring(obj)
        module_lines.append(f'`{pref_path}`: {summ}')
    if module_lines:
        f.write('\n## Modules\n')
        f.write('\n\n'.join(module_lines))
        f.write('\n')

    # Major classes in the current module
    majclass_lines = []
    for obj, pref_path in (
        (m.obj, m.pref_dotpath) for m in members if m.mytype == MemberType.MAJOR
    ):
        summ, _ = split_docstring(obj)
        majclass_lines.append(f'`{pref_path}`: {summ}')
    if majclass_lines:
        f.write('\n## Major Classes\n')
        f.write('\n\n'.join(majclass_lines))
        f.write('\n')

    # Other members in the current module
    other_members = [m for m in members if m.mytype == MemberType.MINOR]
    if other_members:
        other_members_title = 'Other Members' if module_lines or majclass_lines else 'Members'
        f.write(f'\n## {other_members_title}\n\n')
        for m in other_members:
            write_module_other_member(f, m.obj, m.pref_dotpath, aliases_d=aliases_d)
        f.write('\n')


def render_module(out_dir: Path, page: ModulePage, render_context: RenderContext):
    out_path = page.out_path(out_dir=out_dir)
    print(f"Writing {page.pref_path} to {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with WriteIfDifferent(out_path) as f:
        f2 = render_context.get_linking_writer(f)
        assert page.obj is not None, f'Uninitialized {page}'
        assert page.pref_path is not None, f'Uninitialized {page}'
        write_module(f2, page.obj, page.pref_path, page.members, aliases_d=render_context.aliases_d)
        f2.write_link_targets()
