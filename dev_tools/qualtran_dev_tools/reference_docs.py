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

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Type

import jinja2
import tensorflow_docs.api_generator.parser
from tensorflow_docs.api_generator.generate_lib import DocGenerator
from tensorflow_docs.api_generator.obj_type import ObjType
from tensorflow_docs.api_generator.pretty_docs import (
    ClassPageBuilder,
    ClassPageInfo,
    FunctionPageBuilder,
    FunctionPageInfo,
    ModulePageBuilder,
    ModulePageInfo,
    TypeAliasPageBuilder,
    TypeAliasPageInfo,
)
from tensorflow_docs.api_generator.pretty_docs.base_page import MemberInfo
from tensorflow_docs.api_generator.pretty_docs.class_page import Methods
from tensorflow_docs.api_generator.public_api import local_definitions_filter

from .git_tools import get_git_root


def filter_type_checking(path, parent, children):
    return [(name, obj) for name, obj in children if name != 'TYPE_CHECKING']


_TYPE_ALIAS_LOCATIONS = {
    'SoquetT': ('qualtran._infra', 'composite_bloq'),
    'SoquetInT': ('qualtran._infra', 'composite_bloq'),
}


def filter_type_aliases_in_the_wrong_place(path, parent, children):
    """`local_definitions_filter` doesn't work for type aliases.

    Since the object is a value rather than a class or a function, we can't
    ask it where (i.e. in what module) it was defined.
    """
    ret = []
    for name, obj in children:
        if name in _TYPE_ALIAS_LOCATIONS:
            if path != _TYPE_ALIAS_LOCATIONS[name]:
                # Wait for the real location
                continue
        ret.append((name, obj))

    return ret


def _filter_and_sort_members(py_object: object, members: Iterable[MemberInfo]) -> List[MemberInfo]:
    """Sort `members` according to their order in the source definition.

    For example: you can order class methods according to their order of definition
    within the class.

    Additionally, we filter out members that *aren't* defined within their parent. This
    means we sort out inherited methods that are not overridden.
    """
    ordering = {name: i for i, name in enumerate(py_object.__dict__.keys())}
    fmembs = [memb for memb in members if memb.short_name in ordering]
    return sorted(fmembs, key=lambda m: ordering[m.short_name])


def mixin_custom_template(template_name: str) -> Type:
    """Return a mixin for using a custom jinja template in TemplatePageBuilder classes."""

    class _CustomTemplateMixin:
        TEMPLATE = f'templates/{template_name}.jinja'
        TEMPLATE_SEARCH_PATH = tuple([str(Path(__file__).parent.parent)])
        JINJA_ENV = jinja2.Environment(
            trim_blocks=True,
            lstrip_blocks=True,
            loader=jinja2.FileSystemLoader(TEMPLATE_SEARCH_PATH),
        )

    return _CustomTemplateMixin


class MyModulePageBuilder(mixin_custom_template('module'), ModulePageBuilder):
    """Use a custom template for module pages."""


class MyClassPageBuilder(mixin_custom_template('class'), ClassPageBuilder):
    """Use a custom template for class pages.

    Additionally, this will re-sort the class members (i.e. methods) to match
    the order in the source code.
    """

    def __init__(self, page_info):
        super().__init__(page_info)

        # Order methods. Unfortunately, the ClassPageBuilder will sort the members
        # you pass in, so we can't do this sorting where it would make the most sense in
        # MyClassPageInfo.collect_docs()
        methods = _filter_and_sort_members(
            self.page_info.py_object, self.methods.info_dict.values()
        )
        self.methods = Methods(
            info_dict={meth.short_name: meth for meth in methods},
            constructor=self.methods.constructor,
        )


class MyFunctionPageBuilder(mixin_custom_template('function'), FunctionPageBuilder):
    """Use a custom template for function pages."""


class MyTypeAliasPageBuilder(mixin_custom_template('type_alias'), TypeAliasPageBuilder):
    """Use a custom template for type alias pages."""


class MyModulePageInfo(ModulePageInfo):
    """Use custom builder and sort members for module pages."""

    DEFAULT_BUILDER_CLASS = MyModulePageBuilder

    def collect_docs(self):
        ret = super().collect_docs()
        self._classes = _filter_and_sort_members(self.py_object, self._classes)
        return ret


class MyClassPageInfo(ClassPageInfo):
    """Use custom builder and sort members for class pages."""

    DEFAULT_BUILDER_CLASS = MyClassPageBuilder

    def collect_docs(self):
        ret = super().collect_docs()
        # Note: currently the following sort is un-done by the class page builder.
        # If the upstream page builder changes to respect the member order (like for the other
        # page types), we should sort them here.
        self._methods = _filter_and_sort_members(self.py_object, self._methods)
        return ret


class MyFunctionPageInfo(FunctionPageInfo):
    """Use custom builder for function pages."""

    DEFAULT_BUILDER_CLASS = MyFunctionPageBuilder


class MyTypeAliasPageInfo(TypeAliasPageInfo):
    """Use custom builder for type alias pages."""

    DEFAULT_BUILDER_CLASS = MyTypeAliasPageBuilder


_MY_PAGE_BUILDERS = {
    ObjType.CLASS: MyClassPageInfo,
    ObjType.CALLABLE: MyFunctionPageInfo,
    ObjType.MODULE: MyModulePageInfo,
    ObjType.TYPE_ALIAS: MyTypeAliasPageInfo,
}
"""Pass in custom logic to DocGenerator."""

# Github-flavored markdown will switch to raw-html mode if a given line starts with an html tag.
# If the given line starts with a non-whitespace character other than an html tag, it will
# continue processing markdown inside the html elements.
#
# tensorflow_docs puts one space before the first e.g. `<table>` element and that is sufficient.
# It was proposed to modify the table output to emit a zero-width space on each new line before
# an html element. Unfortunately, sphinx will schlorp out all these characters and put them
# each in their own paragraph before the table (!) and there's a huge amount of extra vertical
# whitespace.
#
# So instead, we change all the tables to markdown description lists which are easy to emit,
# and they look really nice as argument/attribute lists.

MY_TABLE_TEMPLATE = """\n{title}\n\n{items}"""


class MyItemTemplate:
    """Sneak in some logic for rendering "tables" as markdown description lists.

        Description list
        : This is an example of a description list. It is one term
          followed by a definition. The definition is indented but
          starts with a colon.

    `tensorflow_docs` calls ITEMS_TEMPLATE.format(). They expect ITEMS_TEMPLATE to be a
    string but there's no reason why it can't be any python object with a "format" method.
    """

    @staticmethod
    def format(*, name: str, anchor: str, description: str) -> str:
        lines = [f'{name}{anchor}']

        desc_lines = description.splitlines()
        if len(desc_lines) == 0:
            # Empty description provided. Use a non-breaking space as the
            # description to preserve the markdown structure.
            desc_lines = ['&nbsp;']

        # Indent each description by two spaces. The first one starts with `:`
        lines.append(f': {desc_lines[0]}')
        lines.extend([f'  {dl}' for dl in desc_lines[1:]])
        lines.append('\n')

        return '\n'.join(lines)


# Monkey patch the table and items template
tensorflow_docs.api_generator.parser.TABLE_TEMPLATE = MY_TABLE_TEMPLATE
tensorflow_docs.api_generator.parser.ITEMS_TEMPLATE = MyItemTemplate


def generate_ref_docs(reporoot: Path):
    """Use `tensorflow_docs` to generate markdown reference docs."""

    # Important: we currently do not import submodules in our module's `__init__` methods,
    # so tensorflow-docs will not find a module that has not been imported. We import
    # them all here.

    import qualtran
    from qualtran import cirq_interop, drawing, resource_counting
    from qualtran.simulation import classical_sim, quimb_sim

    # prevent unused warnings:
    assert [drawing, resource_counting, cirq_interop, quimb_sim, classical_sim]

    output_dir = reporoot / 'docs/reference'
    doc_generator = DocGenerator(
        root_title="Qualtran",
        py_modules=[("qualtran", qualtran)],
        base_dir=[reporoot / 'qualtran'],
        code_url_prefix="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran",
        callbacks=[
            local_definitions_filter,
            filter_type_checking,
            filter_type_aliases_in_the_wrong_place,
        ],
        page_builder_classes=_MY_PAGE_BUILDERS,
    )
    doc_generator.build(output_dir=output_dir)


def write_ref_toc(f, grouped_paths, output_dir):
    """Make the tables of contents for sphinx from the tensorflow_docs output."""
    f.write(".. this file is autogenerated\n\n")

    for parent in sorted(grouped_paths.keys(), key=lambda p: len(p.parts)):
        f.write('\n'.join(['.. toctree::', '   :hidden:', f'   :caption: {parent.name}', '', '']))
        children = grouped_paths[parent]
        for child in sorted(children):
            f.write(f'   {child.relative_to(output_dir)}\n')
        f.write('\n')


def fixup_suffix(content: str) -> str:
    """Fix file extensions in `<a href="...">` links.

    If the markdown files contain markdown-style links: `[page title](./page.md)`, sphinx
    will happily convert the target to `page.html` in the built html output. If the markdown
    files contain html-style links `<a href="./page.md">page title</a>`, sphinx will not.

    This uses a regex with some custom substitution logic to try to intelligently replace
    `.md` and `.ipynb` suffixes in `content`'s `<a>` tags.
    """

    def _replace_internal_suffixes(match):
        """Replace the ".md" and ".ipynb" suffixes from internal links using heuristics."""
        match_dict = match.groupdict()
        before = match_dict["before"]
        after = match_dict["after"]
        full_match = match.group(0)

        # If there is nothing else after the suffix it's often a bare local file
        # reference on github, like: "see README.md", don't change it.
        if not after:
            return full_match

        # if the text after the suffix starts with anything other than ")", '"',
        # or "#" then this doesn't look like a link, don't change it.
        if after[0] not in '#"':
            return full_match

        # Don't change anything with a full url
        if "://" in before:
            return full_match

        # This is probably a local link
        return f'{before}.html{after}'

    suffix_re = re.compile(
        r"""
      (?<=\s)             # only start a match after a whitespace character
      (?P<before>[\S]+)   # At least one non-whitespace character.
      \.(md|ipynb)        # ".md" or ".ipynb" suffix (the last one).
      (?P<after>[\S]*)    # Trailing non-whitespace""",
        re.VERBOSE,
    )

    return suffix_re.sub(_replace_internal_suffixes, content)


def fixup_all_symbols_page(path: Path) -> bool:
    """Remove the 'all symbols' page.

    The relative links are all relative to the wrong location, and this information
    is redundant with our table-of-contents.

    Returns `True` if we found the all symbols page and no further fixups should be applied.
    """
    if path.name == 'all_symbols.md':
        path.unlink()
        return True
    return False


def remove_extra_files(output_dir: Path):
    """Remove metadata files that we don't use."""
    (output_dir / 'qualtran/_toc.yaml').unlink(missing_ok=True)
    (output_dir / 'qualtran/_redirects.yaml').unlink(missing_ok=True)
    (output_dir / 'qualtran/_api_cache.json').unlink(missing_ok=True)
    (output_dir / 'qualtran/api_report.pb').unlink(missing_ok=True)


def apply_fixups(reporoot: Path):
    """For each generated markdown file, apply fixups.

    - `fixup_all_symbols_page`
    - `fixup_suffix`
    """
    output_dir = reporoot / 'docs/reference'
    remove_extra_files(output_dir)

    page_paths = output_dir.glob('qualtran/**/*.md')
    for path in page_paths:
        if fixup_all_symbols_page(path):
            continue

        with path.open('r') as f:
            content = f.read()

        content = fixup_suffix(content)

        with path.open('w') as f:
            f.write(content)


def generate_ref_toc(reporoot: Path):
    """Generate a sphinx-style table of contents (TOC) from generated markdown files."""
    output_dir = reporoot / 'docs/reference'
    page_paths = output_dir.glob('qualtran/**/*.md')

    # Group according to module
    grouped_paths: Dict[Path, List] = defaultdict(list)
    for path in page_paths:
        grouped_paths[path.parent].append(path)

    with (output_dir / 'autotoc.rst').open('w') as f:
        write_ref_toc(f, grouped_paths, output_dir)


def build_reference_docs():
    reporoot = get_git_root()
    generate_ref_docs(reporoot)
    apply_fixups(reporoot)
    generate_ref_toc(reporoot)
