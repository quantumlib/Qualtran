import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import jinja2
from tensorflow_docs.api_generator.generate_lib import DocGenerator
from tensorflow_docs.api_generator.obj_type import ObjType
from tensorflow_docs.api_generator.pretty_docs import (
    ClassPageBuilder,
    ClassPageInfo,
    FunctionPageBuilder,
    FunctionPageInfo,
    ModulePageBuilder,
    ModulePageInfo,
    TypeAliasPageInfo,
)
from tensorflow_docs.api_generator.pretty_docs.base_page import MemberInfo
from tensorflow_docs.api_generator.pretty_docs.class_page import Methods
from tensorflow_docs.api_generator.public_api import local_definitions_filter

import cirq_qubitization
import cirq_qubitization.quantum_graph


def filter_type_checking(path, parent, children):
    return [(name, obj) for name, obj in children if name != 'TYPE_CHECKING']


def get_git_root() -> Path:
    """Get the root git repository path."""
    cp = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'], capture_output=True, universal_newlines=True
    )
    path = Path(cp.stdout.strip()).absolute()
    assert path.exists()
    print('git root', path)
    return path


class MyModulePageBuilder(ModulePageBuilder):
    """Use a custom template for module pages."""

    TEMPLATE = 'templates/module.jinja'
    TEMPLATE_SEARCH_PATH = tuple([str(Path(__file__).parent)])
    JINJA_ENV = jinja2.Environment(
        trim_blocks=True, lstrip_blocks=True, loader=jinja2.FileSystemLoader(TEMPLATE_SEARCH_PATH)
    )


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


class MyClassPageBuilder(ClassPageBuilder):
    """Use a custom template for class pages.

    Additionally, this will re-sort the class members (i.e. methods) to match
    the order in the source code.
    """

    TEMPLATE = 'templates/class.jinja'
    TEMPLATE_SEARCH_PATH = tuple([str(Path(__file__).parent)])
    JINJA_ENV = jinja2.Environment(
        trim_blocks=True, lstrip_blocks=True, loader=jinja2.FileSystemLoader(TEMPLATE_SEARCH_PATH)
    )

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


class MyFunctionPageBuilder(FunctionPageBuilder):
    """Use a custom template for function pages."""

    TEMPLATE = 'templates/function.jinja'
    TEMPLATE_SEARCH_PATH = tuple([str(Path(__file__).parent)])
    JINJA_ENV = jinja2.Environment(
        trim_blocks=True, lstrip_blocks=True, loader=jinja2.FileSystemLoader(TEMPLATE_SEARCH_PATH)
    )


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


_MY_PAGE_BUILDERS = {
    ObjType.CLASS: MyClassPageInfo,
    ObjType.CALLABLE: MyFunctionPageInfo,
    ObjType.MODULE: MyModulePageInfo,
    ObjType.TYPE_ALIAS: TypeAliasPageInfo,
}
"""Pass in custom logic to DocGenerator."""


def generate_ref_docs():
    """Use `tensorflow_docs` to generate markdown reference docs."""
    reporoot = get_git_root()
    output_dir = reporoot / 'docs/reference'
    doc_generator = DocGenerator(
        root_title="Qualtran",
        py_modules=[("cirq_qubitization.quantum_graph", cirq_qubitization.quantum_graph)],
        base_dir=[reporoot / 'cirq_qubitization/quantum_graph'],
        code_url_prefix="https://github.com/quantumlib/cirq-qubitization/blob/main/cirq_qubitization/quantum_graph",
        callbacks=[local_definitions_filter, filter_type_checking],
        page_builder_classes=_MY_PAGE_BUILDERS,
    )
    doc_generator.build(output_dir=output_dir)


def write_ref_toc(f, grouped_paths, output_dir):
    """Make the tables of contents for sphinx from the tensorflow_docs output."""
    f.write(".. this file is autogenerated\n\n")

    for parent in sorted(grouped_paths.keys(), key=lambda p: len(p.parts)):
        f.write(
            '\n'.join([f'.. toctree::', f'   :maxdepth: 2', f'   :caption: {parent.name}', '', ''])
        )
        children = grouped_paths[parent]
        for child in sorted(children):
            f.write(f'   {child.relative_to(output_dir)}\n')
        f.write('\n')


def generate_ref_toc():
    """Generate a sphinx-style table of contents (TOC) from generated markdown files."""
    reporoot = get_git_root()
    output_dir = reporoot / 'docs/reference'
    page_paths = (output_dir / 'cirq_qubitization').glob('quantum_graph/**/*.md')

    # Group according to module
    grouped_paths: Dict[Path, List] = defaultdict(list)
    for path in page_paths:
        grouped_paths[path.parent].append(path)

    with (output_dir / 'autotoc.rst').open('w') as f:
        write_ref_toc(f, grouped_paths, output_dir)


if __name__ == '__main__':
    generate_ref_docs()
    generate_ref_toc()
