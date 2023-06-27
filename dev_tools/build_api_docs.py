import subprocess
import types
from pathlib import Path
from typing import Iterable, Sequence

import jinja2
from tensorflow_docs.api_generator.doc_generator_visitor import ApiTreeNode
from tensorflow_docs.api_generator.generate_lib import DocGenerator
from tensorflow_docs.api_generator.obj_type import ObjType
from tensorflow_docs.api_generator.pretty_docs import (
    ClassPageBuilder,
    ClassPageInfo,
    FunctionPageBuilder,
    FunctionPageInfo,
    ModulePageBuilder,
    ModulePageInfo,
    TemplatePageBuilder,
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
    TEMPLATE = 'templates/module.jinja'
    TEMPLATE_SEARCH_PATH = tuple([str(Path(__file__).parent)])
    JINJA_ENV = jinja2.Environment(
        trim_blocks=True, lstrip_blocks=True, loader=jinja2.FileSystemLoader(TEMPLATE_SEARCH_PATH)
    )


class MyClassPageBuilder(ClassPageBuilder):
    TEMPLATE = 'templates/class.jinja'
    TEMPLATE_SEARCH_PATH = tuple([str(Path(__file__).parent)])
    JINJA_ENV = jinja2.Environment(
        trim_blocks=True, lstrip_blocks=True, loader=jinja2.FileSystemLoader(TEMPLATE_SEARCH_PATH)
    )

    def __init__(self, page_info):
        super().__init__(page_info)

        # Order methods yet again
        methods = _filter_and_sort_members(
            self.page_info.py_object, self.methods.info_dict.values()
        )
        self.methods = Methods(
            info_dict={meth.short_name: meth for meth in methods},
            constructor=self.methods.constructor,
        )


class MyFunctionPageBuilder(FunctionPageBuilder):
    TEMPLATE = 'templates/function.jinja'
    TEMPLATE_SEARCH_PATH = tuple([str(Path(__file__).parent)])
    JINJA_ENV = jinja2.Environment(
        trim_blocks=True, lstrip_blocks=True, loader=jinja2.FileSystemLoader(TEMPLATE_SEARCH_PATH)
    )


def _filter_and_sort_members(py_object: object, members: Iterable[MemberInfo]):
    ordering = {name: i for i, name in enumerate(py_object.__dict__.keys())}
    fmembs = [memb for memb in members if memb.short_name in ordering]
    return sorted(fmembs, key=lambda m: ordering[m.short_name])


class MyModulePageInfo(ModulePageInfo):
    DEFAULT_BUILDER_CLASS = MyModulePageBuilder

    def collect_docs(self):
        ret = super().collect_docs()
        self._classes = _filter_and_sort_members(self.py_object, self._classes)
        return ret


class MyClassPageInfo(ClassPageInfo):
    DEFAULT_BUILDER_CLASS = MyClassPageBuilder

    def collect_docs(self):
        ret = super().collect_docs()
        self._methods = _filter_and_sort_members(self.py_object, self._methods)
        return ret


class MyFunctionPageInfo(FunctionPageInfo):
    DEFAULT_BUILDER_CLASS = MyFunctionPageBuilder


my_page_builders = {
    ObjType.CLASS: MyClassPageInfo,
    ObjType.CALLABLE: MyFunctionPageInfo,
    ObjType.MODULE: MyModulePageInfo,
    ObjType.TYPE_ALIAS: TypeAliasPageInfo,
}


def generate_ref_docs():
    reporoot = get_git_root()
    output_dir = reporoot / 'docs/reference'
    doc_generator = DocGenerator(
        root_title="Qualtran",
        py_modules=[("cirq_qubitization.quantum_graph", cirq_qubitization.quantum_graph)],
        base_dir=str(reporoot / 'cirq_qubitization/quantum_graph'),
        code_url_prefix="https://github.com/quantumlib/cirq-qubitization/blob/master/",
        site_path="reference/python",
        callbacks=[local_definitions_filter, filter_type_checking],
        page_builder_classes=my_page_builders,
    )
    doc_generator.build(output_dir=output_dir)


if __name__ == '__main__':
    generate_ref_docs()
