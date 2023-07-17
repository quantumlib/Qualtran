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
from pathlib import Path

from nbformat import NotebookNode

from .notebook_execution import is_out_of_date, linkify


def test_is_out_of_date(tmpdir):
    tmpdir = Path(tmpdir)
    src_path = tmpdir / 'src.txt'
    dest_path = tmpdir / 'dest.txt'

    with src_path.open('w') as f:
        f.write('hello')

    assert is_out_of_date(src_path, dest_path)

    with dest_path.open('w') as f:
        f.write('HELLO')

    assert not is_out_of_date(src_path, dest_path)

    with src_path.open('w+') as f:
        f.write('2')

    assert is_out_of_date(src_path, dest_path)


def test_linkify():
    nb = NotebookNode()
    nb.cells = []
    nb.cells.append(NotebookNode(cell_type='code', source='print("hello world")'))
    nb.cells.append(NotebookNode(cell_type='markdown', source='Check the `Bloq` documentation'))

    cell0 = nb.cells[0].copy()
    linkify(nb)
    assert nb.cells[0] == cell0
    assert nb.cells[1].source == 'Check the [`Bloq`](/reference/qualtran/Bloq.md) documentation'
