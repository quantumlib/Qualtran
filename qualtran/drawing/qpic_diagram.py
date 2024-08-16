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

"""Classes for drawing latex diagrams for bloqs with QPIC - https://github.com/qpic/qpic.

QPIC is not a dependency of Qualtran and must be manually installed by users via
`pip install qpic`.
"""
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Union

from qualtran import DanglingT, LeftDangle, QBit, RightDangle, Side, Soquet
from qualtran.drawing.musical_score import (
    _soq_to_symb,
    Circle,
    LarrowTextBox,
    ModPlus,
    RarrowTextBox,
    Text,
    TextBox,
    WireSymbol,
)

if TYPE_CHECKING:
    from qualtran import Bloq, Connection, QDType, Signature


def _wire_name_prefix_for_soq(soq: Soquet) -> str:
    return soq.pretty()


def _format_label_text(label: str, scale: float = 0.8) -> str:
    replacements = {
        '&': r'\&',
        '_': r'\_',
        '<=': r'$\le$',
        '>=': r'$\ge$',
        '<': r'$<$',
        '>': r'$>$',
        '⨁': r'$\oplus$',
        '⊕': r'$\oplus$',
        '⇋': r'$\rightleftharpoons$',
        '⨂': r'$\otimes$',
        '∧': r'$\land$',
        '†': r'$^\dagger$',
    }
    for key, val in replacements.items():
        label = label.replace(key, val)
    scale_prefix = r'\scalebox{' + str(scale) + r'}{'
    scale_suffix = r'}'
    return r'\textrm{' + scale_prefix + label + scale_suffix + r'}'


def _get_wire_symbol_text(symbol: WireSymbol) -> str:
    """Wire symbol must be a type that has a text field."""
    assert isinstance(symbol, (LarrowTextBox, RarrowTextBox, Text, TextBox))
    symbol_text = symbol.text.strip()
    return symbol_text


def _soq_symbol_to_text(symbol: WireSymbol) -> str:
    symbol_text = _get_wire_symbol_text(symbol)
    return _format_label_text(symbol_text)


def _gate_width_for_text(text: str) -> int:
    """Scale the width of the gate with number of characters in its label."""
    return 5 + 4 * len(text)


def _soq_symbol_width(symbol: WireSymbol) -> int:
    symbol_text = _get_wire_symbol_text(symbol)
    return _gate_width_for_text(symbol_text)


class QpicWireManager:
    """Methods to manage allocation/deallocation of wires for QPIC diagrams.

    QPIC places wires in the order in which they are defined. For each soquet, the wire manager
    allocates a new wire with prefix `_wire_name_prefix_for_soq(soq)` and an integer suffix that
    corresponds to the smallest integer which does not correspond to an allocated wire.
    """

    def __init__(self):
        self._soq_to_wire_name_tuple: Dict[Soquet, Tuple[str, int]] = {}
        self._alloc_wires_with_prefix: Dict[str, Set[int]] = defaultdict(set)

    def _wire_name_tuple_for_soq(self, soq: Soquet) -> Tuple[str, int]:
        prefix = _wire_name_prefix_for_soq(soq)
        allocated_suffixes = self._alloc_wires_with_prefix[prefix]
        next_i = next(i for i in range(len(allocated_suffixes) + 1) if i not in allocated_suffixes)
        return prefix, next_i

    def _wire_name_tuple_to_str(self, wire_name: Tuple[str, int]) -> str:
        prefix, i = wire_name
        return prefix + '_' + str(i) if i else prefix

    def alloc_wire_for_soq(self, soq: Soquet) -> str:
        prefix, i = self._wire_name_tuple_for_soq(soq)
        self._alloc_wires_with_prefix[prefix].add(i)
        self._soq_to_wire_name_tuple[soq] = (prefix, i)
        return self._wire_name_tuple_to_str((prefix, i))

    def dealloc_wire_for_soq(self, soq: Soquet) -> str:
        prefix, i = self._soq_to_wire_name_tuple[soq]
        self._alloc_wires_with_prefix[prefix].remove(i)
        self._soq_to_wire_name_tuple.pop(soq)
        return self._wire_name_tuple_to_str((prefix, i))

    def soq_to_wirename(self, soq) -> str:
        assert soq in self._soq_to_wire_name_tuple
        return self._wire_name_tuple_to_str(self._soq_to_wire_name_tuple[soq])

    def soq_to_wirelabel(self, soq) -> str:
        return _format_label_text(self.soq_to_wirename(soq))


class QpicCircuit:
    """Builds data corresponding to the input specification of a QPIC diagram"""

    def __init__(self):
        # Init empty data structures.
        self.soq_map = {}
        self.wire_manager = QpicWireManager()
        self.allocated_wires = set()
        self.gates = []
        # Macros useful for styling diagrams.
        self.wires = ['DEFINE off color=white', 'DEFINE on color=black']
        # An empty wire is a transparent wire which can be used to place bloqs that
        # don't have any incoming or outgoing edges.
        self.empty_wire = None

    def add_left_wires_for_signature(self, signature: 'Signature') -> None:
        for reg in signature.lefts():
            for idx in reg.all_idxs():
                self._alloc_wire_for_soq(Soquet(LeftDangle, reg, idx))
        # Add horizontal blank space since left dangling wires would have annotations
        # corresponding to their QDType, which takes up horizontal space.
        self.wires += ['LABEL length=10']

    def add_right_wires_for_signature(self, signature: 'Signature') -> None:
        add_space = False
        for reg in signature.rights():
            if reg.side & Side.LEFT:
                continue
            for idx in reg.all_idxs():
                soq = Soquet(RightDangle, reg, idx)
                wire_name = self.wire_manager.soq_to_wirename(self.soq_map[soq])
                self.gates += [f'{wire_name} / {_format_label_text(soq.pretty(), scale=0.5)} ']
                add_space = True
        if add_space:
            self.gates += ['LABEL length=10']

    @property
    def data(self) -> List[str]:
        return self.wires + self.gates

    def _add_soq(self, soq: Soquet) -> Tuple[str, Optional[str]]:
        symbol = _soq_to_symb(soq)
        suffix = ''
        wire = self.wire_manager.soq_to_wirename(self.soq_map[soq])
        if soq.reg.side == Side.LEFT:
            suffix = ':off'
            self._dealloc_wire_for_soq(soq)
        if soq.reg.side == Side.RIGHT:
            suffix = ':on'
        wire += suffix
        if isinstance(symbol, Circle):
            return f'{wire}' if symbol.filled else f'-{wire}', None
        elif isinstance(symbol, ModPlus):
            return f'+{wire}', None
        symbol_text = _soq_symbol_to_text(symbol)
        width = _soq_symbol_width(symbol)
        shape = 'box'
        if isinstance(symbol, LarrowTextBox) and not symbol.text.strip():
            shape = '<'
        if isinstance(symbol, RarrowTextBox) and not symbol.text.strip():
            shape = '>'
        return f'{wire} ', f'G:width={width}:shape={shape} {symbol_text}'

    def _dealloc_wire_for_soq(self, soq: Soquet) -> None:
        self.wire_manager.dealloc_wire_for_soq(self.soq_map[soq])
        self.soq_map.pop(soq)

    @classmethod
    def _dtype_label_for_wire(cls, wire_name: str, dtype: 'QDType') -> List[str]:
        if dtype != QBit():
            dtype_str = _format_label_text(str(dtype), scale=0.5)
            return [f'{wire_name} / {dtype_str}']
        else:
            return []

    def _alloc_wire_for_soq(self, soq: Soquet) -> None:
        self.soq_map[soq] = soq
        wire_name = self.wire_manager.alloc_wire_for_soq(soq)
        if wire_name in self.allocated_wires:
            return
        self.allocated_wires.add(wire_name)
        # For a soquet on THRU/LEFT register, a wire should already be allocated.
        # assert soq.reg.side == Side.RIGHT
        if isinstance(soq.binst, DanglingT):
            # A RIGHT wire on a LeftDangle bloq should start at the beginning.
            assert soq.binst == LeftDangle
            wire_label = self.wire_manager.soq_to_wirelabel(soq)
            self.wires += [f'{wire_name} W {wire_label}']
            self.wires += self._dtype_label_for_wire(wire_name, soq.reg.dtype)

        else:
            # A RIGHT wire on an intermediate Bloq should start as off and be turned on
            # by the Bloq.
            self.wires += [f'{wire_name} W off']

    def add_connection(self, cxn: 'Connection'):
        if cxn.left not in self.soq_map:
            self._alloc_wire_for_soq(cxn.left)
        self.soq_map[cxn.right] = self.soq_map[cxn.left]

    def _add_bloq_with_no_wire(self, bloq: 'Bloq'):
        """Add bloq, like GlobalPhase, which doesn't have any incoming & outgoing wires."""
        if not self.empty_wire:
            self.empty_wire = '_empty_wire'
            self.wires.insert(2, f'{self.empty_wire} W off')
        gate_text = str(bloq)
        width = _gate_width_for_text(gate_text)
        self.gates += [f'{self.empty_wire} G:width={width}:shape=8 {gate_text}']

    def add_bloq(self, bloq: 'Bloq', pred: List['Connection'], succ: List['Connection']) -> None:
        controls, targets, wire_dtype_labels = [], [], []

        if not (pred or succ):
            self._add_bloq_with_no_wire(bloq)
            return

        def add_soq(soq: Soquet):
            wire, gate = self._add_soq(soq)
            if gate is None:
                controls.append(wire)
            else:
                targets.append(wire + gate)
            if soq.reg.side == Side.RIGHT:
                wire = self.wire_manager.soq_to_wirename(self.soq_map[soq])
                wire_dtype_labels.extend(self._dtype_label_for_wire(wire, soq.reg.dtype))

        for cxn in pred:
            self.add_connection(cxn)
            add_soq(cxn.right)

        for cxn in succ:
            self.add_connection(cxn)
            if cxn.left.reg.side == Side.RIGHT:
                add_soq(cxn.left)
        self.gates += [' '.join(targets + controls)]
        self.gates += wire_dtype_labels


def get_qpic_data(bloq: 'Bloq', file_path: Union[None, pathlib.Path, str] = None) -> List[str]:
    """Get the input data that can be used to draw a latex diagram for `bloq` using `qpic`.

    Args:
        bloq: Bloqs to draw the latex diagram for
        file_path: If specified, the output is stored at the file. Otherwise, the data is returned.

    Returns:
        A list of strings representing the input to `qpic` utility to draw a latex diagram
        for bloq. If specified, the output is written to `file_path`.
    """
    cbloq = bloq.as_composite_bloq()
    qpic_circuit = QpicCircuit()
    qpic_circuit.add_left_wires_for_signature(cbloq.signature)
    for binst, pred, succ in cbloq.iter_bloqnections():
        qpic_circuit.add_bloq(binst.bloq, pred, succ)
    qpic_circuit.add_right_wires_for_signature(cbloq.signature)
    if file_path:
        with open(file_path, 'w') as f:
            f.write('\n'.join(qpic_circuit.data))
    return qpic_circuit.data


def _to_snake_case(name):
    """Convert camel case to snake case. Taken from https://stackoverflow.com/a/1176023"""
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('__([A-Z])', r'_\1', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()


def qpic_input_to_diagram(
    qpic_file: Union[pathlib.Path, str],
    output_file: Union[None, pathlib.Path, str] = None,
    output_type: str = 'pdf',
) -> str:
    r"""Invoke `qpic` script to generate output diagram of type qpic/tex/pdf/png.

    Outputs one of the following files, based on `output_type`:
     - qpic_file.with_suffix('.qpic') - Copies the input qpic_file to output_file.
     - qpic_file.with_suffix('.tex') - Obtained via `qpic -f base_file_path.qpic`
     - qpic_file.with_suffix('.pdf') - Uses `pdflatex` tool to convert tex to pdf. See
        https://tug.org/applications/pdftex/ for more details on how to install `pdflatex`.
     - qpic_file.with_suffix('.png') - Uses `convert` tool to convert pdf to png. See
        https://imagemagick.org/ for more details on how to install `convert`.

    Args:
        qpic_file: Path to file containing input that should be passed to the `qpic` script.
        output_file: Optional path to the output where generated diagram should be saved. Defaults to
            qpic_file.with_suffix(f".{output_type}")
        output_type: Format of the diagram generated using qpic. Supported output types are one of
            ['qpic', 'tex', 'pdf', 'png']
    Returns:
        Path of the output file where generated diagram is saved. By default, corresponds to
        qpic_file.with_suffix(f".{output_type}")'
    """
    supported_types = ['qpic', 'tex', 'pdf', 'png']
    if output_type not in supported_types:
        raise ValueError(f"{output_type=} not supported. Must be one of {supported_types}")

    if isinstance(qpic_file, str):
        qpic_file = pathlib.Path(qpic_file)
    if output_file is None:
        output_file = qpic_file.with_suffix(f'.{output_type}')

    if isinstance(output_file, str):
        output_file = pathlib.Path(output_file)

    # 1. If output type is qpic, simply copy input to output.
    if output_type == 'qpic' and qpic_file != output_file:
        shutil.copy(qpic_file, output_file)
        return output_file.name

    def move_to_output(src_path: pathlib.Path):
        """Helper to move input file to output file and return the output file name."""
        shutil.move(src_path, output_file)
        return output_file.name

    # 2. Run the qpic script and generate the corresponding TEX output.
    tex_file = str(qpic_file) + '_tex'
    command = f'qpic {qpic_file} -f tex -o {tex_file}'
    subprocess.run(command.split(), check=True, capture_output=True)
    tex_file_path = pathlib.Path(tex_file)

    if output_type == 'tex':
        return move_to_output(tex_file_path)

    # 3. Convert TEX to PDF
    output_format = 'pdf' if output_type in ['pdf', 'png'] else 'dvi'
    command = f'pdflatex -interaction batchmode -output-directory {tex_file_path.parent} -output-format={output_format} {tex_file_path} '
    subprocess.run(command.split(), check=False, capture_output=True)
    pdf_file_path = tex_file_path.with_suffix(f'.{output_format}')
    os.remove(tex_file_path)

    if output_type == 'pdf':
        return move_to_output(pdf_file_path)

    # 4. Convert PDF to PNG
    if output_type == 'png':
        png_file_path = tex_file_path.with_suffix('.png')
        command = f'convert -density 1200 -quality 100 {pdf_file_path} {png_file_path}'
        subprocess.run(command.split(), check=True, capture_output=True)
        os.remove(pdf_file_path)
        return move_to_output(png_file_path)

    raise ValueError(f"Could not generate qpic diagram {output_type} for {qpic_file}")


def qpic_diagram_for_bloq(
    bloq: 'Bloq', base_file_path: Union[None, pathlib.Path, str] = None, output_type: str = 'pdf'
) -> str:
    r"""Generate latex diagram for `bloq` by invoking `qpic`. Assumes qpic is already installed.

    Outputs one of the following files, based on `output_type`:
     - base_file_path + '.qpic' - Obtained via get_qpic_data(bloq)
     - base_file_path + '.tex' - Obtained via `qpic -f base_file_path.qpic`
     - base_file_path + '.pdf' - Uses `pdflatex` tool to convert tex to pdf. See
        https://tug.org/applications/pdftex/ for more details on how to install `pdflatex`.
     - base_file_path + '.png' - Uses `convert` tool to convert pdf to png. See
        https://imagemagick.org/ for more details on how to install `convert`.

    Args:
        bloq: The bloq to generate a qpic diagram for
        base_file_path: Prefix of the path where output file is saved. The output file corresponds
            to f'{base_file_path}.{output_type}'
        output_type: Format of the diagram generated using qpic. Supported output types are one of
            ['qpic', 'tex', 'pdf', 'png']
    Returns:
        Path of the output file where generated diagram is saved. By default, corresponds to
        f'./bloq_class_name.{output_type}'
    """
    if base_file_path is None:
        base_file_path = f'./{_to_snake_case(bloq.__class__.__name__)}'

    if isinstance(base_file_path, str):
        base_file_path = pathlib.Path(base_file_path)

    output_file = base_file_path.with_suffix(f'.{output_type}')
    qpic_file = tempfile.NamedTemporaryFile(delete=True)
    get_qpic_data(bloq, qpic_file.name)
    return qpic_input_to_diagram(pathlib.Path(qpic_file.name), output_file, output_type=output_type)
