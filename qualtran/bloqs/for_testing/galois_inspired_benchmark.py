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
"""Synthetic multi-register benchmark for Qualtran compiler and L1 IR performance testing.

This script constructs a mock bloq hierarchy using Qualtran infrastructure to serve as
a compiler and L1 IR serialization performance benchmark:

Hierarchy:
  - Primitive Bloqs: CNOT, Split, Join
  - Inner Multiplier (MockConstantMultiplier): Splits a register into qubits, applies a
    synthetic CNOT gate sequence, and joins the qubits back into the register.
  - Round Subroutine (MockMultiRegisterRound): Applies synthetic multipliers across registers.
  - Top-Level Benchmark (MockAllRoundsBenchmark): Iterates over all rounds and invokes MockMultiRegisterRound.
"""

import argparse
import logging
import os
import time
from functools import cached_property
from typing import Dict

import attrs
import numpy as np

import qualtran.l1 as ql1
from qualtran import Bloq, BloqBuilder, QUInt, Register, Signature, SoquetT
from qualtran.bloqs.basic_gates import CNOT
from qualtran.l1._ast_to_code_fast import FastL1ASTPrinter

logging.basicConfig(level=logging.ERROR)
logging.disable(logging.WARNING)
logging.getLogger("qualtran").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)


@attrs.frozen
class MockConstantMultiplier(Bloq):
    r"""Mock constant multiplier used as a performance benchmark target.

    Splits a register into `degree` qubits, applies a synthetic CNOT gate sequence,
    and joins the qubits back into the register.

    Args:
        dtype: Register data type of bitwidth `degree`.
        const_id: Integer identifier distinguishing unique synthetic multiplier variants in [2, 2^degree - 1].
    """

    dtype: QUInt
    const_id: int

    @cached_property
    def degree(self) -> int:
        return int(self.dtype.bitsize)

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('g', self.dtype)])

    def build_composite_bloq(self, bb: BloqBuilder, g: SoquetT) -> Dict[str, SoquetT]:
        g_arr = bb.split(g)
        q_list = list(g_arr[::-1])
        if self.degree >= 2:
            q_list[1], q_list[0] = bb.add(CNOT(), ctrl=q_list[1], target=q_list[0])
        g_arr = np.array(q_list)[::-1]
        g = bb.join(g_arr, dtype=self.dtype)
        return {'g': g}


@attrs.frozen
class MockMultiRegisterRound(Bloq):
    r"""Mock multi-register round applying synthetic multiplier bloqs across `num_regs` registers.

    Applies synthetic unique multiplier variants across registers:
      1. When round_id == 1: identity round (returns registers with 0 MockConstantMultiplier calls).
      2. When round_id > 1: reg[0] is identity (no-op), and reg[1..num_regs-1] invoke MockConstantMultiplier.
      3. Across all rounds in [1, 2^degree - 1], exactly (2^degree - 2) unique multiplier variants are generated.

    Args:
        round_id: Integer identifier for the round in [1, 2^degree - 1].
        num_regs: Number of registers (default: 140).
        degree: Bitwidth of each register (default: 12).
    """

    round_id: int
    num_regs: int = 140
    degree: int = 12

    @cached_property
    def quint(self) -> QUInt:
        return QUInt(bitsize=self.degree)

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('regs', dtype=self.quint, shape=(self.num_regs,))])

    def build_composite_bloq(self, bb: BloqBuilder, regs: np.ndarray) -> Dict[str, SoquetT]:
        regs_list = list(regs)
        if self.round_id != 1:
            num_non_identity = (2**self.degree) - 2
            for i in range(1, self.num_regs):
                mock_const_id = ((self.round_id - 2 + (i - 1)) % num_non_identity) + 2
                mul_k = MockConstantMultiplier(dtype=self.quint, const_id=mock_const_id)
                regs_list[i] = bb.add(mul_k, g=regs_list[i])
        return {'regs': np.array(regs_list)}


@attrs.frozen
class MockAllRoundsBenchmark(Bloq):
    r"""Top-level benchmark bloq iterating over all rounds and invoking `MockMultiRegisterRound`.

    Constructs a multi-level hierarchy of synthetic multi-register subroutines.

    Args:
        num_regs: Number of registers (default: 140).
        degree: Bitwidth of each register (default: 12).
    """

    num_regs: int = 140
    degree: int = 12

    @cached_property
    def quint(self) -> QUInt:
        return QUInt(bitsize=self.degree)

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('regs', dtype=self.quint, shape=(self.num_regs,))])

    def build_composite_bloq(self, bb: BloqBuilder, regs: np.ndarray) -> Dict[str, SoquetT]:
        for r in range(1, 2**self.degree):
            round_bloq = MockMultiRegisterRound(
                round_id=r, num_regs=self.num_regs, degree=self.degree
            )
            regs = bb.add(round_bloq, regs=regs)
        return {'regs': regs}


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic multi-register benchmark for Qualtran L1 IR performance testing."
    )
    parser.add_argument(
        "--num_regs", type=int, default=140, help="Number of registers (default: 140)"
    )
    parser.add_argument(
        "--degree", type=int, default=12, help="Bitwidth of registers (default: 12)"
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        default="mock_benchmark.qlt",
        help="Path to output .qlt file",
    )
    args = parser.parse_args()

    num_regs = args.num_regs
    degree = args.degree
    total_qubits = num_regs * degree
    num_rounds = (2**degree) - 1
    output_path = os.path.abspath(args.output_file)

    # Unique bloq definition counts
    expected_rounds = num_rounds
    expected_muls = num_rounds - 1 if num_regs >= 2 else 0
    expected_qdefs = 1 + expected_rounds + expected_muls

    print("=" * 80)
    print(f"Multi-Register Performance Benchmark (Bitwidth: {degree})")
    print("=" * 80)
    print(f"Number of registers            : {num_regs}")
    print(f"Register dtype                 : QUInt({degree})")
    print(
        f"Total register qubits          : {total_qubits} qubits ({degree} qubits x {num_regs} registers)"
    )
    print(f"Number of rounds               : {num_rounds}")
    print(f"Per-round operation            : reg[i] *= mock_const_i for i in range({num_regs})")
    print("-" * 80)
    print("Unique Bloq Definitions (qdefs):")
    print("  - Top-level bloq             : 1 (MockAllRoundsBenchmark)")
    print(f"  - Round subroutines          : {expected_rounds:,} (MockMultiRegisterRound)")
    print(f"  - Constant multipliers       : {expected_muls:,} (MockConstantMultiplier)")
    print(f"  - Total unique qdefs         : {expected_qdefs:,}")
    print(f"Output file                    : {output_path}")
    print("-" * 80)

    # Instantiate the top-level bloq
    bloq = MockAllRoundsBenchmark(num_regs=num_regs, degree=degree)

    # Serialize bloq to generate Qualtran L1 IR (.qlt format)
    print("\nSerializing bloq to Qualtran L1 IR (.qlt format)...")
    t0_l1 = time.time()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mb = ql1.L1ModuleBuilder()
    mb.add_bloqs(bloq, skip_aliases=True)
    l1_mod = mb.finalize()
    l1_txt = FastL1ASTPrinter().visit(l1_mod)
    with open(output_path, "w") as f:
        f.write(l1_txt)
    t1_l1 = time.time()
    l1_time = t1_l1 - t0_l1
    file_size = os.path.getsize(output_path)
    print(f"✓ L1 IR generation completed in {l1_time:.2f} seconds.")
    print(f"✓ Generated Qualtran L1 IR file: {output_path} ({file_size:,} bytes)")

    # Print summary of timings
    print("\n" + "=" * 80)
    print("TIMING BENCHMARK RESULTS")
    print("=" * 80)
    print(f"  Qualtran L1 IR serialization (.qlt)  : {l1_time:.2f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()
