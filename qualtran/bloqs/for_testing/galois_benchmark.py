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
"""Script to generate and benchmark Qualtran L1 IR serialization for GF(2, 12) constant multiplications.

Hierarchy:
  - Inner Bloq (GF2PowerMulRound): Applies one round of reg[i] *= gamma^i across 140 QGF(2, 12) registers.
  - Top-level Bloq (GF2AllGammasPowerMul): Iterates over all gamma in [1, 2^degree - 1] and calls GF2PowerMulRound.

Outputs:
  Timing and generation of Qualtran L1 IR (.qlt file).
"""

import argparse
import logging
import os
import time
import warnings
from functools import cached_property, lru_cache
from typing import Dict, Optional, Tuple

import attrs
import numpy as np

import qualtran as qlt
import qualtran.l1 as ql1
from qualtran import Bloq, BloqBuilder, QGF, Register, Signature, SoquetT
from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2MulK
from qualtran.drawing import Text, TextBox, WireSymbol
from qualtran.l1._ast_to_code_fast import FastL1ASTPrinter

# Suppress warnings and non-error log messages
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logging.disable(logging.WARNING)
logging.getLogger("qualtran").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)


@lru_cache(maxsize=32)
def compute_all_gamma_powers(characteristic: int, degree: int, num_regs: int) -> np.ndarray:
    r"""Computes all $\gamma^i$ powers for $\gamma \in [1, \text{characteristic}^{\text{degree}} - 1]$ and $i \in [0, \text{num\_regs} - 1]$.

    Args:
        characteristic: Characteristic of the Galois Field.
        degree: Degree of the Galois Field.
        num_regs: Number of registers.

    Returns:
        2D numpy array of shape `(characteristic**degree - 1, num_regs)` containing integer constants.
    """
    import galois

    gf = galois.GF(characteristic**degree)
    num_gammas = (characteristic**degree) - 1
    gammas = gf(np.arange(1, num_gammas + 1))
    i_arr = np.arange(num_regs)
    powers = gammas[:, None] ** i_arr[None, :]
    return np.asarray(powers, dtype=int)


def get_gamma_powers(gamma: int, characteristic: int, degree: int, num_regs: int) -> np.ndarray:
    r"""Retrieves precomputed $\gamma^i$ constants for a given $\gamma$ base in GF(p^m).

    Args:
        gamma: Integer representation of the base gamma element (1-indexed).
        characteristic: Characteristic of the Galois Field.
        degree: Degree of the Galois Field.
        num_regs: Number of registers.

    Returns:
        1D numpy array of shape `(num_regs,)` containing integer power constants.
    """
    all_powers = compute_all_gamma_powers(characteristic, degree, num_regs)
    return all_powers[gamma - 1]


@attrs.frozen
class GF2PowerMulRound(Bloq):
    r"""Inner Bloq: Applies one round of power multiplications: reg[i] *= gamma^i for i in [0, num_regs - 1].

    Args:
        gamma: Base constant in GF(2^degree).
        num_regs: Number of QGF registers (default: 140).
        characteristic: Characteristic of the Galois Field (default: 2).
        degree: Degree of the Galois Field (default: 12).
    """

    gamma: int
    num_regs: int = 140
    characteristic: int = 2
    degree: int = 12

    @cached_property
    def qgf(self) -> QGF:
        return QGF(characteristic=self.characteristic, degree=self.degree)

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('regs', dtype=self.qgf, shape=(self.num_regs,))])

    def build_composite_bloq(self, bb: BloqBuilder, regs: np.ndarray) -> Dict[str, SoquetT]:
        consts = get_gamma_powers(
            gamma=self.gamma,
            characteristic=self.characteristic,
            degree=self.degree,
            num_regs=self.num_regs,
        )
        regs_list = list(regs)
        for i, const_i in enumerate(consts):
            if const_i != 1:  # gamma^0 = 1 is identity (no-op)
                mul_k = GF2MulK(dtype=self.qgf, const=int(const_i))
                regs_list[i] = bb.add(mul_k, g=regs_list[i])
        return {'regs': np.array(regs_list)}

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text('GFMul')
        assert reg.name == 'regs'
        (i,) = idx
        return TextBox(f'gamma^{i}')

    def __str__(self):
        return f'GF2PowerMulRound({self.gamma})'


@attrs.frozen
class GF2AllGammasPowerMul(Bloq):
    r"""Top-level Bloq: Iterates over all gamma in [1, 2^degree - 1] and invokes GF2PowerMulRound.

    Args:
        num_regs: Number of QGF registers (default: 140).
        characteristic: Characteristic of the Galois Field (default: 2).
        degree: Degree of the Galois Field (default: 12).
    """

    num_regs: int = 140
    characteristic: int = 2
    degree: int = 12

    @cached_property
    def qgf(self) -> QGF:
        return QGF(characteristic=self.characteristic, degree=self.degree)

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('regs', dtype=self.qgf, shape=(self.num_regs,))])

    def build_composite_bloq(self, bb: BloqBuilder, regs: np.ndarray) -> Dict[str, SoquetT]:
        for g in range(1, 2**self.degree):
            round_bloq = GF2PowerMulRound(
                gamma=g,
                num_regs=self.num_regs,
                characteristic=self.characteristic,
                degree=self.degree,
            )
            regs = bb.add(round_bloq, regs=regs)
        return {'regs': regs}


def verify_dumped_file(
    output_path: str,
    original_bloq: GF2AllGammasPowerMul,
    root_bloq_key: str,
    expected_qdefs_count: int,
) -> bool:
    """Reload the dumped .qlt file and verify that it is a faithful representation.

    Args:
        output_path: Path to the written .qlt file.
        original_bloq: The original top-level GF2AllGammasPowerMul bloq.
        root_bloq_key: Expected root bloq key string.
        expected_qdefs_count: Expected number of qdefs in the module.

    Returns:
        True if all verification checks pass.

    Raises:
        AssertionError: If any verification check fails.
    """
    print("\n" + "=" * 80)
    print("RUNNING CORRECTNESS & FAITHFULNESS SANITY CHECKS (--validate)")
    print("=" * 80)
    t0 = time.time()

    # 1. Read file and reload module
    print("1. Parsing and evaluating dumped .qlt file with safe=True...")
    with open(output_path, "r") as f:
        l1_code = f.read()

    loaded_module = ql1.load_module(l1_code, safe=True)
    t_load = time.time() - t0
    print(f"   ✓ Successfully loaded {len(loaded_module)} bloq definition(s) in {t_load:.2f} s.")

    # 2. Check bloq keys and definitions count
    print("2. Verifying module definitions and root bloq key...")
    assert root_bloq_key in loaded_module, f"Root key '{root_bloq_key}' not found in loaded module."
    assert (
        len(loaded_module) == expected_qdefs_count
    ), f"Expected {expected_qdefs_count} qdefs, found {len(loaded_module)}."
    print(f"   ✓ Root bloq key '{root_bloq_key}' found and definition count matches.")

    # 3. Check root signature equivalence
    print("3. Verifying root bloq signature...")
    loaded_root = loaded_module[root_bloq_key]
    assert (
        loaded_root.signature == original_bloq.signature
    ), f"Signature mismatch: {loaded_root.signature} != {original_bloq.signature}"
    print(f"   ✓ Signature matches: {loaded_root.signature}")

    # 4. Verify composite structure & subbloq instance counts
    print("4. Verifying graph decomposition topology and round counts...")
    assert isinstance(
        loaded_root, qlt.CompositeBloq
    ), f"Expected CompositeBloq for loaded root, got {type(loaded_root)}"
    num_rounds = (2**original_bloq.degree) - 1
    assert (
        len(loaded_root.bloq_instances) == num_rounds
    ), f"Expected {num_rounds} round subbloqs, found {len(loaded_root.bloq_instances)}"
    print(f"   ✓ Verified {num_rounds} round subbloq instances in root circuit.")

    # 5. Check round subbloqs and constant multipliers
    print("5. Verifying round subbloqs and constituent multipliers...")
    for g, binst in enumerate(loaded_root.bloq_instances, start=1):
        subbloq = binst.bloq
        assert isinstance(
            subbloq, qlt.CompositeBloq
        ), f"Expected CompositeBloq for round {g}, got {type(subbloq)}"
        pows = get_gamma_powers(
            gamma=g,
            characteristic=original_bloq.characteristic,
            degree=original_bloq.degree,
            num_regs=original_bloq.num_regs,
        )
        expected_multipliers = int(np.sum(pows != 1))
        assert (
            len(subbloq.bloq_instances) == expected_multipliers
        ), f"Round gamma={g}: expected {expected_multipliers} multipliers, found {len(subbloq.bloq_instances)}"
    print(
        f"   ✓ Verified all {num_rounds} rounds against algebraic non-identity power distributions."
    )

    total_verify_time = time.time() - t0
    print("-" * 80)
    print(f"✓ All verification checks passed in {total_verify_time:.2f} s.")
    print("=" * 80)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Compute gate counts and export Qualtran L1 IR for all gamma in [1, 2^b) power-of-gamma multiplications."
    )
    parser.add_argument(
        "--num_regs", type=int, default=140, help="Number of QGF registers (default: 140)"
    )
    parser.add_argument(
        "--degree", type=int, default=12, help="Degree of the field GF(2, b) (default: 12)"
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        default="gf2_mul_140_regs.qlt",
        help="Path to output .qlt file",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="Re-load the dumped .qlt file and run sanity checks to verify correctness",
    )
    args = parser.parse_args()

    num_regs = args.num_regs
    degree = args.degree
    characteristic = 2
    total_qubits = num_regs * degree
    num_gammas = (2**degree) - 1
    output_path = os.path.abspath(args.output_file)

    print("=" * 80)
    print(f"GF(2, {degree}) Multi-Register Power-of-Gamma Multiplications in Qualtran")
    print("=" * 80)
    print(f"Number of registers        : {num_regs}")
    print(
        f"Galois Field               : GF({characteristic}^{degree}) ({degree} qubits per element)"
    )
    print(f"Total register qubits      : {degree} * {num_regs} = {total_qubits} qubits")
    print(f"Base gamma range           : [1, {num_gammas}] ({num_gammas} non-zero constants)")
    print(f"Subroutine per round (gamma): reg[i] *= gamma^i for i in [0, {num_regs - 1}]")
    print(f"Output L1 IR (.qlt) file   : {output_path}")
    if args.validate:
        print("Verification               : Enabled (--validate)")
    print("-" * 80)

    # Instantiate the top-level bloq
    bloq = GF2AllGammasPowerMul(num_regs=num_regs, characteristic=characteristic, degree=degree)

    print("\nSerializing bloq to Qualtran L1 IR (.qlt format)...")
    t0_l1 = time.time()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mb = ql1.L1ModuleBuilder()
    root_bloq_key = mb.add_bloqs(bloq, skip_aliases=True)
    l1_mod = mb.finalize()
    l1_txt = FastL1ASTPrinter().visit(l1_mod)
    with open(output_path, "w") as f:
        f.write(l1_txt)
    t1_l1 = time.time()
    l1_time = t1_l1 - t0_l1
    file_size = os.path.getsize(output_path)
    print(f"✓ L1 IR generation completed in {l1_time:.2f} seconds.")
    print(f"✓ Generated Qualtran L1 IR file: {output_path} ({file_size:,} bytes)")

    # Optional validation of the dumped file
    if args.validate:
        verify_dumped_file(
            output_path=output_path,
            original_bloq=bloq,
            root_bloq_key=root_bloq_key,
            expected_qdefs_count=len(l1_mod.qdefs),
        )

    # Print summary of timings
    print("\n" + "=" * 80)
    print("TIMING BENCHMARK RESULTS")
    print("=" * 80)
    print(f"  Qualtran L1 IR serialization (.qlt)  : {l1_time:.2f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()
