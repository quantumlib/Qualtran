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

from typing import Type

import qualtran.bloqs.arithmetic.addition
import qualtran.bloqs.arithmetic.bitwise
import qualtran.bloqs.arithmetic.comparison
import qualtran.bloqs.arithmetic.conversions
import qualtran.bloqs.arithmetic.hamming_weight
import qualtran.bloqs.arithmetic.multiplication
import qualtran.bloqs.arithmetic.negate
import qualtran.bloqs.arithmetic.permutation
import qualtran.bloqs.arithmetic.sorting
import qualtran.bloqs.basic_gates.cnot
import qualtran.bloqs.basic_gates.hadamard
import qualtran.bloqs.basic_gates.identity
import qualtran.bloqs.basic_gates.on_each
import qualtran.bloqs.basic_gates.rotation
import qualtran.bloqs.basic_gates.s_gate
import qualtran.bloqs.basic_gates.swap
import qualtran.bloqs.basic_gates.t_gate
import qualtran.bloqs.basic_gates.toffoli
import qualtran.bloqs.basic_gates.x_basis
import qualtran.bloqs.basic_gates.y_gate
import qualtran.bloqs.basic_gates.z_basis
import qualtran.bloqs.block_encoding
import qualtran.bloqs.block_encoding.chebyshev_polynomial
import qualtran.bloqs.block_encoding.lcu_block_encoding
import qualtran.bloqs.block_encoding.lcu_select_and_prepare
import qualtran.bloqs.block_encoding.linear_combination
import qualtran.bloqs.block_encoding.phase
import qualtran.bloqs.block_encoding.product
import qualtran.bloqs.block_encoding.tensor_product
import qualtran.bloqs.block_encoding.unitary
import qualtran.bloqs.bookkeeping
import qualtran.bloqs.bookkeeping.allocate
import qualtran.bloqs.bookkeeping.arbitrary_clifford
import qualtran.bloqs.bookkeeping.auto_partition
import qualtran.bloqs.bookkeeping.cast
import qualtran.bloqs.bookkeeping.free
import qualtran.bloqs.bookkeeping.join
import qualtran.bloqs.bookkeeping.partition
import qualtran.bloqs.bookkeeping.split
import qualtran.bloqs.chemistry.black_boxes
import qualtran.bloqs.chemistry.df.double_factorization
import qualtran.bloqs.chemistry.df.prepare
import qualtran.bloqs.chemistry.df.select_bloq
import qualtran.bloqs.chemistry.hubbard_model.qubitization.prepare_hubbard
import qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard
import qualtran.bloqs.chemistry.pbc.first_quantization.prepare
import qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu
import qualtran.bloqs.chemistry.pbc.first_quantization.prepare_t
import qualtran.bloqs.chemistry.pbc.first_quantization.prepare_uv
import qualtran.bloqs.chemistry.pbc.first_quantization.prepare_zeta
import qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_nu
import qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_t
import qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_uv
import qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_and_prepare
import qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_t
import qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_uv
import qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare
import qualtran.bloqs.chemistry.pbc.first_quantization.select_t
import qualtran.bloqs.chemistry.pbc.first_quantization.select_uv
import qualtran.bloqs.chemistry.sf.prepare
import qualtran.bloqs.chemistry.sf.select_bloq
import qualtran.bloqs.chemistry.sf.single_factorization
import qualtran.bloqs.chemistry.sparse.prepare
import qualtran.bloqs.chemistry.sparse.select_bloq
import qualtran.bloqs.chemistry.thc.prepare
import qualtran.bloqs.chemistry.thc.select_bloq
import qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt
import qualtran.bloqs.chemistry.trotter.grid_ham.kinetic
import qualtran.bloqs.chemistry.trotter.grid_ham.potential
import qualtran.bloqs.chemistry.trotter.grid_ham.qvr
import qualtran.bloqs.chemistry.trotter.hubbard.hopping
import qualtran.bloqs.chemistry.trotter.hubbard.interaction
import qualtran.bloqs.chemistry.trotter.ising.unitaries
import qualtran.bloqs.chemistry.trotter.trotterized_unitary
import qualtran.bloqs.data_loading.qrom
import qualtran.bloqs.data_loading.select_swap_qrom
import qualtran.bloqs.factoring.mod_exp
import qualtran.bloqs.factoring.mod_mul
import qualtran.bloqs.factoring.mod_sub
import qualtran.bloqs.for_testing.atom
import qualtran.bloqs.for_testing.casting
import qualtran.bloqs.for_testing.interior_alloc
import qualtran.bloqs.for_testing.many_registers
import qualtran.bloqs.for_testing.matrix_gate
import qualtran.bloqs.for_testing.with_call_graph
import qualtran.bloqs.for_testing.with_decomposition
import qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp
import qualtran.bloqs.mcmt.and_bloq
import qualtran.bloqs.mcmt.ctrl_spec_and
import qualtran.bloqs.mcmt.multi_control_multi_target_pauli
import qualtran.bloqs.mean_estimation.arctan
import qualtran.bloqs.mean_estimation.complex_phase_oracle
import qualtran.bloqs.mean_estimation.mean_estimation_operator
import qualtran.bloqs.mod_arithmetic
import qualtran.bloqs.multiplexers.apply_gate_to_lth_target
import qualtran.bloqs.multiplexers.apply_lth_bloq
import qualtran.bloqs.multiplexers.select_pauli_lcu
import qualtran.bloqs.multiplexers.selected_majorana_fermion
import qualtran.bloqs.multiplexers.unary_iteration_bloq
import qualtran.bloqs.phase_estimation.lp_resource_state
import qualtran.bloqs.phase_estimation.qubitization_qpe
import qualtran.bloqs.phase_estimation.text_book_qpe
import qualtran.bloqs.qft.approximate_qft
import qualtran.bloqs.qft.qft_phase_gradient
import qualtran.bloqs.qft.qft_text_book
import qualtran.bloqs.qft.two_bit_ffft
import qualtran.bloqs.qsp.generalized_qsp
import qualtran.bloqs.qubitization.qubitization_walk_operator
import qualtran.bloqs.reflections.prepare_identity
import qualtran.bloqs.reflections.reflection_using_prepare
import qualtran.bloqs.rotations.hamming_weight_phasing
import qualtran.bloqs.rotations.phase_gradient
import qualtran.bloqs.rotations.phasing_via_cost_function
import qualtran.bloqs.rotations.programmable_rotation_gate_array
import qualtran.bloqs.rotations.quantum_variable_rotation
import qualtran.bloqs.state_preparation.prepare_uniform_superposition
import qualtran.bloqs.state_preparation.state_preparation_alias_sampling
import qualtran.bloqs.state_preparation.state_preparation_via_rotation
import qualtran.bloqs.swap_network.cswap_approx
import qualtran.bloqs.swap_network.multiplexed_cswap
import qualtran.bloqs.swap_network.swap_with_zero
from qualtran import Adjoint, Bloq, CompositeBloq, Controlled, CtrlSpec
from qualtran.cirq_interop import CirqGateAsBloq

RESOLVER_DICT = {
    "qualtran._infra.adjoint.Adjoint": Adjoint,
    "qualtran._infra.controlled.CtrlSpec": CtrlSpec,
    "qualtran._infra.controlled.Controlled": Controlled,
    "qualtran._infra.composite_bloq.CompositeBloq": CompositeBloq,
    "qualtran.cirq_interop._cirq_to_bloq.CirqGateAsBloq": CirqGateAsBloq,
    "qualtran.bloqs.arithmetic.addition.Add": qualtran.bloqs.arithmetic.addition.Add,
    "qualtran.bloqs.arithmetic.addition.OutOfPlaceAdder": qualtran.bloqs.arithmetic.addition.OutOfPlaceAdder,
    "qualtran.bloqs.arithmetic.addition.AddK": qualtran.bloqs.arithmetic.AddK,
    "qualtran.bloqs.arithmetic.bitwise.Xor": qualtran.bloqs.arithmetic.bitwise.Xor,
    "qualtran.bloqs.arithmetic.bitwise.XorK": qualtran.bloqs.arithmetic.bitwise.XorK,
    "qualtran.bloqs.arithmetic.comparison.BiQubitsMixer": qualtran.bloqs.arithmetic.comparison.BiQubitsMixer,
    "qualtran.bloqs.arithmetic.comparison.EqualsAConstant": qualtran.bloqs.arithmetic.comparison.EqualsAConstant,
    "qualtran.bloqs.arithmetic.comparison.GreaterThan": qualtran.bloqs.arithmetic.comparison.GreaterThan,
    "qualtran.bloqs.arithmetic.comparison.GreaterThanConstant": qualtran.bloqs.arithmetic.comparison.GreaterThanConstant,
    "qualtran.bloqs.arithmetic.comparison.LessThanConstant": qualtran.bloqs.arithmetic.comparison.LessThanConstant,
    "qualtran.bloqs.arithmetic.comparison.LessThanEqual": qualtran.bloqs.arithmetic.comparison.LessThanEqual,
    "qualtran.bloqs.arithmetic.comparison.LinearDepthGreaterThan": qualtran.bloqs.arithmetic.comparison.LinearDepthGreaterThan,
    "qualtran.bloqs.arithmetic.comparison.SingleQubitCompare": qualtran.bloqs.arithmetic.comparison.SingleQubitCompare,
    "qualtran.bloqs.arithmetic.conversions.SignedIntegerToTwosComplement": qualtran.bloqs.arithmetic.conversions.SignedIntegerToTwosComplement,
    "qualtran.bloqs.arithmetic.conversions.ToContiguousIndex": qualtran.bloqs.arithmetic.conversions.ToContiguousIndex,
    "qualtran.bloqs.arithmetic.hamming_weight.HammingWeightCompute": qualtran.bloqs.arithmetic.hamming_weight.HammingWeightCompute,
    "qualtran.bloqs.arithmetic.multiplication.MultiplyTwoReals": qualtran.bloqs.arithmetic.multiplication.MultiplyTwoReals,
    "qualtran.bloqs.arithmetic.multiplication.PlusEqualProduct": qualtran.bloqs.arithmetic.multiplication.PlusEqualProduct,
    "qualtran.bloqs.arithmetic.multiplication.Product": qualtran.bloqs.arithmetic.multiplication.Product,
    "qualtran.bloqs.arithmetic.multiplication.ScaleIntByReal": qualtran.bloqs.arithmetic.multiplication.ScaleIntByReal,
    "qualtran.bloqs.arithmetic.multiplication.Square": qualtran.bloqs.arithmetic.multiplication.Square,
    "qualtran.bloqs.arithmetic.multiplication.SquareRealNumber": qualtran.bloqs.arithmetic.multiplication.SquareRealNumber,
    "qualtran.bloqs.arithmetic.multiplication.SumOfSquares": qualtran.bloqs.arithmetic.multiplication.SumOfSquares,
    "qualtran.bloqs.arithmetic.negate.Negate": qualtran.bloqs.arithmetic.negate.Negate,
    "qualtran.bloqs.arithmetic.permutation.Permutation": qualtran.bloqs.arithmetic.permutation.Permutation,
    "qualtran.bloqs.arithmetic.permutation.PermutationCycle": qualtran.bloqs.arithmetic.permutation.PermutationCycle,
    "qualtran.bloqs.arithmetic.sorting.BitonicMerge": qualtran.bloqs.arithmetic.sorting.BitonicMerge,
    "qualtran.bloqs.arithmetic.sorting.BitonicSort": qualtran.bloqs.arithmetic.sorting.BitonicSort,
    "qualtran.bloqs.arithmetic.sorting.Comparator": qualtran.bloqs.arithmetic.sorting.Comparator,
    "qualtran.bloqs.arithmetic.sorting.ParallelComparators": qualtran.bloqs.arithmetic.sorting.ParallelComparators,
    "qualtran.bloqs.basic_gates.cnot.CNOT": qualtran.bloqs.basic_gates.cnot.CNOT,
    "qualtran.bloqs.basic_gates.identity.Identity": qualtran.bloqs.basic_gates.identity.Identity,
    "qualtran.bloqs.basic_gates.global_phase.GlobalPhase": qualtran.bloqs.basic_gates.global_phase.GlobalPhase,
    "qualtran.bloqs.basic_gates.hadamard.Hadamard": qualtran.bloqs.basic_gates.hadamard.Hadamard,
    "qualtran.bloqs.basic_gates.on_each.OnEach": qualtran.bloqs.basic_gates.on_each.OnEach,
    "qualtran.bloqs.basic_gates.rotation.CZPowGate": qualtran.bloqs.basic_gates.rotation.CZPowGate,
    "qualtran.bloqs.basic_gates.rotation.Rx": qualtran.bloqs.basic_gates.rotation.Rx,
    "qualtran.bloqs.basic_gates.rotation.Ry": qualtran.bloqs.basic_gates.rotation.Ry,
    "qualtran.bloqs.basic_gates.rotation.Rz": qualtran.bloqs.basic_gates.rotation.Rz,
    "qualtran.bloqs.basic_gates.rotation.XPowGate": qualtran.bloqs.basic_gates.rotation.XPowGate,
    "qualtran.bloqs.basic_gates.rotation.YPowGate": qualtran.bloqs.basic_gates.rotation.YPowGate,
    "qualtran.bloqs.basic_gates.rotation.ZPowGate": qualtran.bloqs.basic_gates.rotation.ZPowGate,
    "qualtran.bloqs.basic_gates.s_gate.SGate": qualtran.bloqs.basic_gates.s_gate.SGate,
    "qualtran.bloqs.basic_gates.su2_rotation.SU2RotationGate": qualtran.bloqs.basic_gates.su2_rotation.SU2RotationGate,
    "qualtran.bloqs.basic_gates.swap.CSwap": qualtran.bloqs.basic_gates.swap.CSwap,
    "qualtran.bloqs.basic_gates.swap.Swap": qualtran.bloqs.basic_gates.swap.Swap,
    "qualtran.bloqs.basic_gates.swap.TwoBitCSwap": qualtran.bloqs.basic_gates.swap.TwoBitCSwap,
    "qualtran.bloqs.basic_gates.swap.TwoBitSwap": qualtran.bloqs.basic_gates.swap.TwoBitSwap,
    "qualtran.bloqs.basic_gates.t_gate.TGate": qualtran.bloqs.basic_gates.t_gate.TGate,
    "qualtran.bloqs.basic_gates.toffoli.Toffoli": qualtran.bloqs.basic_gates.toffoli.Toffoli,
    "qualtran.bloqs.basic_gates.x_basis.MinusEffect": qualtran.bloqs.basic_gates.x_basis.MinusEffect,
    "qualtran.bloqs.basic_gates.x_basis.MinusState": qualtran.bloqs.basic_gates.x_basis.MinusState,
    "qualtran.bloqs.basic_gates.x_basis.PlusEffect": qualtran.bloqs.basic_gates.x_basis.PlusEffect,
    "qualtran.bloqs.basic_gates.x_basis.PlusState": qualtran.bloqs.basic_gates.x_basis.PlusState,
    "qualtran.bloqs.basic_gates.x_basis.XGate": qualtran.bloqs.basic_gates.x_basis.XGate,
    "qualtran.bloqs.basic_gates.y_gate.YGate": qualtran.bloqs.basic_gates.y_gate.YGate,
    "qualtran.bloqs.basic_gates.z_basis.IntEffect": qualtran.bloqs.basic_gates.z_basis.IntEffect,
    "qualtran.bloqs.basic_gates.z_basis.IntState": qualtran.bloqs.basic_gates.z_basis.IntState,
    "qualtran.bloqs.basic_gates.z_basis.OneEffect": qualtran.bloqs.basic_gates.z_basis.OneEffect,
    "qualtran.bloqs.basic_gates.z_basis.OneState": qualtran.bloqs.basic_gates.z_basis.OneState,
    "qualtran.bloqs.basic_gates.z_basis.ZGate": qualtran.bloqs.basic_gates.z_basis.ZGate,
    "qualtran.bloqs.basic_gates.z_basis.ZeroEffect": qualtran.bloqs.basic_gates.z_basis.ZeroEffect,
    "qualtran.bloqs.basic_gates.z_basis.ZeroState": qualtran.bloqs.basic_gates.z_basis.ZeroState,
    "qualtran.bloqs.basic_gates.power.Power": qualtran.bloqs.basic_gates.power.Power,
    "qualtran.bloqs.block_encoding.lcu_block_encoding.LCUBlockEncoding": qualtran.bloqs.block_encoding.lcu_block_encoding.LCUBlockEncoding,
    "qualtran.bloqs.block_encoding.lcu_block_encoding.LCUBlockEncodingZeroState": qualtran.bloqs.block_encoding.lcu_block_encoding.LCUBlockEncodingZeroState,
    "qualtran.bloqs.block_encoding.lcu_block_encoding.BlackBoxPrepare": qualtran.bloqs.block_encoding.lcu_block_encoding.BlackBoxPrepare,
    "qualtran.bloqs.block_encoding.lcu_block_encoding.BlackBoxSelect": qualtran.bloqs.block_encoding.lcu_block_encoding.BlackBoxSelect,
    "qualtran.bloqs.block_encoding.chebyshev_polynomial.ChebyshevPolynomial": qualtran.bloqs.block_encoding.chebyshev_polynomial.ChebyshevPolynomial,
    "qualtran.bloqs.block_encoding.unitary.Unitary": qualtran.bloqs.block_encoding.unitary.Unitary,
    "qualtran.bloqs.block_encoding.tensor_product.TensorProduct": qualtran.bloqs.block_encoding.tensor_product.TensorProduct,
    "qualtran.bloqs.block_encoding.product.Product": qualtran.bloqs.block_encoding.product.Product,
    "qualtran.bloqs.block_encoding.linear_combination.LinearCombination": qualtran.bloqs.block_encoding.linear_combination.LinearCombination,
    "qualtran.bloqs.block_encoding.phase.phase": qualtran.bloqs.block_encoding.phase.Phase,
    "qualtran.bloqs.bookkeeping.allocate.Allocate": qualtran.bloqs.bookkeeping.allocate.Allocate,
    "qualtran.bloqs.bookkeeping.arbitrary_clifford.ArbitraryClifford": qualtran.bloqs.bookkeeping.arbitrary_clifford.ArbitraryClifford,
    "qualtran.bloqs.bookkeeping.auto_partition.AutoPartition": qualtran.bloqs.bookkeeping.auto_partition.AutoPartition,
    "qualtran.bloqs.bookkeeping.cast.Cast": qualtran.bloqs.bookkeeping.cast.Cast,
    "qualtran.bloqs.bookkeeping.free.Free": qualtran.bloqs.bookkeeping.free.Free,
    "qualtran.bloqs.bookkeeping.join.Join": qualtran.bloqs.bookkeeping.join.Join,
    "qualtran.bloqs.bookkeeping.partition.Partition": qualtran.bloqs.bookkeeping.partition.Partition,
    "qualtran.bloqs.bookkeeping.split.Split": qualtran.bloqs.bookkeeping.split.Split,
    "qualtran.bloqs.chemistry.black_boxes.ApplyControlledZs": qualtran.bloqs.chemistry.black_boxes.ApplyControlledZs,
    "qualtran.bloqs.chemistry.black_boxes.QROAM": qualtran.bloqs.chemistry.black_boxes.QROAM,
    "qualtran.bloqs.chemistry.black_boxes.QROAMTwoRegs": qualtran.bloqs.chemistry.black_boxes.QROAMTwoRegs,
    "qualtran.bloqs.chemistry.df.double_factorization.DoubleFactorizationBlockEncoding": qualtran.bloqs.chemistry.df.double_factorization.DoubleFactorizationBlockEncoding,
    "qualtran.bloqs.chemistry.df.double_factorization.DoubleFactorizationOneBody": qualtran.bloqs.chemistry.df.double_factorization.DoubleFactorizationOneBody,
    "qualtran.bloqs.chemistry.df.prepare.InnerPrepareDoubleFactorization": qualtran.bloqs.chemistry.df.prepare.InnerPrepareDoubleFactorization,
    "qualtran.bloqs.chemistry.df.prepare.OuterPrepareDoubleFactorization": qualtran.bloqs.chemistry.df.prepare.OuterPrepareDoubleFactorization,
    "qualtran.bloqs.chemistry.df.prepare.OutputIndexedData": qualtran.bloqs.chemistry.df.prepare.OutputIndexedData,
    "qualtran.bloqs.chemistry.df.select_bloq.ProgRotGateArray": qualtran.bloqs.chemistry.df.select_bloq.ProgRotGateArray,
    "qualtran.bloqs.chemistry.pbc.first_quantization.prepare.UniformSuperpostionIJFirstQuantization": qualtran.bloqs.chemistry.pbc.first_quantization.prepare.UniformSuperpostionIJFirstQuantization,
    "qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu.FlagZeroAsFailure": qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu.FlagZeroAsFailure,
    "qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu.PrepareMuUnaryEncodedOneHot": qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu.PrepareMuUnaryEncodedOneHot,
    "qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu.PrepareNuState": qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu.PrepareNuState,
    "qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu.PrepareNuSuperPositionState": qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu.PrepareNuSuperPositionState,
    "qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu.TestNuInequality": qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu.TestNuInequality,
    "qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu.TestNuLessThanMu": qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu.TestNuLessThanMu,
    "qualtran.bloqs.chemistry.pbc.first_quantization.prepare_t.PreparePowerTwoState": qualtran.bloqs.chemistry.pbc.first_quantization.prepare_t.PreparePowerTwoState,
    "qualtran.bloqs.chemistry.pbc.first_quantization.prepare_t.PrepareTFirstQuantization": qualtran.bloqs.chemistry.pbc.first_quantization.prepare_t.PrepareTFirstQuantization,
    "qualtran.bloqs.chemistry.pbc.first_quantization.prepare_uv.PrepareUVFirstQuantization": qualtran.bloqs.chemistry.pbc.first_quantization.prepare_uv.PrepareUVFirstQuantization,
    "qualtran.bloqs.chemistry.pbc.first_quantization.prepare_zeta.PrepareZetaState": qualtran.bloqs.chemistry.pbc.first_quantization.prepare_zeta.PrepareZetaState,
    "qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_nu.PrepareMuUnaryEncodedOneHotWithProj": qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_nu.PrepareMuUnaryEncodedOneHotWithProj,
    "qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_nu.PrepareNuStateWithProj": qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_nu.PrepareNuStateWithProj,
    "qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_t.PreparePowerTwoStateWithProj": qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_t.PreparePowerTwoStateWithProj,
    "qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_t.PrepareTFirstQuantizationWithProj": qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_t.PrepareTFirstQuantizationWithProj,
    "qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_uv.PrepareUVFirstQuantizationWithProj": qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_uv.PrepareUVFirstQuantizationWithProj,
    "qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_and_prepare.ControlledMultiplexedCSwap3D": qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_and_prepare.ControlledMultiplexedCSwap3D,
    "qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_and_prepare.PrepareFirstQuantizationWithProj": qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_and_prepare.PrepareFirstQuantizationWithProj,
    "qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_and_prepare.PrepareTUVSuperpositions": qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_and_prepare.PrepareTUVSuperpositions,
    "qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_and_prepare.SelectFirstQuantizationWithProj": qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_and_prepare.SelectFirstQuantizationWithProj,
    "qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_t.SelectTFirstQuantizationWithProj": qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_t.SelectTFirstQuantizationWithProj,
    "qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_uv.SelectUVFirstQuantizationWithProj": qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_uv.SelectUVFirstQuantizationWithProj,
    "qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare.MultiplexedCSwap3D": qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare.MultiplexedCSwap3D,
    "qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare.PrepareFirstQuantization": qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare.PrepareFirstQuantization,
    "qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare.PrepareTUVSuperpositions": qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare.PrepareTUVSuperpositions,
    "qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare.SelectFirstQuantization": qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare.SelectFirstQuantization,
    "qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare.UniformSuperpostionIJFirstQuantization": qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare.UniformSuperpostionIJFirstQuantization,
    "qualtran.bloqs.chemistry.pbc.first_quantization.select_t.SelectTFirstQuantization": qualtran.bloqs.chemistry.pbc.first_quantization.select_t.SelectTFirstQuantization,
    "qualtran.bloqs.chemistry.pbc.first_quantization.select_uv.ApplyNuclearPhase": qualtran.bloqs.chemistry.pbc.first_quantization.select_uv.ApplyNuclearPhase,
    "qualtran.bloqs.chemistry.pbc.first_quantization.select_uv.SelectUVFirstQuantization": qualtran.bloqs.chemistry.pbc.first_quantization.select_uv.SelectUVFirstQuantization,
    "qualtran.bloqs.chemistry.sf.prepare.InnerPrepareSingleFactorization": qualtran.bloqs.chemistry.sf.prepare.InnerPrepareSingleFactorization,
    "qualtran.bloqs.chemistry.sf.prepare.OuterPrepareSingleFactorization": qualtran.bloqs.chemistry.sf.prepare.OuterPrepareSingleFactorization,
    "qualtran.bloqs.chemistry.sf.select_bloq.SelectSingleFactorization": qualtran.bloqs.chemistry.sf.select_bloq.SelectSingleFactorization,
    "qualtran.bloqs.chemistry.sf.single_factorization.SingleFactorizationBlockEncoding": qualtran.bloqs.chemistry.sf.single_factorization.SingleFactorizationBlockEncoding,
    "qualtran.bloqs.chemistry.sf.single_factorization.SingleFactorizationOneBody": qualtran.bloqs.chemistry.sf.single_factorization.SingleFactorizationOneBody,
    "qualtran.bloqs.chemistry.sparse.prepare.PrepareSparse": qualtran.bloqs.chemistry.sparse.prepare.PrepareSparse,
    "qualtran.bloqs.chemistry.sparse.select_bloq.SelectSparse": qualtran.bloqs.chemistry.sparse.select_bloq.SelectSparse,
    "qualtran.bloqs.chemistry.thc.prepare.PrepareTHC": qualtran.bloqs.chemistry.thc.prepare.PrepareTHC,
    "qualtran.bloqs.chemistry.thc.prepare.UniformSuperpositionTHC": qualtran.bloqs.chemistry.thc.prepare.UniformSuperpositionTHC,
    "qualtran.bloqs.chemistry.thc.select_bloq.SelectTHC": qualtran.bloqs.chemistry.thc.select_bloq.SelectTHC,
    "qualtran.bloqs.chemistry.thc.select_bloq.THCRotations": qualtran.bloqs.chemistry.thc.select_bloq.THCRotations,
    "qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt.NewtonRaphsonApproxInverseSquareRoot": qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt.NewtonRaphsonApproxInverseSquareRoot,
    "qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt.PolynmomialEvaluationInverseSquareRoot": qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt.PolynmomialEvaluationInverseSquareRoot,
    "qualtran.bloqs.chemistry.trotter.grid_ham.kinetic.KineticEnergy": qualtran.bloqs.chemistry.trotter.grid_ham.kinetic.KineticEnergy,
    "qualtran.bloqs.chemistry.trotter.grid_ham.potential.PairPotential": qualtran.bloqs.chemistry.trotter.grid_ham.potential.PairPotential,
    "qualtran.bloqs.chemistry.trotter.grid_ham.potential.PotentialEnergy": qualtran.bloqs.chemistry.trotter.grid_ham.potential.PotentialEnergy,
    "qualtran.bloqs.chemistry.trotter.grid_ham.qvr.QuantumVariableRotation": qualtran.bloqs.chemistry.trotter.grid_ham.qvr.QuantumVariableRotation,
    "qualtran.bloqs.chemistry.trotter.ising.unitaries.IsingXUnitary": qualtran.bloqs.chemistry.trotter.ising.unitaries.IsingXUnitary,
    "qualtran.bloqs.chemistry.trotter.ising.unitaries.IsingZZUnitary": qualtran.bloqs.chemistry.trotter.ising.unitaries.IsingZZUnitary,
    "qualtran.bloqs.chemistry.trotter.hubbard.interaction.Interaction": qualtran.bloqs.chemistry.trotter.hubbard.interaction.Interaction,
    "qualtran.bloqs.chemistry.trotter.hubbard.interaction.InteractionHWP": qualtran.bloqs.chemistry.trotter.hubbard.interaction.InteractionHWP,
    "qualtran.bloqs.chemistry.trotter.hubbard.hopping.HoppingPlaquette": qualtran.bloqs.chemistry.trotter.hubbard.hopping.HoppingPlaquette,
    "qualtran.bloqs.chemistry.trotter.hubbard.hopping.HoppingTile": qualtran.bloqs.chemistry.trotter.hubbard.hopping.HoppingTile,
    "qualtran.bloqs.chemistry.trotter.hubbard.hopping.HoppingTileHWP": qualtran.bloqs.chemistry.trotter.hubbard.hopping.HoppingTileHWP,
    "qualtran.bloqs.chemistry.trotter.trotterized_unitary": qualtran.bloqs.chemistry.trotter.trotterized_unitary,
    "qualtran.bloqs.data_loading.qrom.QROM": qualtran.bloqs.data_loading.qrom.QROM,
    "qualtran.bloqs.data_loading.select_swap_qrom.SelectSwapQROM": qualtran.bloqs.data_loading.select_swap_qrom.SelectSwapQROM,
    "qualtran.bloqs.mod_arithmetic.CModAddK": qualtran.bloqs.mod_arithmetic.CModAddK,
    "qualtran.bloqs.mod_arithmetic.mod_addition.ModAddK": qualtran.bloqs.mod_arithmetic.mod_addition.ModAddK,
    "qualtran.bloqs.mod_arithmetic.mod_addition.CtrlScaleModAdd": qualtran.bloqs.mod_arithmetic.CtrlScaleModAdd,
    "qualtran.bloqs.mod_arithmetic.ModAdd": qualtran.bloqs.mod_arithmetic.ModAdd,
    "qualtran.bloqs.factoring.mod_exp.ModExp": qualtran.bloqs.factoring.mod_exp.ModExp,
    "qualtran.bloqs.factoring.mod_mul.CtrlModMul": qualtran.bloqs.factoring.mod_mul.CtrlModMul,
    "qualtran.bloqs.factoring.mod_mul.MontgomeryModDbl": qualtran.bloqs.factoring.mod_mul.MontgomeryModDbl,
    "qualtran.bloqs.factoring.mod_sub.MontgomeryModNeg": qualtran.bloqs.factoring.mod_sub.MontgomeryModNeg,
    "qualtran.bloqs.factoring.mod_sub.MontgomeryModSub": qualtran.bloqs.factoring.mod_sub.MontgomeryModSub,
    "qualtran.bloqs.for_testing.atom.TestAtom": qualtran.bloqs.for_testing.atom.TestAtom,
    "qualtran.bloqs.for_testing.atom.TestGWRAtom": qualtran.bloqs.for_testing.atom.TestGWRAtom,
    "qualtran.bloqs.for_testing.atom.TestTwoBitOp": qualtran.bloqs.for_testing.atom.TestTwoBitOp,
    "qualtran.bloqs.for_testing.casting.TestCastToFrom": qualtran.bloqs.for_testing.casting.TestCastToFrom,
    "qualtran.bloqs.for_testing.interior_alloc.InteriorAlloc": qualtran.bloqs.for_testing.interior_alloc.InteriorAlloc,
    "qualtran.bloqs.for_testing.many_registers.TestBoundedQUInt": qualtran.bloqs.for_testing.many_registers.TestBoundedQUInt,
    "qualtran.bloqs.for_testing.many_registers.TestMultiRegister": qualtran.bloqs.for_testing.many_registers.TestMultiRegister,
    "qualtran.bloqs.for_testing.many_registers.TestMultiTypedRegister": qualtran.bloqs.for_testing.many_registers.TestMultiTypedRegister,
    "qualtran.bloqs.for_testing.many_registers.TestQFxp": qualtran.bloqs.for_testing.many_registers.TestQFxp,
    "qualtran.bloqs.for_testing.matrix_gate.MatrixGate": qualtran.bloqs.for_testing.matrix_gate.MatrixGate,
    "qualtran.bloqs.for_testing.with_call_graph.TestBloqWithCallGraph": qualtran.bloqs.for_testing.with_call_graph.TestBloqWithCallGraph,
    "qualtran.bloqs.for_testing.with_decomposition.TestIndependentParallelCombo": qualtran.bloqs.for_testing.with_decomposition.TestIndependentParallelCombo,
    "qualtran.bloqs.for_testing.with_decomposition.TestParallelCombo": qualtran.bloqs.for_testing.with_decomposition.TestParallelCombo,
    "qualtran.bloqs.for_testing.with_decomposition.TestSerialCombo": qualtran.bloqs.for_testing.with_decomposition.TestSerialCombo,
    "qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp.HamiltonianSimulationByGQSP": qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp.HamiltonianSimulationByGQSP,
    "qualtran.bloqs.chemistry.hubbard_model.qubitization.prepare_hubbard.PrepareHubbard": qualtran.bloqs.chemistry.hubbard_model.qubitization.prepare_hubbard.PrepareHubbard,
    "qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard.SelectHubbard": qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard.SelectHubbard,
    "qualtran.bloqs.mcmt.and_bloq.And": qualtran.bloqs.mcmt.and_bloq.And,
    "qualtran.bloqs.mcmt.and_bloq.MultiAnd": qualtran.bloqs.mcmt.and_bloq.MultiAnd,
    "qualtran.bloqs.mcmt.ctrl_spec_and.CtrlSpecAnd": qualtran.bloqs.mcmt.ctrl_spec_and.CtrlSpecAnd,
    "qualtran.bloqs.mcmt.multi_control_multi_target_pauli.MultiControlPauli": qualtran.bloqs.mcmt.multi_control_multi_target_pauli.MultiControlPauli,
    "qualtran.bloqs.mcmt.multi_control_multi_target_pauli.MultiControlX": qualtran.bloqs.mcmt.multi_control_multi_target_pauli.MultiControlX,
    "qualtran.bloqs.mcmt.multi_control_multi_target_pauli.MultiTargetCNOT": qualtran.bloqs.mcmt.multi_control_multi_target_pauli.MultiTargetCNOT,
    "qualtran.bloqs.mean_estimation.arctan.ArcTan": qualtran.bloqs.mean_estimation.arctan.ArcTan,
    "qualtran.bloqs.mean_estimation.complex_phase_oracle.ComplexPhaseOracle": qualtran.bloqs.mean_estimation.complex_phase_oracle.ComplexPhaseOracle,
    "qualtran.bloqs.mean_estimation.mean_estimation_operator.MeanEstimationOperator": qualtran.bloqs.mean_estimation.mean_estimation_operator.MeanEstimationOperator,
    "qualtran.bloqs.multiplexers.apply_gate_to_lth_target.ApplyGateToLthQubit": qualtran.bloqs.multiplexers.apply_gate_to_lth_target.ApplyGateToLthQubit,
    "qualtran.bloqs.multiplexers.apply_lth_bloq.ApplyLthBloq": qualtran.bloqs.multiplexers.apply_lth_bloq.ApplyLthBloq,
    "qualtran.bloqs.multiplexers.select_pauli_lcu.SelectPauliLCU": qualtran.bloqs.multiplexers.select_pauli_lcu.SelectPauliLCU,
    "qualtran.bloqs.multiplexers.selected_majorana_fermion.SelectedMajoranaFermion": qualtran.bloqs.multiplexers.selected_majorana_fermion.SelectedMajoranaFermion,
    "qualtran.bloqs.multiplexers.unary_iteration_bloq.UnaryIterationGate": qualtran.bloqs.multiplexers.unary_iteration_bloq.UnaryIterationGate,
    "qualtran.bloqs.phase_estimation.lp_resource_state.LPRSInterimPrep": qualtran.bloqs.phase_estimation.lp_resource_state.LPRSInterimPrep,
    "qualtran.bloqs.phase_estimation.lp_resource_state.LPResourceState": qualtran.bloqs.phase_estimation.lp_resource_state.LPResourceState,
    "qualtran.bloqs.phase_estimation.qubitization_qpe.QubitizationQPE": qualtran.bloqs.phase_estimation.qubitization_qpe.QubitizationQPE,
    "qualtran.bloqs.phase_estimation.text_book_qpe.TextbookQPE": qualtran.bloqs.phase_estimation.text_book_qpe.TextbookQPE,
    "qualtran.bloqs.qft.approximate_qft.ApproximateQFT": qualtran.bloqs.qft.approximate_qft.ApproximateQFT,
    "qualtran.bloqs.qft.qft_phase_gradient.QFTPhaseGradient": qualtran.bloqs.qft.qft_phase_gradient.QFTPhaseGradient,
    "qualtran.bloqs.qft.qft_text_book.QFTTextBook": qualtran.bloqs.qft.qft_text_book.QFTTextBook,
    "qualtran.bloqs.qft.two_bit_ffft.TwoBitFFFT": qualtran.bloqs.qft.two_bit_ffft.TwoBitFFFT,
    "qualtran.bloqs.qsp.generalized_qsp.GeneralizedQSP": qualtran.bloqs.qsp.generalized_qsp.GeneralizedQSP,
    "qualtran.bloqs.qubitization.qubitization_walk_operator.QubitizationWalkOperator": qualtran.bloqs.qubitization.qubitization_walk_operator.QubitizationWalkOperator,
    "qualtran.bloqs.reflections.prepare_identity.PrepareIdentity": qualtran.bloqs.reflections.prepare_identity.PrepareIdentity,
    "qualtran.bloqs.reflections.reflection_using_prepare.ReflectionUsingPrepare": qualtran.bloqs.reflections.reflection_using_prepare.ReflectionUsingPrepare,
    "qualtran.bloqs.rotations.hamming_weight_phasing.HammingWeightPhasing": qualtran.bloqs.rotations.hamming_weight_phasing.HammingWeightPhasing,
    "qualtran.bloqs.rotations.hamming_weight_phasing.HammingWeightPhasingViaPhaseGradient": qualtran.bloqs.rotations.hamming_weight_phasing.HammingWeightPhasingViaPhaseGradient,
    "qualtran.bloqs.rotations.phase_gradient.AddIntoPhaseGrad": qualtran.bloqs.rotations.phase_gradient.AddIntoPhaseGrad,
    "qualtran.bloqs.rotations.phase_gradient.AddScaledValIntoPhaseReg": qualtran.bloqs.rotations.phase_gradient.AddScaledValIntoPhaseReg,
    "qualtran.bloqs.rotations.phase_gradient.PhaseGradientState": qualtran.bloqs.rotations.phase_gradient.PhaseGradientState,
    "qualtran.bloqs.rotations.phase_gradient.PhaseGradientUnitary": qualtran.bloqs.rotations.phase_gradient.PhaseGradientUnitary,
    "qualtran.bloqs.rotations.phasing_via_cost_function.PhasingViaCostFunction": qualtran.bloqs.rotations.phasing_via_cost_function.PhasingViaCostFunction,
    "qualtran.bloqs.rotations.programmable_rotation_gate_array.ProgrammableRotationGateArray": qualtran.bloqs.rotations.programmable_rotation_gate_array.ProgrammableRotationGateArray,
    "qualtran.bloqs.rotations.programmable_rotation_gate_array.ProgrammableRotationGateArrayBase": qualtran.bloqs.rotations.programmable_rotation_gate_array.ProgrammableRotationGateArrayBase,
    "qualtran.bloqs.rotations.quantum_variable_rotation.QvrInterface": qualtran.bloqs.rotations.quantum_variable_rotation.QvrInterface,
    "qualtran.bloqs.rotations.quantum_variable_rotation.QvrPhaseGradient": qualtran.bloqs.rotations.quantum_variable_rotation.QvrPhaseGradient,
    "qualtran.bloqs.rotations.quantum_variable_rotation.QvrZPow": qualtran.bloqs.rotations.quantum_variable_rotation.QvrZPow,
    "qualtran.bloqs.block_encoding.lcu_select_and_prepare.PrepareOracle": qualtran.bloqs.block_encoding.lcu_select_and_prepare.PrepareOracle,
    "qualtran.bloqs.block_encoding.lcu_select_and_prepare.SelectOracle": qualtran.bloqs.block_encoding.lcu_select_and_prepare.SelectOracle,
    "qualtran.bloqs.state_preparation.prepare_uniform_superposition.PrepareUniformSuperposition": qualtran.bloqs.state_preparation.prepare_uniform_superposition.PrepareUniformSuperposition,
    "qualtran.bloqs.state_preparation.state_preparation_alias_sampling.StatePreparationAliasSampling": qualtran.bloqs.state_preparation.state_preparation_alias_sampling.StatePreparationAliasSampling,
    "qualtran.bloqs.state_preparation.state_preparation_alias_sampling.SparseStatePreparationAliasSampling": qualtran.bloqs.state_preparation.state_preparation_alias_sampling.SparseStatePreparationAliasSampling,
    "qualtran.bloqs.state_preparation.state_preparation_via_rotation.PRGAViaPhaseGradient": qualtran.bloqs.state_preparation.state_preparation_via_rotation.PRGAViaPhaseGradient,
    "qualtran.bloqs.state_preparation.state_preparation_via_rotation.StatePreparationViaRotations": qualtran.bloqs.state_preparation.state_preparation_via_rotation.StatePreparationViaRotations,
    "qualtran.bloqs.swap_network.cswap_approx.CSwapApprox": qualtran.bloqs.swap_network.cswap_approx.CSwapApprox,
    "qualtran.bloqs.swap_network.multiplexed_cswap.MultiplexedCSwap": qualtran.bloqs.swap_network.multiplexed_cswap.MultiplexedCSwap,
    "qualtran.bloqs.swap_network.swap_with_zero.SwapWithZero": qualtran.bloqs.swap_network.swap_with_zero.SwapWithZero,
}


def add_to_resolver_dict(*bloqs: Type[Bloq]):
    """Adds given Bloqs to the resolver dict using fully qualified Bloq names as keys."""
    RESOLVER_DICT.update({bloq.__module__ + '.' + bloq.__qualname__: bloq for bloq in bloqs})
