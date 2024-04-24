# Module: surface_code






## Modules

[`algorithm_summary`](../qualtran/surface_code/algorithm_summary.md) module

[`ccz2t_cost_model`](../qualtran/surface_code/ccz2t_cost_model.md) module

[`data_block`](../qualtran/surface_code/data_block.md) module

[`magic_count`](../qualtran/surface_code/magic_count.md) module

[`magic_state_factory`](../qualtran/surface_code/magic_state_factory.md) module

[`multi_factory`](../qualtran/surface_code/multi_factory.md) module

[`physical_cost`](../qualtran/surface_code/physical_cost.md) module

[`physical_parameters`](../qualtran/surface_code/physical_parameters.md) module

[`quantum_error_correction_scheme_summary`](../qualtran/surface_code/quantum_error_correction_scheme_summary.md) module

[`reference`](../qualtran/surface_code/reference.md) module

[`rotation_cost_model`](../qualtran/surface_code/rotation_cost_model.md) module

## Classes

[`class AlgorithmSummary`](../qualtran/surface_code/AlgorithmSummary.md): Properties of a quantum algorithm that impact its physical cost

[`class CCZ2TFactory`](../qualtran/surface_code/CCZ2TFactory.md): Magic state factory costs using the model from catalyzed CCZ to 2T paper.

[`class CompactDataBlock`](../qualtran/surface_code/CompactDataBlock.md): The compact data block uses a fixed code distance and routing overhead.

[`class FastDataBlock`](../qualtran/surface_code/FastDataBlock.md): The fast data block uses a fixed code distance and a square layout.

[`class IntermediateDataBlock`](../qualtran/surface_code/IntermediateDataBlock.md): The intermediate data block uses a fixed code distance and routing overhead.

[`class SimpleDataBlock`](../qualtran/surface_code/SimpleDataBlock.md): A simple data block that uses a fixed code distance and routing overhead.

[`class MagicCount`](../qualtran/surface_code/MagicCount.md): A count of magic states.

[`class MagicStateFactory`](../qualtran/surface_code/MagicStateFactory.md): A cost model for the magic state distillation factory of a surface code compilation.

[`class MultiFactory`](../qualtran/surface_code/MultiFactory.md): Overlay of MagicStateFactory representing multiple factories of the same kind.

[`class PhysicalCost`](../qualtran/surface_code/PhysicalCost.md)

[`class PhysicalParameters`](../qualtran/surface_code/PhysicalParameters.md): The physical properties of a quantum computer.

[`class QuantumErrorCorrectionSchemeSummary`](../qualtran/surface_code/QuantumErrorCorrectionSchemeSummary.md): QuantumErrorCorrectionSchemeSummary represents a high-level view of a QEC scheme.

[`class Reference`](../qualtran/surface_code/Reference.md): A reference to a source material.

[`class ConstantWithOverheadRotationCost`](../qualtran/surface_code/ConstantWithOverheadRotationCost.md): A rotation cost of bitsize - 2 toffoli per rotation independent of the error budget.

[`class RotationCostModel`](../qualtran/surface_code/RotationCostModel.md): Analytical estimate of the complexity of approximating a rotation given an error budget.

[`class RotationLogarithmicModel`](../qualtran/surface_code/RotationLogarithmicModel.md): A linear model in the log of the error budget with no preparation cost.

## Functions

[`get_ccz2t_costs(...)`](../qualtran/surface_code/get_ccz2t_costs.md): Generate spacetime cost and failure probability given physical and logical parameters.

[`get_ccz2t_costs_from_error_budget(...)`](../qualtran/surface_code/get_ccz2t_costs_from_error_budget.md): Physical costs using the model from catalyzed CCZ to 2T paper.

[`get_ccz2t_costs_from_grid_search(...)`](../qualtran/surface_code/get_ccz2t_costs_from_grid_search.md): Grid search over parameters to minimize space time volume.



<h2 class="add-link">Other Members</h2>

BeverlandEtAlRotationCost<a id="BeverlandEtAlRotationCost"></a>
: Instance of <a href="../qualtran/surface_code/RotationLogarithmicModel.html"><code>qualtran.surface_code.RotationLogarithmicModel</code></a>

BeverlandMajoranaQubits<a id="BeverlandMajoranaQubits"></a>
: Instance of <a href="../qualtran/surface_code/quantum_error_correction_scheme_summary/SimpliedSurfaceCode.html"><code>qualtran.surface_code.quantum_error_correction_scheme_summary.SimpliedSurfaceCode</code></a>

BeverlandSuperconductingQubits<a id="BeverlandSuperconductingQubits"></a>
: Instance of <a href="../qualtran/surface_code/quantum_error_correction_scheme_summary/SimpliedSurfaceCode.html"><code>qualtran.surface_code.quantum_error_correction_scheme_summary.SimpliedSurfaceCode</code></a>

BeverlandTrappedIonQubits<a id="BeverlandTrappedIonQubits"></a>
: Instance of <a href="../qualtran/surface_code/quantum_error_correction_scheme_summary/SimpliedSurfaceCode.html"><code>qualtran.surface_code.quantum_error_correction_scheme_summary.SimpliedSurfaceCode</code></a>

FowlerSuperconductingQubits<a id="FowlerSuperconductingQubits"></a>
: Instance of <a href="../qualtran/surface_code/quantum_error_correction_scheme_summary/SimpliedSurfaceCode.html"><code>qualtran.surface_code.quantum_error_correction_scheme_summary.SimpliedSurfaceCode</code></a>

SevenDigitsOfPrecisionConstantCost<a id="SevenDigitsOfPrecisionConstantCost"></a>
: Instance of <a href="../qualtran/surface_code/ConstantWithOverheadRotationCost.html"><code>qualtran.surface_code.ConstantWithOverheadRotationCost</code></a>


