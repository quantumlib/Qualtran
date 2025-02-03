# Module: surface_code


Physical cost models for surface code architectures.



## Modules

[`algorithm_summary`](../qualtran/surface_code/algorithm_summary.md) module

[`ccz2t_factory`](../qualtran/surface_code/ccz2t_factory.md) module

[`data_block`](../qualtran/surface_code/data_block.md) module

[`fifteen_to_one_factory`](../qualtran/surface_code/fifteen_to_one_factory.md) module

[`gidney_fowler_model`](../qualtran/surface_code/gidney_fowler_model.md) module

[`magic_state_factory`](../qualtran/surface_code/magic_state_factory.md) module

[`multi_factory`](../qualtran/surface_code/multi_factory.md) module

[`physical_cost_model`](../qualtran/surface_code/physical_cost_model.md) module

[`physical_cost_summary`](../qualtran/surface_code/physical_cost_summary.md) module

[`physical_parameters`](../qualtran/surface_code/physical_parameters.md) module

[`qec_scheme`](../qualtran/surface_code/qec_scheme.md) module

[`rotation_cost_model`](../qualtran/surface_code/rotation_cost_model.md) module

[`t_factory_utils`](../qualtran/surface_code/t_factory_utils.md) module

## Classes

[`class AlgorithmSummary`](../qualtran/surface_code/AlgorithmSummary.md): Logical costs of a quantum algorithm that impact modeling of its physical cost.

[`class PhysicalCostsSummary`](../qualtran/surface_code/PhysicalCostsSummary.md)

[`class PhysicalParameters`](../qualtran/surface_code/PhysicalParameters.md): The physical properties of a quantum computer.

[`class LogicalErrorModel`](../qualtran/surface_code/LogicalErrorModel.md): A model for getting the logical error rate at a given code distance.

[`class QECScheme`](../qualtran/surface_code/QECScheme.md): A model of the error-correction scheme used to suppress errors

[`class MagicStateFactory`](../qualtran/surface_code/MagicStateFactory.md): Methods for modeling the costs of the magic state factories of a surface code compilation.

[`class CCZ2TFactory`](../qualtran/surface_code/CCZ2TFactory.md): Magic state factory costs using the model from catalyzed CCZ to 2T paper.

[`class FifteenToOne`](../qualtran/surface_code/FifteenToOne.md): 15-to-1 Magic T state factory.

[`class CompactDataBlock`](../qualtran/surface_code/CompactDataBlock.md): The compact data block uses a fixed code distance and one, long access hallway.

[`class DataBlock`](../qualtran/surface_code/DataBlock.md): Methods for modeling the costs of the data block of a surface code compilation.

[`class FastDataBlock`](../qualtran/surface_code/FastDataBlock.md): The fast data block uses a fixed code distance and a square layout.

[`class IntermediateDataBlock`](../qualtran/surface_code/IntermediateDataBlock.md): The intermediate data block uses a fixed code distance and routing overhead.

[`class SimpleDataBlock`](../qualtran/surface_code/SimpleDataBlock.md): A simple data block that uses a fixed code distance and routing overhead.

[`class MultiFactory`](../qualtran/surface_code/MultiFactory.md): Overlay of MagicStateFactory representing multiple factories of the same kind.

[`class ConstantWithOverheadRotationCost`](../qualtran/surface_code/ConstantWithOverheadRotationCost.md): A rotation cost of bitsize - 2 toffoli per rotation independent of the error budget.

[`class RotationCostModel`](../qualtran/surface_code/RotationCostModel.md): Analytical estimate of the complexity of approximating a rotation given an error budget.

[`class RotationLogarithmicModel`](../qualtran/surface_code/RotationLogarithmicModel.md): A linear model in the log of the error budget with no preparation cost.

[`class PhysicalCostModel`](../qualtran/surface_code/PhysicalCostModel.md): A model for estimating physical costs from algorithm counts.

## Functions

[`get_ccz2t_costs(...)`](../qualtran/surface_code/get_ccz2t_costs.md): Generate spacetime cost and failure probability given physical and logical parameters.

[`get_ccz2t_costs_from_error_budget(...)`](../qualtran/surface_code/get_ccz2t_costs_from_error_budget.md): Physical costs using the model from catalyzed CCZ to 2T paper.

[`get_ccz2t_costs_from_grid_search(...)`](../qualtran/surface_code/get_ccz2t_costs_from_grid_search.md): Grid search over parameters to minimize the space-time volume.

[`iter_ccz2t_factories(...)`](../qualtran/surface_code/iter_ccz2t_factories.md): Iterate over CCZ2T (multi)factories in the given range of distillation code distances

[`iter_simple_data_blocks(...)`](../qualtran/surface_code/iter_simple_data_blocks.md)



<h2 class="add-link">Other Members</h2>

BeverlandEtAlRotationCost<a id="BeverlandEtAlRotationCost"></a>
: Instance of <a href="../qualtran/surface_code/RotationLogarithmicModel.html"><code>qualtran.surface_code.RotationLogarithmicModel</code></a>

SevenDigitsOfPrecisionConstantCost<a id="SevenDigitsOfPrecisionConstantCost"></a>
: Instance of <a href="../qualtran/surface_code/ConstantWithOverheadRotationCost.html"><code>qualtran.surface_code.ConstantWithOverheadRotationCost</code></a>


