# Module: testing






## Classes

[`class BloqCheckResult`](../qualtran/testing/BloqCheckResult.md): The status result of the `check_bloq_example_xxx` functions.

[`class BloqCheckException`](../qualtran/testing/BloqCheckException.md): An exception raised by the `assert_bloq_example_xxx` functions in this module.

## Functions

[`assert_bloq_example_decompose(...)`](../qualtran/testing/assert_bloq_example_decompose.md): Assert that the BloqExample has a valid decomposition.

[`assert_bloq_example_make(...)`](../qualtran/testing/assert_bloq_example_make.md): Assert that the BloqExample returns the desired bloq.

[`assert_connections_compatible(...)`](../qualtran/testing/assert_connections_compatible.md): Check that all connections are between compatible registers.

[`assert_registers_match_dangling(...)`](../qualtran/testing/assert_registers_match_dangling.md): Check that connections to LeftDangle and RightDangle match the declared registers.

[`assert_registers_match_parent(...)`](../qualtran/testing/assert_registers_match_parent.md): Check that the registers following decomposition match those of the original bloq.

[`assert_soquets_belong_to_registers(...)`](../qualtran/testing/assert_soquets_belong_to_registers.md): Check that all soquet's registers make sense.

[`assert_soquets_used_exactly_once(...)`](../qualtran/testing/assert_soquets_used_exactly_once.md): Check that all soquets are used once and only once.

[`assert_valid_bloq_decomposition(...)`](../qualtran/testing/assert_valid_bloq_decomposition.md): Check the validity of a bloq decomposition.

[`assert_valid_cbloq(...)`](../qualtran/testing/assert_valid_cbloq.md): Perform all composite-bloq validity assertions.

[`assert_wire_symbols_match_expected(...)`](../qualtran/testing/assert_wire_symbols_match_expected.md): Assert a bloq's wire symbols match the expected ones.

[`check_bloq_example_decompose(...)`](../qualtran/testing/check_bloq_example_decompose.md): Check that the BloqExample has a valid decomposition.

[`check_bloq_example_make(...)`](../qualtran/testing/check_bloq_example_make.md): Check that the BloqExample returns the desired bloq.

[`execute_notebook(...)`](../qualtran/testing/execute_notebook.md): Execute a jupyter notebook in the caller's directory.



<h2 class="add-link">Other Members</h2>

LeftDangle<a id="LeftDangle"></a>
: Instance of <a href="../qualtran/DanglingT.html"><code>qualtran.DanglingT</code></a>

RightDangle<a id="RightDangle"></a>
: Instance of <a href="../qualtran/DanglingT.html"><code>qualtran.DanglingT</code></a>


