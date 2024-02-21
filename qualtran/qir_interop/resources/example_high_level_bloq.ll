; ModuleID = 'ExampleHighLevelBloq'
source_filename = "ExampleHighLevelBloq"

%Qubit = type opaque

define void @main() #0 {
entry:
  call void @ExampleHighLevelBloq_10(%Qubit* null, %Qubit* inttoptr (i64 1 to %Qubit*), %Qubit* inttoptr (i64 2 to %Qubit*), %Qubit* inttoptr (i64 3 to %Qubit*), %Qubit* inttoptr (i64 4 to %Qubit*), %Qubit* inttoptr (i64 5 to %Qubit*), %Qubit* inttoptr (i64 6 to %Qubit*), %Qubit* inttoptr (i64 7 to %Qubit*), %Qubit* inttoptr (i64 8 to %Qubit*), %Qubit* inttoptr (i64 9 to %Qubit*))
  ret void
}

declare %Qubit* @__quantum__rt__qubit_allocate()

define void @ExampleHighLevelBloq_10(%Qubit* %0, %Qubit* %1, %Qubit* %2, %Qubit* %3, %Qubit* %4, %Qubit* %5, %Qubit* %6, %Qubit* %7, %Qubit* %8, %Qubit* %9) {
block1:
  call void @ExampleBaseBloq_2(%Qubit* %0, %Qubit* %5)
  call void @ExampleBaseBloq_2(%Qubit* %1, %Qubit* %6)
  call void @ExampleBaseBloq_2(%Qubit* %2, %Qubit* %7)
  call void @ExampleBaseBloq_2(%Qubit* %3, %Qubit* %8)
  call void @ExampleBaseBloq_2(%Qubit* %4, %Qubit* %9)
  ret void
}

declare void @ExampleBaseBloq_2(%Qubit*, %Qubit*)

attributes #0 = { "entry_point" "output_labeling_schema" "qir_profiles"="custom" "required_num_qubits"="10" "required_num_results"="0" }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"qir_major_version", i32 1}
!1 = !{i32 7, !"qir_minor_version", i32 0}
!2 = !{i32 1, !"dynamic_qubit_management", i1 true}
!3 = !{i32 1, !"dynamic_result_management", i1 false}
