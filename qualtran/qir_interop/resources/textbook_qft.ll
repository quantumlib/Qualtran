; ModuleID = 'QFTTextBook'
source_filename = "QFTTextBook"

%Qubit = type opaque

define void @main() #0 {
entry:
  call void @QFTTextBook_10(%Qubit* null, %Qubit* inttoptr (i64 1 to %Qubit*), %Qubit* inttoptr (i64 2 to %Qubit*), %Qubit* inttoptr (i64 3 to %Qubit*), %Qubit* inttoptr (i64 4 to %Qubit*), %Qubit* inttoptr (i64 5 to %Qubit*), %Qubit* inttoptr (i64 6 to %Qubit*), %Qubit* inttoptr (i64 7 to %Qubit*), %Qubit* inttoptr (i64 8 to %Qubit*), %Qubit* inttoptr (i64 9 to %Qubit*))
  ret void
}

declare %Qubit* @__quantum__rt__qubit_allocate()

define void @QFTTextBook_10(%Qubit* %0, %Qubit* %1, %Qubit* %2, %Qubit* %3, %Qubit* %4, %Qubit* %5, %Qubit* %6, %Qubit* %7, %Qubit* %8, %Qubit* %9) {
block1:
  call void @__quantum__qis__h__body(%Qubit* %0)
  call void @PhaseGradientUnitary_2(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__h__body(%Qubit* %1)
  call void @PhaseGradientUnitary_3(%Qubit* %2, %Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__h__body(%Qubit* %2)
  call void @PhaseGradientUnitary_4(%Qubit* %3, %Qubit* %2, %Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__h__body(%Qubit* %3)
  call void @PhaseGradientUnitary_5(%Qubit* %4, %Qubit* %3, %Qubit* %2, %Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__h__body(%Qubit* %4)
  call void @PhaseGradientUnitary_6(%Qubit* %5, %Qubit* %4, %Qubit* %3, %Qubit* %2, %Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__h__body(%Qubit* %5)
  call void @PhaseGradientUnitary_7(%Qubit* %6, %Qubit* %5, %Qubit* %4, %Qubit* %3, %Qubit* %2, %Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__h__body(%Qubit* %6)
  call void @PhaseGradientUnitary_8(%Qubit* %7, %Qubit* %6, %Qubit* %5, %Qubit* %4, %Qubit* %3, %Qubit* %2, %Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__h__body(%Qubit* %7)
  call void @PhaseGradientUnitary_9(%Qubit* %8, %Qubit* %7, %Qubit* %6, %Qubit* %5, %Qubit* %4, %Qubit* %3, %Qubit* %2, %Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__h__body(%Qubit* %8)
  call void @PhaseGradientUnitary_10(%Qubit* %9, %Qubit* %8, %Qubit* %7, %Qubit* %6, %Qubit* %5, %Qubit* %4, %Qubit* %3, %Qubit* %2, %Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__h__body(%Qubit* %9)
  call void @cirq.SWAP_2(%Qubit* %1, %Qubit* %8)
  call void @cirq.SWAP_2(%Qubit* %2, %Qubit* %7)
  call void @cirq.SWAP_2(%Qubit* %3, %Qubit* %6)
  call void @cirq.SWAP_2(%Qubit* %4, %Qubit* %5)
  call void @cirq.SWAP_2(%Qubit* %0, %Qubit* %9)
  ret void
}

declare void @__quantum__qis__h__body(%Qubit*)

define void @PhaseGradientUnitary_2(%Qubit* %0, %Qubit* %1) {
block1:
  %2 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %1, %Qubit* %2)
  call void @__quantum__qis__rz__body(double 5.000000e-01, %Qubit* %1)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %1, %Qubit* %2)
  ret void
}

declare void @__quantum__qis__ccx__body(%Qubit*, %Qubit*, %Qubit*)

declare void @__quantum__qis__rz__body(double, %Qubit*)

define void @PhaseGradientUnitary_3(%Qubit* %0, %Qubit* %1, %Qubit* %2) {
block1:
  %3 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %1, %Qubit* %3)
  call void @__quantum__qis__rz__body(double 5.000000e-01, %Qubit* %1)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %1, %Qubit* %3)
  %4 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %2, %Qubit* %4)
  call void @__quantum__qis__rz__body(double 2.500000e-01, %Qubit* %2)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %2, %Qubit* %4)
  ret void
}

define void @PhaseGradientUnitary_4(%Qubit* %0, %Qubit* %1, %Qubit* %2, %Qubit* %3) {
block1:
  %4 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %1, %Qubit* %4)
  call void @__quantum__qis__rz__body(double 5.000000e-01, %Qubit* %1)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %1, %Qubit* %4)
  %5 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %2, %Qubit* %5)
  call void @__quantum__qis__rz__body(double 2.500000e-01, %Qubit* %2)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %2, %Qubit* %5)
  %6 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %3, %Qubit* %6)
  call void @__quantum__qis__rz__body(double 1.250000e-01, %Qubit* %3)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %3, %Qubit* %6)
  ret void
}

define void @PhaseGradientUnitary_5(%Qubit* %0, %Qubit* %1, %Qubit* %2, %Qubit* %3, %Qubit* %4) {
block1:
  %5 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %1, %Qubit* %5)
  call void @__quantum__qis__rz__body(double 5.000000e-01, %Qubit* %1)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %1, %Qubit* %5)
  %6 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %2, %Qubit* %6)
  call void @__quantum__qis__rz__body(double 2.500000e-01, %Qubit* %2)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %2, %Qubit* %6)
  %7 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %3, %Qubit* %7)
  call void @__quantum__qis__rz__body(double 1.250000e-01, %Qubit* %3)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %3, %Qubit* %7)
  %8 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %4, %Qubit* %8)
  call void @__quantum__qis__rz__body(double 6.250000e-02, %Qubit* %4)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %4, %Qubit* %8)
  ret void
}

define void @PhaseGradientUnitary_6(%Qubit* %0, %Qubit* %1, %Qubit* %2, %Qubit* %3, %Qubit* %4, %Qubit* %5) {
block1:
  %6 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %1, %Qubit* %6)
  call void @__quantum__qis__rz__body(double 5.000000e-01, %Qubit* %1)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %1, %Qubit* %6)
  %7 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %2, %Qubit* %7)
  call void @__quantum__qis__rz__body(double 2.500000e-01, %Qubit* %2)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %2, %Qubit* %7)
  %8 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %3, %Qubit* %8)
  call void @__quantum__qis__rz__body(double 1.250000e-01, %Qubit* %3)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %3, %Qubit* %8)
  %9 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %4, %Qubit* %9)
  call void @__quantum__qis__rz__body(double 6.250000e-02, %Qubit* %4)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %4, %Qubit* %9)
  %10 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %5, %Qubit* %10)
  call void @__quantum__qis__rz__body(double 3.125000e-02, %Qubit* %5)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %5, %Qubit* %10)
  ret void
}

define void @PhaseGradientUnitary_7(%Qubit* %0, %Qubit* %1, %Qubit* %2, %Qubit* %3, %Qubit* %4, %Qubit* %5, %Qubit* %6) {
block1:
  %7 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %1, %Qubit* %7)
  call void @__quantum__qis__rz__body(double 5.000000e-01, %Qubit* %1)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %1, %Qubit* %7)
  %8 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %2, %Qubit* %8)
  call void @__quantum__qis__rz__body(double 2.500000e-01, %Qubit* %2)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %2, %Qubit* %8)
  %9 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %3, %Qubit* %9)
  call void @__quantum__qis__rz__body(double 1.250000e-01, %Qubit* %3)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %3, %Qubit* %9)
  %10 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %4, %Qubit* %10)
  call void @__quantum__qis__rz__body(double 6.250000e-02, %Qubit* %4)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %4, %Qubit* %10)
  %11 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %5, %Qubit* %11)
  call void @__quantum__qis__rz__body(double 3.125000e-02, %Qubit* %5)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %5, %Qubit* %11)
  %12 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %6, %Qubit* %12)
  call void @__quantum__qis__rz__body(double 1.562500e-02, %Qubit* %6)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %6, %Qubit* %12)
  ret void
}

define void @PhaseGradientUnitary_8(%Qubit* %0, %Qubit* %1, %Qubit* %2, %Qubit* %3, %Qubit* %4, %Qubit* %5, %Qubit* %6, %Qubit* %7) {
block1:
  %8 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %1, %Qubit* %8)
  call void @__quantum__qis__rz__body(double 5.000000e-01, %Qubit* %1)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %1, %Qubit* %8)
  %9 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %2, %Qubit* %9)
  call void @__quantum__qis__rz__body(double 2.500000e-01, %Qubit* %2)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %2, %Qubit* %9)
  %10 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %3, %Qubit* %10)
  call void @__quantum__qis__rz__body(double 1.250000e-01, %Qubit* %3)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %3, %Qubit* %10)
  %11 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %4, %Qubit* %11)
  call void @__quantum__qis__rz__body(double 6.250000e-02, %Qubit* %4)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %4, %Qubit* %11)
  %12 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %5, %Qubit* %12)
  call void @__quantum__qis__rz__body(double 3.125000e-02, %Qubit* %5)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %5, %Qubit* %12)
  %13 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %6, %Qubit* %13)
  call void @__quantum__qis__rz__body(double 1.562500e-02, %Qubit* %6)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %6, %Qubit* %13)
  %14 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %7, %Qubit* %14)
  call void @__quantum__qis__rz__body(double 7.812500e-03, %Qubit* %7)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %7, %Qubit* %14)
  ret void
}

define void @PhaseGradientUnitary_9(%Qubit* %0, %Qubit* %1, %Qubit* %2, %Qubit* %3, %Qubit* %4, %Qubit* %5, %Qubit* %6, %Qubit* %7, %Qubit* %8) {
block1:
  %9 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %1, %Qubit* %9)
  call void @__quantum__qis__rz__body(double 5.000000e-01, %Qubit* %1)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %1, %Qubit* %9)
  %10 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %2, %Qubit* %10)
  call void @__quantum__qis__rz__body(double 2.500000e-01, %Qubit* %2)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %2, %Qubit* %10)
  %11 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %3, %Qubit* %11)
  call void @__quantum__qis__rz__body(double 1.250000e-01, %Qubit* %3)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %3, %Qubit* %11)
  %12 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %4, %Qubit* %12)
  call void @__quantum__qis__rz__body(double 6.250000e-02, %Qubit* %4)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %4, %Qubit* %12)
  %13 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %5, %Qubit* %13)
  call void @__quantum__qis__rz__body(double 3.125000e-02, %Qubit* %5)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %5, %Qubit* %13)
  %14 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %6, %Qubit* %14)
  call void @__quantum__qis__rz__body(double 1.562500e-02, %Qubit* %6)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %6, %Qubit* %14)
  %15 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %7, %Qubit* %15)
  call void @__quantum__qis__rz__body(double 7.812500e-03, %Qubit* %7)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %7, %Qubit* %15)
  %16 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %8, %Qubit* %16)
  call void @__quantum__qis__rz__body(double 3.906250e-03, %Qubit* %8)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %8, %Qubit* %16)
  ret void
}

define void @PhaseGradientUnitary_10(%Qubit* %0, %Qubit* %1, %Qubit* %2, %Qubit* %3, %Qubit* %4, %Qubit* %5, %Qubit* %6, %Qubit* %7, %Qubit* %8, %Qubit* %9) {
block1:
  %10 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %1, %Qubit* %10)
  call void @__quantum__qis__rz__body(double 5.000000e-01, %Qubit* %1)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %1, %Qubit* %10)
  %11 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %2, %Qubit* %11)
  call void @__quantum__qis__rz__body(double 2.500000e-01, %Qubit* %2)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %2, %Qubit* %11)
  %12 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %3, %Qubit* %12)
  call void @__quantum__qis__rz__body(double 1.250000e-01, %Qubit* %3)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %3, %Qubit* %12)
  %13 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %4, %Qubit* %13)
  call void @__quantum__qis__rz__body(double 6.250000e-02, %Qubit* %4)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %4, %Qubit* %13)
  %14 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %5, %Qubit* %14)
  call void @__quantum__qis__rz__body(double 3.125000e-02, %Qubit* %5)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %5, %Qubit* %14)
  %15 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %6, %Qubit* %15)
  call void @__quantum__qis__rz__body(double 1.562500e-02, %Qubit* %6)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %6, %Qubit* %15)
  %16 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %7, %Qubit* %16)
  call void @__quantum__qis__rz__body(double 7.812500e-03, %Qubit* %7)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %7, %Qubit* %16)
  %17 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %8, %Qubit* %17)
  call void @__quantum__qis__rz__body(double 3.906250e-03, %Qubit* %8)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %8, %Qubit* %17)
  %18 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %9, %Qubit* %18)
  call void @__quantum__qis__rz__body(double 0x3F60000000000000, %Qubit* %9)
  call void @__quantum__qis__ccx__body(%Qubit* %0, %Qubit* %9, %Qubit* %18)
  ret void
}

define void @cirq.SWAP_2(%Qubit* %0, %Qubit* %1) {
block1:
  call void @__quantum__qis__cnot__body(%Qubit* %0, %Qubit* %1)
  call void @__quantum__qis__cnot__body(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__cnot__body(%Qubit* %0, %Qubit* %1)
  ret void
}

declare void @__quantum__qis__cnot__body(%Qubit*, %Qubit*)

attributes #0 = { "entry_point" "output_labeling_schema" "qir_profiles"="custom" "required_num_qubits"="10" "required_num_results"="0" }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"qir_major_version", i32 1}
!1 = !{i32 7, !"qir_minor_version", i32 0}
!2 = !{i32 1, !"dynamic_qubit_management", i1 true}
!3 = !{i32 1, !"dynamic_result_management", i1 false}
