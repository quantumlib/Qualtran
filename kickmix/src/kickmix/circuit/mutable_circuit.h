#ifndef KICKMIX_MUTABLE_CIRCUIT_H
#define KICKMIX_MUTABLE_CIRCUIT_H

#include <vector>

#include "circuit.h"
#include "kickmix/mem/monotonic_arena.h"
#include "op_type.h"
#include "register_data.h"

namespace kickmix {

/// Stores circuit register data and operation data in a form that can be appended to.
struct MutableCircuit {
    std::vector<RegisterData> register_data;
    MonotonicArena<OpType, 32> op_types;  // Stores the type of each operation in the circuit.
    MonotonicArena<uint32_t, 32> qqq0;    // Stores the first qubit target of each operation with 3 qubit targets.
    MonotonicArena<uint32_t, 32> qqq1;    // Stores the second qubit target of each operation with 3 qubit targets.
    MonotonicArena<uint32_t, 32> qqq2;    // Stores the third qubit target of each operation with 3 qubit targets.
    MonotonicArena<uint32_t, 32> qq0;     // Stores the first qubit target of each operation with 2 qubit targets.
    MonotonicArena<uint32_t, 32> qq1;     // Stores the second qubit target of each operation with 2 qubit targets.
    MonotonicArena<uint32_t, 32> q0;      // Stores the qubit target of each operation with 1 qubit target.
    MonotonicArena<uint32_t, 32> bc;      // Stores the bit condition of each operation with 1 bit condition.
    MonotonicArena<uint32_t, 32> b0;      // Stores the bit target of each operation with 1 bit target.
    MonotonicArena<FixedPrecisionAngle128, 32> angles;  // Stores the angle of each operation with an angle argument.

    MutableCircuit() = default;
    MutableCircuit(MutableCircuit &&) noexcept = default;
    MutableCircuit &operator=(MutableCircuit &&) noexcept = default;

    // (this has to be explicit because MonotonicArena has no copy constructor,
    // since it is used in cases where data cannot move and copy construction
    // is not well defined in such cases)
    MutableCircuit(const MutableCircuit &other)
        : register_data(other.register_data),
          op_types(),
          qqq0(),
          qqq1(),
          qqq2(),
          qq0(),
          qq1(),
          q0(),
          bc(),
          b0(),
          angles() {
        op_types.push_back_many(other.op_types);
        qqq0.push_back_many(other.qqq0);
        qqq1.push_back_many(other.qqq1);
        qqq2.push_back_many(other.qqq2);
        qq0.push_back_many(other.qq0);
        qq1.push_back_many(other.qq1);
        q0.push_back_many(other.q0);
        bc.push_back_many(other.bc);
        b0.push_back_many(other.b0);
        angles.push_back_many(other.angles);
    }
    // (this has to be explicit because MonotonicArena has no copy assignment,
    // since it is used in cases where data cannot move and copy assignment
    // is not well defined in such cases)
    MutableCircuit &operator=(const MutableCircuit &other) {
        register_data = other.register_data;
        op_types.clear();
        op_types.push_back_many(other.op_types);
        qqq0.clear();
        qqq0.push_back_many(other.qqq0);
        qqq1.clear();
        qqq1.push_back_many(other.qqq1);
        qqq2.clear();
        qqq2.push_back_many(other.qqq2);
        qq0.clear();
        qq0.push_back_many(other.qq0);
        qq1.clear();
        qq1.push_back_many(other.qq1);
        q0.clear();
        q0.push_back_many(other.q0);
        bc.clear();
        bc.push_back_many(other.bc);
        b0.clear();
        b0.push_back_many(other.b0);
        angles.clear();
        angles.push_back_many(other.angles);
        return *this;
    }

    /// Computes a value larger than the maximum qubit id that appears in the register/operation data.
    size_t compute_cur_num_qubits() const;

    /// Validates all operations are well-formed, and then converts to a frozen Circuit.
    Circuit to_validated_circuit() const;

    /// Parses the given KMX file format text and appends the contained operations
    /// to the mutable circuit.
    ///
    /// Caution: Operations that are syntactically correct, but semantically wrong,
    /// like "CX q0 q0" being invalid (because the same qubit is used for the control
    /// and the target) are not detected by this method. They are only detected when
    /// later calling `to_validated_circuit()`.
    void append_from_kmx_text(std::string_view kmx);
    void append_from_kmx_file(FILE *file);
    void append_from_kmx_line(std::string_view kmx_line);

    /// Appends all operations from the other circuit to this circuit.
    void append(const Circuit &other);
    /// Appends all operations from the other circuit to this circuit.
    void append(const MutableCircuit &other);
    /// Appends all operations from the other circuit to this circuit, but in reversed order.
    void append_reversed(const MutableCircuit &other);

    /// Clears all circuit data.
    void clear();
};

}  // namespace kickmix

#endif
