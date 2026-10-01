#ifndef KICKMIX_FAST_CIRCUIT_H
#define KICKMIX_FAST_CIRCUIT_H

#include <vector>

#include "kickmix/circuit/op_type.h"
#include "kickmix/id/bit_id.h"
#include "kickmix/id/bit_or_false.h"
#include "kickmix/id/qubit_id.h"
#include "kickmix/id/qubit_or_false.h"
#include "kickmix/util/fixed_precision_angle_128.h"
#include "register_data.h"

namespace kickmix {

struct Op {
    OpType kind;
    FixedPrecisionAngle128 angle;
    QubitOrFalse q_control2;
    QubitOrFalse q_control1;
    QubitOrFalse q_target;
    BitIdOrFalse c_target;
    BitIdOrFalse c_condition;
    bool operator==(const Op &rhs) const = default;
    std::string str() const;
    std::string kmx_str() const;
};
std::ostream &operator<<(std::ostream &out, const Op &op);

/// Stores the register data and operation data defining a kickmix circuit.
///
/// Convention: wild instances of this class should always contain validated data.
/// For example, argument given a Circuit argument may assume that `qqq0`
/// points to data with a length of `num_qqq_ops` and that the number of
/// three-qubit operation types that appear in `op_types` is equal to `num_qqq_ops`.
/// Thus, all initialization methods should end with a call to
/// `circuit.validate_and_compute_stats()`.
struct Circuit {
    std::vector<RegisterData> register_data;
    OpType *op_types{};                // Stores the type of each operation in the circuit.
    uint32_t *qqq0{};                  // Stores the first qubit target of each operation with 3 qubit targets.
    uint32_t *qqq1{};                  // Stores the second qubit target of each operation with 3 qubit targets.
    uint32_t *qqq2{};                  // Stores the third qubit target of each operation with 3 qubit targets.
    uint32_t *qq0{};                   // Stores the first qubit target of each operation with 2 qubit targets.
    uint32_t *qq1{};                   // Stores the second qubit target of each operation with 2 qubit targets.
    uint32_t *q0{};                    // Stores the qubit target of each operation with 1 qubit target.
    uint32_t *bc{};                    // Stores the bit condition of each operation with 1 bit condition.
    uint32_t *b0{};                    // Stores the bit target of each operation with 1 bit target.
    FixedPrecisionAngle128 *angles{};  // Stores the angle of each operation with an angle argument.
    size_t num_ops{};                  // Number of operations.
    size_t num_qqq_ops{};              // Number of operations with 3 qubit targets.
    size_t num_qq_ops{};               // Number of operations with 2 qubit targets.
    size_t num_q_ops{};                // Number of operations with 1 qubit target.
    size_t num_c_ops{};                // Number of operations with 1 bit condition.
    size_t num_b_ops{};                // Number of operations with 1 bit target.
    size_t num_angle_ops{};            // Number of operations with an angle.
    size_t num_qubits{};               // All qubit identifiers that appear in the circuit are less than this value.
    size_t num_bits{};                 // All bit identifiers that appear in the circuit are less than this value.

    Circuit() = default;
    void clear();
    ~Circuit();
    Circuit(Circuit &&) noexcept;
    Circuit &operator=(Circuit &&) noexcept;
    Circuit(const Circuit &);
    Circuit &operator=(const Circuit &);
    bool operator==(const Circuit &rhs) const;

    explicit Circuit(std::string_view kmx_text);
    static Circuit from_kmx_file(FILE *file);
    static Circuit from_kmb_file(FILE *file, bool skip_magic = false);
    static Circuit from_kmx_or_kmb_file(FILE *file);

    size_t max_magic() const;
    size_t reaction_depth() const;
    size_t compute_max_condition_depth() const;
    std::string text_diagram() const;
    std::string html_diagram() const;
    std::string svg_diagram() const;
    void write_text_diagram_to(FILE *out, bool use_unicode = false) const;
    void write_text_diagram_to(std::ostream &out_stream, bool use_unicode = false) const;
    void write_svg_or_html_diagram_to(std::ostream &out_stream, bool html) const;

    std::string describe_qubit_control_collision() const;
    bool compute_has_qubit_control_collision() const;
    size_t compute_num_qubits() const;
    size_t compute_num_bits() const;
    void validate_and_compute_stats();
    size_t compute_num_qqq() const;
    size_t compute_num_qq() const;
    size_t compute_num_q() const;
    size_t compute_num_bit_cond() const;
    size_t compute_num_bit_targ() const;
    size_t compute_num_angles() const;

    std::string str() const;
    void write_kmx_to(std::ostream &out) const;
    void write_kmx_to(FILE *file) const;
    void write_kmb_to(FILE *file) const;

    Circuit operator+(const Circuit &other) const;
    Circuit operator*(size_t n) const;

    template <typename TCallback>
    void iter_ops(const TCallback &op_callback) const {
        const uint32_t *ptr_qqq0 = qqq0;
        const uint32_t *ptr_qqq1 = qqq1;
        const uint32_t *ptr_qqq2 = qqq2;
        const uint32_t *ptr_qq0 = qq0;
        const uint32_t *ptr_qq1 = qq1;
        const uint32_t *ptr_q0 = q0;
        const uint32_t *ptr_bit_cond = bc;
        const uint32_t *ptr_bit_targ = b0;
        const FixedPrecisionAngle128 *ptr_angles = angles;

        for (size_t k = 0; k < num_ops; k++) {
            QubitOrFalse q_control2{};
            QubitOrFalse q_control1{};
            QubitOrFalse q_target{};
            BitIdOrFalse c_target{};
            BitIdOrFalse c_condition{};
            FixedPrecisionAngle128 c_angle{};
            auto q_flag = (uint8_t)op_types[k] & OP_HAS_3Q;
            if (q_flag == OP_HAS_3Q) {
                q_control2 = QubitId{*ptr_qqq0++};
                q_control1 = QubitId{*ptr_qqq1++};
                q_target = QubitId{*ptr_qqq2++};
            } else if (q_flag == OP_HAS_2Q) {
                q_control1 = QubitId{*ptr_qq0++};
                q_target = QubitId{*ptr_qq1++};
            } else if (q_flag == OP_HAS_1Q) {
                q_target = QubitId{*ptr_q0++};
            }
            if ((uint8_t)op_types[k] & OP_HAS_COND) {
                c_condition = BitId{*ptr_bit_cond++};
            }
            if ((uint8_t)op_types[k] & OP_HAS_1B) {
                c_target = BitId{*ptr_bit_targ++};
            }
            if (op_types[k] == OpType::Z_POW || op_types[k] == OpType::Z_POW_IF) {
                c_angle = *ptr_angles++;
            }
            op_callback(
                Op{
                    .kind = op_types[k],
                    .angle = c_angle,
                    .q_control2 = q_control2,
                    .q_control1 = q_control1,
                    .q_target = q_target,
                    .c_target = c_target,
                    .c_condition = c_condition,
                });
        }
    }
};
std::ostream &operator<<(std::ostream &out, const Circuit &circuit);

}  // namespace kickmix

#endif
