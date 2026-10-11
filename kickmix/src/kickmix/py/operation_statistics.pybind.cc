#include "kickmix/py/operation_statistics.pybind.h"

#include <algorithm>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <utility>
#include <vector>

#include "kickmix/circuit/op_type.h"
#include "kickmix/py/interactive_simulator.pybind.h"
#include "kickmix/py/util.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

namespace {

const std::array<std::pair<std::string_view, std::vector<OpType>>, 17> OP_COUNT_GROUPS{{
    {"NEG", {OpType::NEG, OpType::NEG_IF}},
    {"BIT_INVERT", {OpType::BIT_INVERT, OpType::BIT_INVERT_IF}},
    {"BIT_STORE0", {OpType::BIT_STORE0, OpType::BIT_STORE0_IF}},
    {"BIT_STORE1", {OpType::BIT_STORE1, OpType::BIT_STORE1_IF}},
    {"X", {OpType::X, OpType::X_IF}},
    {"Z", {OpType::Z, OpType::Z_IF}},
    {"R", {OpType::R, OpType::R_IF}},
    {"HMR", {OpType::HMR, OpType::HMR_IF}},
    {"CX", {OpType::CX, OpType::CX_IF}},
    {"CZ", {OpType::CZ, OpType::CZ_IF}},
    {"SWAP", {OpType::SWAP, OpType::SWAP_IF}},
    {"CCX", {OpType::CCX, OpType::CCX_IF}},
    {"CCZ", {OpType::CCZ, OpType::CCZ_IF}},
    {"Z_POW", {OpType::Z_POW, OpType::Z_POW_IF}},
    {"DEBUG_PRINT",
     {OpType::DEBUG_PRINT_EMPTY,
      OpType::DEBUG_PRINT_Q,
      OpType::DEBUG_PRINT_C,
      OpType::DEBUG_PRINT_EMPTY_IF,
      OpType::DEBUG_PRINT_Q_IF,
      OpType::DEBUG_PRINT_C_IF}},
    {"POP_CONDITION", {OpType::POP_CONDITION}},
    {"PUSH_CONDITION", {OpType::PUSH_CONDITION}},
}};

pybind11::object fixed_precision_angle_to_py_fraction(const FixedPrecisionAngle128 &angle) {
    uint64_t w0 = angle.words[0];
    uint64_t w1 = angle.words[1];
    return pybind11::module_::import("fractions")
        .attr("Fraction")(
            pybind11::cast(w0) | (pybind11::cast(w1) << pybind11::cast(64)), pybind11::cast(1) << pybind11::cast(127));
}

void populate_z_pow_counts(
    pybind11::dict &result,
    const pybind11::object &z_pow_key,
    const std::map<FixedPrecisionAngle128, uint64_t> &angle_counts) {
    for (const auto &[angle, count] : angle_counts) {
        pybind11::object frac = fixed_precision_angle_to_py_fraction(angle);
        pybind11::object key_obj = z_pow_key(frac);
        if (!pybind11::isinstance<pybind11::str>(key_obj)) {
            throw pybind11::type_error("z_pow_key must return a str");
        }
        uint64_t prev = 0;
        if (result.contains(key_obj)) {
            prev = pybind11::cast<uint64_t>(result[key_obj]);
        }
        result[key_obj] = prev + count;
    }
}

}  // namespace

bool PyOperationStatistics::operator==(const PyOperationStatistics &other) const {
    if (raw_op_counts != other.raw_op_counts) {
        return false;
    }
    for (const auto &[angle, count] : z_pow_angle_counts) {
        auto it = other.z_pow_angle_counts.find(angle);
        uint64_t other_count = it != other.z_pow_angle_counts.end() ? it->second : 0;
        if (count != other_count) {
            return false;
        }
    }
    for (const auto &[angle, other_count] : other.z_pow_angle_counts) {
        auto it = z_pow_angle_counts.find(angle);
        uint64_t count = it != z_pow_angle_counts.end() ? it->second : 0;
        if (count != other_count) {
            return false;
        }
    }
    return true;
}

PyOperationStatistics PyOperationStatistics::from_circuit(const Circuit &circuit) {
    PyOperationStatistics stats;
    for (size_t k = 0; k < circuit.num_ops; k++) {
        stats.raw_op_counts[static_cast<uint8_t>(circuit.op_types[k])]++;
    }
    for (size_t k = 0; k < circuit.num_angle_ops; k++) {
        stats.z_pow_angle_counts[circuit.angles[k]]++;
    }
    return stats;
}

PyOperationStatistics PyOperationStatistics::from_simulator(
    const PyInteractiveSimulator &sim, std::optional<int64_t> shot_index) {
    if (!sim.counting_operations) {
        throw std::invalid_argument(
            "Operation counting is not enabled on this simulator. "
            "Pass counting_operations=True to km.Simulator(...).");
    }
    PyOperationStatistics stats;
    if (shot_index.has_value()) {
        if (*shot_index < 0 || static_cast<size_t>(*shot_index) >= sim.batch_size) {
            throw pybind11::index_error("Need 0 <= shot_index < sim.batch_size");
        }
        size_t s = static_cast<size_t>(*shot_index);
        size_t sim_idx = s / PY_SIM_WORD_BITS;
        size_t bit_idx = s % PY_SIM_WORD_BITS;
        const auto &csim = sim.counting_simulators[sim_idx];
        for (size_t op = 0; op < 256; op++) {
            stats.raw_op_counts[op] = csim.new_op_counters[op].compute_total(bit_idx);
        }
        for (const auto &[angle, counter] : csim.z_pow_angle_counters) {
            stats.z_pow_angle_counts.emplace(angle, counter.compute_total(bit_idx));
        }
        return stats;
    }

    for (size_t s_base = 0; s_base < sim.batch_size; s_base += PY_SIM_WORD_BITS) {
        size_t sim_idx = s_base / PY_SIM_WORD_BITS;
        size_t n_bits = std::min(sim.batch_size - s_base, size_t{PY_SIM_WORD_BITS});
        const auto &csim = sim.counting_simulators[sim_idx];
        for (size_t op = 0; op < 256; op++) {
            const auto &counter = csim.new_op_counters[op];
            for (size_t r = 0; r < n_bits; r++) {
                stats.raw_op_counts[op] += counter.compute_total(r);
            }
        }
        for (const auto &[angle, counter] : csim.z_pow_angle_counters) {
            uint64_t total = 0;
            for (size_t r = 0; r < n_bits; r++) {
                total += counter.compute_total(r);
            }
            stats.z_pow_angle_counts[angle] += total;
        }
    }
    return stats;
}

uint64_t PyOperationStatistics::ccp() const {
    return raw_op_counts[static_cast<uint8_t>(OpType::CCX)] + raw_op_counts[static_cast<uint8_t>(OpType::CCX_IF)] +
           raw_op_counts[static_cast<uint8_t>(OpType::CCZ)] + raw_op_counts[static_cast<uint8_t>(OpType::CCZ_IF)];
}

uint64_t PyOperationStatistics::t() const {
    uint64_t total = 0;
    for (const auto &[angle, count] : z_pow_angle_counts) {
        if (angle.is_multiple_of_45_degrees() && !angle.is_multiple_of_90_degrees()) {
            total += count;
        }
    }
    return total;
}

uint64_t PyOperationStatistics::rotations() const {
    uint64_t total = 0;
    for (const auto &[angle, count] : z_pow_angle_counts) {
        if (!angle.is_multiple_of_45_degrees()) {
            total += count;
        }
    }
    return total;
}

uint64_t PyOperationStatistics::clifford_z_pow() const {
    uint64_t total = 0;
    for (const auto &[angle, count] : z_pow_angle_counts) {
        if (angle.is_multiple_of_90_degrees()) {
            total += count;
        }
    }
    return total;
}

uint64_t PyOperationStatistics::total() const {
    uint64_t sum = 0;
    for (uint64_t c : raw_op_counts) {
        sum += c;
    }
    return sum;
}

uint64_t PyOperationStatistics::get_item(std::string_view key) const {
    for (const auto &[name, ops] : OP_COUNT_GROUPS) {
        if (name == key) {
            uint64_t sum = 0;
            for (OpType op : ops) {
                sum += raw_op_counts[static_cast<uint8_t>(op)];
            }
            return sum;
        }
    }
    throw pybind11::key_error(std::string(key));
}

pybind11::dict PyOperationStatistics::z_pow_counts() const {
    pybind11::dict result;
    for (const auto &[angle, count] : z_pow_angle_counts) {
        result[fixed_precision_angle_to_py_fraction(angle)] = count;
    }
    return result;
}

pybind11::dict PyOperationStatistics::op_counts(const pybind11::object &z_pow_key) const {
    bool custom_z_pow = !z_pow_key.is_none();
    pybind11::dict result;
    for (const auto &[name, ops] : OP_COUNT_GROUPS) {
        if (custom_z_pow && name == "Z_POW") {
            continue;
        }
        uint64_t sum = 0;
        for (OpType op : ops) {
            sum += raw_op_counts[static_cast<uint8_t>(op)];
        }
        result[pybind11::str(name.data(), name.size())] = sum;
    }
    if (custom_z_pow) {
        populate_z_pow_counts(result, z_pow_key, z_pow_angle_counts);
    }
    return result;
}

std::string PyOperationStatistics::repr() const {
    std::stringstream ss;
    ss << "<km.OperationStatistics total=" << total() << " ccp=" << ccp() << " t=" << t()
       << " rotations=" << rotations() << ">";
    return ss.str();
}

pybind11::class_<PyOperationStatistics> kickmix_py::register_operation_statistics_class(pybind11::module &m) {
    return pybind11::class_<PyOperationStatistics>(
        m,
        "OperationStatistics",
        clean_doc_string(R"DOC(
            Operation count statistics from a `km.Circuit` or `km.Simulator`.

            Provides per-operation counts (`stats['X']`, `stats.op_counts()`)
            and resource summaries (`ccp` for `CCX`/`CCZ` gates, `t`,
            `rotations`, `clifford_z_pow`, `total`, and `z_pow_counts`).

            Examples:
                >>> import kickmix as km
                >>> c = km.Circuit('''
                ...     X q0
                ...     CCX q0 q1 q2
                ...     Z_POW q0 0.25
                ... ''')
                >>> stats = c.operation_statistics()
                >>> stats.ccp, stats.t, stats['X']
                (1, 1, 1)
        )DOC")
            .data());
}

void kickmix_py::register_operation_statistics_methods(pybind11::class_<PyOperationStatistics> &c_stats) {
    c_stats.def_property_readonly(
        "ccp",
        &PyOperationStatistics::ccp,
        clean_doc_string(R"DOC(
            The number of doubly-controlled Pauli (`CCX` and `CCZ`) operations.

            Includes both unconditional (`CCX`, `CCZ`) and classically
            conditioned (`CCX if ...`, `CCZ if ...`) operations.

            Examples:
                >>> import kickmix as km
                >>> c = km.Circuit('''
                ...     CCX q0 q1 q2
                ...     CCZ q0 q1 q2 if b0
                ...     CX q0 q1
                ... ''')
                >>> c.operation_statistics().ccp
                2
        )DOC")
            .data());

    c_stats.def_property_readonly(
        "t",
        &PyOperationStatistics::t,
        clean_doc_string(R"DOC(
            The number of `Z_POW` operations with odd-multiple-of-45-degree angles.

            T gates are `Z_POW` operations whose angle is an odd multiple of
            0.25 half turns (45 degrees).

            Examples:
                >>> import kickmix as km
                >>> c = km.Circuit('''
                ...     Z_POW q0 0.5
                ...     Z_POW q0 0.25
                ...     Z_POW q0 -0.25 if b0
                ...     Z_POW q0 0.125
                ... ''')
                >>> c.operation_statistics().t
                2
        )DOC")
            .data());

    c_stats.def_property_readonly(
        "rotations",
        &PyOperationStatistics::rotations,
        clean_doc_string(R"DOC(
            The number of arbitrary-angle `Z_POW` rotations (neither Clifford nor T).

            Rotations are `Z_POW` operations whose angle is not a multiple of
            0.25 half turns.

            Examples:
                >>> import kickmix as km
                >>> c = km.Circuit('''
                ...     Z_POW q0 0.5
                ...     Z_POW q0 0.25
                ...     Z_POW q0 0.125
                ...     Z_POW q0 0.1 if b0
                ... ''')
                >>> c.operation_statistics().rotations
                2
        )DOC")
            .data());

    c_stats.def_property_readonly(
        "clifford_z_pow",
        &PyOperationStatistics::clifford_z_pow,
        clean_doc_string(R"DOC(
            The number of Clifford `Z_POW` operations (multiples of 0.5 half turns).

            Examples:
                >>> import kickmix as km
                >>> c = km.Circuit('''
                ...     Z_POW q0 0.0
                ...     Z_POW q0 0.5
                ...     Z_POW q0 -0.5 if b0
                ...     Z_POW q0 0.25
                ... ''')
                >>> c.operation_statistics().clifford_z_pow
                3
        )DOC")
            .data());

    c_stats.def_property_readonly(
        "total",
        &PyOperationStatistics::total,
        clean_doc_string(R"DOC(
            The total number of operations across all operation types.

            Examples:
                >>> import kickmix as km
                >>> c = km.Circuit('''
                ...     X q0
                ...     CX q0 q1 if b0
                ...     CCX q0 q1 q2
                ... ''')
                >>> c.operation_statistics().total
                3
        )DOC")
            .data());

    c_stats.def_property_readonly(
        "z_pow_counts",
        &PyOperationStatistics::z_pow_counts,
        clean_doc_string(R"DOC(
            @signature def z_pow_counts(self) -> dict[fractions.Fraction, int]:
            A dictionary mapping each distinct `Z_POW` angle to its count.

            Angles are represented in half turns in `[0, 2)` as a
            `fractions.Fraction`.

            Examples:
                >>> import kickmix as km
                >>> c = km.Circuit('''
                ...     Z_POW q0 0.25
                ...     Z_POW q1 0.25 if b0
                ...     Z_POW q0 -0.25
                ... ''')
                >>> c.operation_statistics().z_pow_counts
                {Fraction(1, 4): 2, Fraction(7, 4): 1}
        )DOC")
            .data());

    c_stats.def(
        "op_counts",
        &PyOperationStatistics::op_counts,
        pybind11::arg("z_pow_key") = pybind11::none(),
        clean_doc_string(R"DOC(
            @signature def op_counts(self, z_pow_key: Callable[[fractions.Fraction], str] | None = None) -> dict[str, int]:
            Returns a dictionary mapping each operation group name to its count.

            Unconditional and conditional variants of an operation (e.g. `CCX`
            and `CCX_IF`) are summed under the base operation name (`'CCX'`).

            Args:
                z_pow_key: Optional callable taking a `Z_POW` angle (in half
                    turns in `[0, 2)` as a `fractions.Fraction`) and returning
                    a dictionary key `str` under which to accumulate its count.
                    If `None` (the default), all `Z_POW` operations are summed
                    under `'Z_POW'`.

            Returns:
                A dictionary mapping each operation name (`'NEG'`,
                `'BIT_INVERT'`, `'BIT_STORE0'`, `'BIT_STORE1'`, `'X'`, `'Z'`,
                `'R'`, `'HMR'`, `'CX'`, `'CZ'`, `'SWAP'`, `'CCX'`, `'CCZ'`,
                `'Z_POW'`, `'DEBUG_PRINT'`, `'POP_CONDITION'`,
                `'PUSH_CONDITION'`, or custom `z_pow_key` names) to its count.

            Examples:
                >>> import kickmix as km
                >>> c = km.Circuit('''
                ...     X q0
                ...     CX q0 q1 if b0
                ...     Z_POW q0 0.5
                ...     Z_POW q0 0.25
                ...     Z_POW q0 -0.25 if b0
                ...     Z_POW q0 0.125
                ... ''')
                >>> stats = c.operation_statistics()
                >>> stats.op_counts()['X']
                1
                >>> stats.op_counts()['Z_POW']
                4
                >>> def classify_z_pow(angle):
                ...     if angle.denominator <= 2:
                ...         return 'Z_POW_CLIFFORD'
                ...     if angle.denominator == 4:
                ...         return 'T'
                ...     return f'Z_POW_2^-{angle.denominator.bit_length() - 1}'
                >>> grouped = stats.op_counts(z_pow_key=classify_z_pow)
                >>> grouped['Z_POW_CLIFFORD'], grouped['T'], grouped['Z_POW_2^-3']
                (1, 2, 1)
        )DOC")
            .data());

    c_stats.def(
        "__getitem__",
        &PyOperationStatistics::get_item,
        pybind11::arg("key"),
        clean_doc_string(R"DOC(
            @signature def __getitem__(self, key: str) -> int:
            Returns the count for the given base operation name.

            Args:
                key: One of the base operation names (`'NEG'`, `'BIT_INVERT'`,
                    `'BIT_STORE0'`, `'BIT_STORE1'`, `'X'`, `'Z'`, `'R'`,
                    `'HMR'`, `'CX'`, `'CZ'`, `'SWAP'`, `'CCX'`, `'CCZ'`,
                    `'Z_POW'`, `'DEBUG_PRINT'`, `'POP_CONDITION'`,
                    `'PUSH_CONDITION'`).

            Raises:
                KeyError: `key` is not a recognized base operation name.

            Examples:
                >>> import kickmix as km
                >>> c = km.Circuit('''
                ...     X q0
                ...     CCX q0 q1 q2 if b0
                ... ''')
                >>> stats = c.operation_statistics()
                >>> stats['X']
                1
                >>> stats['CCX']
                1
        )DOC")
            .data());

    c_stats.def(
        "__repr__",
        &PyOperationStatistics::repr,
        clean_doc_string(R"DOC(
            @signature def __repr__(self) -> str:
            Returns a summary string describing the operation statistics.
        )DOC")
            .data());

    c_stats.def(pybind11::self == pybind11::self, "Determines if two OperationStatistics have identical counts.");
    c_stats.def(pybind11::self != pybind11::self, "Determines if two OperationStatistics have different counts.");
}
