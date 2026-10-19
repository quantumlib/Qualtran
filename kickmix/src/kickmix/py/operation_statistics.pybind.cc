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

const std::array<std::pair<std::string_view, std::vector<OpType>>, 19> OP_COUNT_GROUPS{{
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
    {"CSWAP", {OpType::CSWAP, OpType::CSWAP_IF}},
    {"CCX", {OpType::CCX, OpType::CCX_IF}},
    {"CCZ", {OpType::CCZ, OpType::CCZ_IF}},
    {"Z_POW", {OpType::Z_POW, OpType::Z_POW_IF}},
    {"INFO_TAG", {OpType::INFO_TAG}},
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

uint64_t group_count(const PyOperationStatistics &stats, const std::vector<OpType> &ops) {
    uint64_t sum = 0;
    for (OpType op : ops) {
        sum += stats.raw_op_counts[static_cast<uint8_t>(op)];
    }
    return sum;
}

}  // namespace

PyOperationStatistics PyOperationStatistics::from_circuit(const Circuit &circuit) {
    PyOperationStatistics stats;
    for (size_t k = 0; k < circuit.num_ops; k++) {
        stats.raw_op_counts[static_cast<uint8_t>(circuit.op_types[k])]++;
    }
    for (size_t k = 0; k < circuit.num_angle_ops; k++) {
        stats.z_pow_level_counts[circuit.angles[k].clifford_hierarchy_level()]++;
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
        for (size_t lvl = 0; lvl < 129; lvl++) {
            stats.z_pow_level_counts[lvl] = csim.z_pow_level_counters[lvl].compute_total(bit_idx);
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
        for (size_t lvl = 0; lvl < 129; lvl++) {
            const auto &counter = csim.z_pow_level_counters[lvl];
            for (size_t r = 0; r < n_bits; r++) {
                stats.z_pow_level_counts[lvl] += counter.compute_total(r);
            }
        }
    }
    return stats;
}

uint64_t PyOperationStatistics::num_toffoli_like() const {
    return raw_op_counts[static_cast<uint8_t>(OpType::CCX)] + raw_op_counts[static_cast<uint8_t>(OpType::CCX_IF)] +
           raw_op_counts[static_cast<uint8_t>(OpType::CCZ)] + raw_op_counts[static_cast<uint8_t>(OpType::CCZ_IF)] +
           raw_op_counts[static_cast<uint8_t>(OpType::CSWAP)] + raw_op_counts[static_cast<uint8_t>(OpType::CSWAP_IF)];
}

uint64_t PyOperationStatistics::num_non_clifford() const {
    return num_toffoli_like() + num_z_pow(3, {});
}

uint64_t PyOperationStatistics::num_t_like() const {
    return z_pow_level_counts[3];
}

uint64_t PyOperationStatistics::num_instructions() const {
    uint64_t sum = 0;
    for (uint64_t c : raw_op_counts) {
        sum += c;
    }
    return sum;
}

uint64_t PyOperationStatistics::num_z_pow(int64_t min_level, std::optional<int64_t> max_level) const {
    int64_t lo = std::max<int64_t>(0, min_level);
    int64_t hi = std::min<int64_t>(128, max_level.value_or(128));
    uint64_t sum = 0;
    for (int64_t k = lo; k <= hi; k++) {
        sum += z_pow_level_counts[static_cast<size_t>(k)];
    }
    return sum;
}

pybind11::dict PyOperationStatistics::z_pow_clifford_hierarchy_counts() const {
    pybind11::dict result;
    for (size_t k = 0; k < z_pow_level_counts.size(); k++) {
        if (z_pow_level_counts[k] > 0) {
            result[pybind11::int_(k)] = z_pow_level_counts[k];
        }
    }
    return result;
}

uint64_t PyOperationStatistics::get_item(std::string_view key) const {
    for (const auto &[name, ops] : OP_COUNT_GROUPS) {
        if (name == key) {
            return group_count(*this, ops);
        }
    }
    throw pybind11::key_error(std::string(key));
}

pybind11::object PyOperationStatistics::get_with_default(
    std::string_view key, const pybind11::object &default_val) const {
    for (const auto &[name, ops] : OP_COUNT_GROUPS) {
        if (name == key) {
            uint64_t sum = group_count(*this, ops);
            return sum > 0 ? pybind11::cast(sum) : default_val;
        }
    }
    return default_val;
}

bool PyOperationStatistics::contains(const pybind11::object &key) const {
    if (!pybind11::isinstance<pybind11::str>(key)) {
        return false;
    }
    auto s = pybind11::cast<std::string_view>(key);
    for (const auto &[name, ops] : OP_COUNT_GROUPS) {
        if (name == s) {
            return group_count(*this, ops) > 0;
        }
    }
    return false;
}

size_t PyOperationStatistics::len() const {
    size_t count = 0;
    for (const auto &[_, ops] : OP_COUNT_GROUPS) {
        if (group_count(*this, ops) > 0) {
            count++;
        }
    }
    return count;
}

pybind11::list PyOperationStatistics::keys() const {
    pybind11::list result;
    for (const auto &[name, ops] : OP_COUNT_GROUPS) {
        if (group_count(*this, ops) > 0) {
            result.append(pybind11::str(name.data(), name.size()));
        }
    }
    return result;
}

pybind11::list PyOperationStatistics::values() const {
    pybind11::list result;
    for (const auto &[_, ops] : OP_COUNT_GROUPS) {
        uint64_t sum = group_count(*this, ops);
        if (sum > 0) {
            result.append(sum);
        }
    }
    return result;
}

pybind11::list PyOperationStatistics::items() const {
    pybind11::list result;
    for (const auto &[name, ops] : OP_COUNT_GROUPS) {
        uint64_t sum = group_count(*this, ops);
        if (sum > 0) {
            result.append(pybind11::make_tuple(pybind11::str(name.data(), name.size()), sum));
        }
    }
    return result;
}

std::string PyOperationStatistics::repr() const {
    std::stringstream ss;
    ss << "<km.OperationStatistics num_instructions=" << num_instructions()
       << " num_toffoli_like=" << num_toffoli_like() << " num_t_like=" << num_t_like() << ">";
    return ss.str();
}

pybind11::class_<PyOperationStatistics> kickmix_py::register_operation_statistics_class(pybind11::module &m) {
    return pybind11::class_<PyOperationStatistics>(
        m,
        "OperationStatistics",
        clean_doc_string(R"DOC(
            Operation count statistics from a `km.Circuit` or `km.Simulator`.

            Acts as a read-only mapping from base operation name (`'X'`,
            `'CX'`, `'CCX'`, `'Z_POW'`, etc.) to count, so `stats['X']`,
            `stats.get('X')`, `'X' in stats`, `dict(stats)`, `stats.keys()`,
            `stats.values()`, and `stats.items()` all work directly. Also
            provides resource helpers (`num_instructions`, `num_toffoli_like`,
            `num_t_like`, `num_z_pow`, and `z_pow_clifford_hierarchy_counts`).

            Examples:
                >>> import kickmix as km
                >>> c = km.Circuit('''
                ...     X q0
                ...     CCX q0 q1 q2
                ...     Z_POW q0 0.25
                ... ''')
                >>> stats = c.operation_statistics()
                >>> stats.num_toffoli_like, stats.num_t_like, stats['X']
                (1, 1, 1)
                >>> dict(stats)['CCX']
                1
        )DOC")
            .data());
}

void kickmix_py::register_operation_statistics_methods(pybind11::class_<PyOperationStatistics> &c_stats) {
    c_stats.def_property_readonly(
        "num_toffoli_like",
        &PyOperationStatistics::num_toffoli_like,
        clean_doc_string(R"DOC(
            The number of Toffoli-like (`CCX`, `CCZ`, and `CSWAP`) operations.

            Includes both unconditional (`CCX q0 q1 q2`, `CCZ q0 q1 q2`,
            `CSWAP q0 q1 q2`) and classically conditioned (`CCX q0 q1 q2 if b3`,
            etc.) operations.

            Caution: This doesn't count Z_POW operations! For example, note that a CCZ
            can be decomposed into seven Z_POW instructions (plus six CNOTs). Different
            architectures have different costs for different Z_POWs and CCZs, so kickmix
            leaves merging those costs as a task for the user. (Consider: it takes four
            T states to produce a CCZ state, but a CCZ state can only be split into two T
            states [using catalysis], so whether the T-to-CCZ cost ratio is 4:1 or 2:1
            depends on whether your magic state factories are producing T states or CCZ
            states.)

            Examples:
                >>> import kickmix as km
                >>> c = km.Circuit('''
                ...     CCX q0 q1 q2
                ...     CCZ q0 q1 q2 if b0
                ...     CSWAP q0 q1 q2
                ...     CX q0 q1
                ...     Z_POW q0 0.25  # not counted!
                ... ''')
                >>> c.operation_statistics().num_toffoli_like
                3
        )DOC")
            .data());

    c_stats.def_property_readonly(
        "num_non_clifford",
        &PyOperationStatistics::num_non_clifford,
        clean_doc_string(R"DOC(
            The number of non-Clifford operations that were counted.

            This count includes CCX gates, CCZ gates, CSWAP gates, and Z_POW gates with
            an exponent that isn't a multiple of 0.5 (i.e. non-Clifford rotations).

            Examples:
                >>> import kickmix as km
                >>> c = km.Circuit('''
                ...     CCX q0 q1 q2
                ...     CCZ q0 q1 q2
                ...     CSWAP q0 q1 q2
                ...     Z_POW q0 0.25
                ...
                ...     CZ q0 1
                ...     Z_POW q0 0.5
                ... ''')
                >>> c.operation_statistics().num_non_clifford
                4
        )DOC")
            .data());

    c_stats.def_property_readonly(
        "num_t_like",
        &PyOperationStatistics::num_t_like,
        clean_doc_string(R"DOC(
            The number of T-like `Z_POW` operations (Clifford hierarchy level 3).

            T-like gates are `Z_POW` operations whose angle is an odd multiple
            of `0.25` half turns (45 degrees). Equivalent to
            `stats.num_z_pow(min_clifford_hierarchy_level=3,
            max_clifford_hierarchy_level=3)`.

            Examples:
                >>> import kickmix as km
                >>> c = km.Circuit('''
                ...     Z_POW q0 0.5
                ...     Z_POW q0 0.25
                ...     Z_POW q0 -0.25 if b0
                ...     Z_POW q0 0.125
                ... ''')
                >>> c.operation_statistics().num_t_like
                2
        )DOC")
            .data());

    c_stats.def_property_readonly(
        "num_instructions",
        &PyOperationStatistics::num_instructions,
        clean_doc_string(R"DOC(
            The total number of operations across all operation types.

            Examples:
                >>> import kickmix as km
                >>> c = km.Circuit('''
                ...     X q0
                ...     CX q0 q1 if b0
                ...     CCX q0 q1 q2
                ... ''')
                >>> c.operation_statistics().num_instructions
                3
        )DOC")
            .data());

    c_stats.def(
        "num_z_pow",
        &PyOperationStatistics::num_z_pow,
        pybind11::arg("min_clifford_hierarchy_level") = 0,
        pybind11::arg("max_clifford_hierarchy_level") = pybind11::none(),
        clean_doc_string(R"DOC(
            @signature def num_z_pow(self, min_clifford_hierarchy_level: int = 0, max_clifford_hierarchy_level: int | None = None) -> int:
            Counts `Z_POW` operations within a Clifford hierarchy level range.

            The Clifford hierarchy level `k` (`0 <= k <= 128`) of a `Z_POW`
            angle is:
            - `k = 0`: `0.0` half turns (identity)
            - `k = 1`: `1.0` half turns (Pauli `Z`)
            - `k = 2`: `0.5` or `1.5` half turns (Clifford `S` / `S^-1`)
            - `k = 3`: odd multiples of `0.25` half turns (`T` / `T^-1`)
            - `k >= 4`: odd multiples of `2**(1 - k)` half turns

            Args:
                min_clifford_hierarchy_level: Minimum Clifford hierarchy level
                    to include (inclusive). Defaults to `0`.
                max_clifford_hierarchy_level: Maximum Clifford hierarchy level
                    to include (inclusive), or `None` for no upper bound.
                    Defaults to `None`.

            Returns:
                Total number of `Z_POW` operations whose Clifford hierarchy
                level satisfies `min <= level <= max`.

            Examples:
                >>> import kickmix as km
                >>> c = km.Circuit('''
                ...     Z_POW q0 0.0
                ...     Z_POW q0 0.5
                ...     Z_POW q0 0.25
                ...     Z_POW q0 -0.25 if b0
                ...     Z_POW q0 0.125
                ...     Z_POW q0 0.1 if b1
                ... ''')
                >>> stats = c.operation_statistics()
                >>> stats.num_z_pow()
                6
                >>> stats.num_z_pow(max_clifford_hierarchy_level=2)
                2
                >>> stats.num_z_pow(min_clifford_hierarchy_level=4)
                2
        )DOC")
            .data());

    c_stats.def_property_readonly(
        "z_pow_clifford_hierarchy_counts",
        &PyOperationStatistics::z_pow_clifford_hierarchy_counts,
        clean_doc_string(R"DOC(
            @signature def z_pow_clifford_hierarchy_counts(self) -> dict[int, int]:
            Maps each present `Z_POW` Clifford hierarchy level to its count.

            Only levels `k` (`0 <= k <= 128`) with a non-zero count are
            included as keys in the returned dictionary.

            Examples:
                >>> import kickmix as km
                >>> c = km.Circuit('''
                ...     Z_POW q0 0.5
                ...     Z_POW q0 0.25
                ...     Z_POW q1 -0.25 if b0
                ...     Z_POW q0 0.125
                ... ''')
                >>> c.operation_statistics().z_pow_clifford_hierarchy_counts
                {2: 1, 3: 2, 4: 1}
        )DOC")
            .data());

    c_stats.def(
        "__getitem__",
        &PyOperationStatistics::get_item,
        pybind11::arg("key"),
        clean_doc_string(R"DOC(
            @signature def __getitem__(self, key: str | Literal['NEG', 'BIT_INVERT', 'BIT_STORE0', 'BIT_STORE1', 'X', 'Z', 'R', 'HMR', 'CX', 'CZ', 'SWAP', 'CSWAP', 'CCX', 'CCZ', 'Z_POW', 'INFO_TAG', 'DEBUG_PRINT', 'POP_CONDITION', 'PUSH_CONDITION']) -> int:
            Returns the count for the given base operation name.

            Unconditional and conditional variants of an operation (e.g.
            `CCX q0 q1 q2` and `CCX q0 q1 q2 if b3`) are summed together under
            the base operation name (`'CCX'`).

            Args:
                key: One of the base operation names (`'NEG'`, `'BIT_INVERT'`,
                    `'BIT_STORE0'`, `'BIT_STORE1'`, `'X'`, `'Z'`, `'R'`,
                    `'HMR'`, `'CX'`, `'CZ'`, `'SWAP'`, `'CSWAP'`, `'CCX'`,
                    `'CCZ'`, `'Z_POW'`, `'INFO_TAG'`, `'DEBUG_PRINT'`,
                    `'POP_CONDITION'`, `'PUSH_CONDITION'`).

            Raises:
                KeyError: `key` is not a recognized base operation name.

            Examples:
                >>> import kickmix as km
                >>> c = km.Circuit('''
                ...     X q0
                ...     CCX q0 q1 q2 if b0
                ... ''')
                >>> stats = c.operation_statistics()
                >>> stats['X'], stats['CCX']
                (1, 1)
        )DOC")
            .data());

    c_stats.def(
        "get",
        &PyOperationStatistics::get_with_default,
        pybind11::arg("key"),
        pybind11::arg("default") = pybind11::none(),
        clean_doc_string(R"DOC(
            @signature def get(self, key: str | Literal['NEG', 'BIT_INVERT', 'BIT_STORE0', 'BIT_STORE1', 'X', 'Z', 'R', 'HMR', 'CX', 'CZ', 'SWAP', 'CSWAP', 'CCX', 'CCZ', 'Z_POW', 'INFO_TAG', 'DEBUG_PRINT', 'POP_CONDITION', 'PUSH_CONDITION'], default: Any = None) -> Any:
            Returns the count for `key` if non-zero, else `default`.

            Args:
                key: Base operation name to look up.
                default: Value to return if `key` is not present with a
                    non-zero count. Defaults to `None`.

            Examples:
                >>> import kickmix as km
                >>> stats = km.Circuit('X q0').operation_statistics()
                >>> stats.get('X'), stats.get('CCX'), stats.get('CCX', 0)
                (1, None, 0)
        )DOC")
            .data());

    c_stats.def(
        "__contains__",
        &PyOperationStatistics::contains,
        pybind11::arg("key"),
        clean_doc_string(R"DOC(
            @signature def __contains__(self, key: object) -> bool:
            Returns whether `key` is a base operation with a non-zero count.

            Examples:
                >>> import kickmix as km
                >>> stats = km.Circuit('CCX q0 q1 q2').operation_statistics()
                >>> 'CCX' in stats, 'X' in stats, 'UNKNOWN' in stats
                (True, False, False)
        )DOC")
            .data());

    c_stats.def(
        "__len__",
        &PyOperationStatistics::len,
        clean_doc_string(R"DOC(
            @signature def __len__(self) -> int:
            Returns the number of distinct base operation types with non-zero counts.

            Examples:
                >>> import kickmix as km
                >>> len(km.Circuit('').operation_statistics())
                0
                >>> len(km.Circuit('X q0\nCCX q0 q1 q2').operation_statistics())
                2
        )DOC")
            .data());

    c_stats.def(
        "__iter__",
        [](const PyOperationStatistics &self) {
            return pybind11::iter(self.keys());
        },
        clean_doc_string(R"DOC(
            @signature def __iter__(self) -> Iterator[str]:
            Iterates over base operation names that have non-zero counts.

            Examples:
                >>> import kickmix as km
                >>> list(km.Circuit('X q0\nCCX q0 q1 q2').operation_statistics())
                ['X', 'CCX']
        )DOC")
            .data());

    c_stats.def(
        "keys",
        &PyOperationStatistics::keys,
        clean_doc_string(R"DOC(
            @signature def keys(self) -> list[str]:
            Returns base operation names that have non-zero counts.

            Examples:
                >>> import kickmix as km
                >>> km.Circuit('X q0\nCCX q0 q1 q2').operation_statistics().keys()
                ['X', 'CCX']
        )DOC")
            .data());

    c_stats.def(
        "values",
        &PyOperationStatistics::values,
        clean_doc_string(R"DOC(
            @signature def values(self) -> list[int]:
            Returns non-zero counts corresponding to `stats.keys()`.

            Examples:
                >>> import kickmix as km
                >>> km.Circuit('X q0\nX q1\nCCX q0 q1 q2').operation_statistics().values()
                [2, 1]
        )DOC")
            .data());

    c_stats.def(
        "items",
        &PyOperationStatistics::items,
        clean_doc_string(R"DOC(
            @signature def items(self) -> list[tuple[str, int]]:
            Returns `(op_name, count)` pairs for all non-zero operation counts.

            Examples:
                >>> import kickmix as km
                >>> km.Circuit('X q0\nCCX q0 q1 q2').operation_statistics().items()
                [('X', 1), ('CCX', 1)]
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
