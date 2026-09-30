#include "interactive_simulator.pybind.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kickmix/py/circuit/circuit.pybind.h"
#include "kickmix/py/util.pybind.h"
#include "kickmix/py/val/id.pybind.h"
#include "val/angle.pybind.h"
#include "val/converted_array.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

static std::mt19937_64 externally_seeded_rng() {
    std::random_device d;
    std::seed_seq seq{d(), d(), d(), d(), d(), d(), d(), d()};
    std::mt19937_64 result(seq);
    return result;
}

static void sim_clear_for_shot(PyInteractiveSimulator &self) {
    for (auto &sim : self.simulators) {
        sim.clear_for_shot();
    }
}
bool peek_single_bit(const PyInteractiveSimulator &self, QubitOrBitOrBool index, size_t shot) {
    if (shot >= self.batch_size) {
        throw pybind11::index_error("Need shot_index < sim.batch_size");
    }
    return self.simulators[shot / PY_SIM_WORD_BITS].safe_read_val(index).bit(shot % PY_SIM_WORD_BITS);
}

void set_single_bit(PyInteractiveSimulator &self, QubitOrBit index, size_t shot, bool new_value) {
    if (shot >= self.batch_size) {
        throw pybind11::index_error("Need shot_index < sim.batch_size");
    }
    self.simulators[shot / PY_SIM_WORD_BITS].val_for(index).set_bit(shot % PY_SIM_WORD_BITS, new_value);
}

/// Writes the bytes of a python int in range(0, 2**max_bits) into the given byte buffer.
void int_obj_to_bytes_with_max_bits(
    const pybind11::object &int_obj,
    uint8_t *bytes_buffer,
    size_t max_bits,
    const char *value_name,
    const char *range_name) {
    auto type_ptr = Py_TYPE(int_obj.ptr());
    if (type_ptr != &PyLong_Type && type_ptr != &PyBool_Type) {
        std::stringstream ss;
        ss << "Expected an int but got " << value_name << "=" << pybind11::repr(int_obj);
        throw std::invalid_argument(ss.str());
    }

    size_t max_bytes = (max_bits + 7) / 8;
    if (int_obj.equal(pybind11::cast(0))) {
        // Need this special case because PyLong_AsNativeBytes claims 0 takes 8 bytes to store rather than 0.
        memset(bytes_buffer, 0, max_bytes);
        return;
    }

    Py_ssize_t non_zero_bytes = PyLong_AsNativeBytes(
        int_obj.ptr(),
        bytes_buffer,
        max_bytes,
        Py_ASNATIVEBYTES_LITTLE_ENDIAN | Py_ASNATIVEBYTES_REJECT_NEGATIVE | Py_ASNATIVEBYTES_UNSIGNED_BUFFER);
    bool has_extra = false;
    if (max_bits % 8) {
        uint8_t mask = 0xFF >> (8 - (max_bits % 8));
        uint8_t b = bytes_buffer[max_bytes - 1];
        has_extra = (b & mask) != b;
    }
    bool has_err = non_zero_bytes < 0 || (size_t)non_zero_bytes > max_bytes || has_extra;
    if (has_err) {
        std::stringstream ss;
        ss << value_name << "=" << pybind11::repr(int_obj);
        ss << " not in range(2**(" << range_name << "=";
        ss << max_bits;
        ss << "))";
        throw std::invalid_argument(ss.str());
    }
}

pybind11::object int_obj_from_bytes_with_max_bits(uint8_t *bytes_buffer, size_t max_bits) {
    if (max_bits == 0) {
        return pybind11::cast(0);
    }
    size_t bytes_len = (max_bits + 7) / 8;
    if (max_bits % 8) {
        uint8_t mask = 0xFF >> (8 - (max_bits % 8));
        bytes_buffer[bytes_len - 1] &= mask;
    }
    PyObject *py_long = PyLong_FromUnsignedNativeBytes(bytes_buffer, bytes_len, Py_ASNATIVEBYTES_LITTLE_ENDIAN);
    if (!py_long) {
        throw pybind11::error_already_set();
    }
    return pybind11::reinterpret_steal<pybind11::object>(py_long);
}

PyInteractiveSimulator::PyInteractiveSimulator(size_t init_batch_size) {
    batch_size = init_batch_size;
    for (size_t k = 0; k < batch_size; k += PY_SIM_WORD_BITS) {
        simulators.emplace_back(externally_seeded_rng());
    }
    ensure_byte_buf_can_store_bits(simulators.size() * PY_SIM_WORD_BITS);
}

pybind11::class_<PyInteractiveSimulator> kickmix_py::register_interactive_simulator_class(pybind11::module &m) {
    return pybind11::class_<PyInteractiveSimulator>(
        m,
        "Simulator",
        clean_doc_string(R"DOC(
             An interactive kickmix simulator.
         )DOC")
            .data());
}

void PyInteractiveSimulator::ensure_big_enough_state_for(QubitOrBit val) {
    if (val.is_qubit()) {
        ensure_big_enough_state_for(val.untagged_id() + 1, 0);
    } else {
        ensure_big_enough_state_for(0, val.untagged_id() + 1);
    }
}

void PyInteractiveSimulator::ensure_byte_buf_can_store_bits(size_t num_bits) {
    size_t num_bytes = (num_bits + 7) / 8;
    if (byte_buf.size() < num_bytes) {
        byte_buf.resize(num_bytes);
    }
}
void PyInteractiveSimulator::ensure_big_enough_state_for(size_t num_qubits, size_t num_bits) {
    for (auto &sim : simulators) {
        sim.ensure_big_enough_state_for(num_qubits, num_bits);
    }
}

pybind11::object peek_within_shot_2d_np_bool(
    const PyInteractiveSimulator &self, const stride_span_z &items, size_t shot_index, pybind11::object out) {
    size_t n0 = items.size();
    if (out.is_none()) {
        auto numpy = pybind11::module::import("numpy");
        out = numpy.attr("empty")(pybind11::make_tuple(n0), numpy.attr("bool_"));
    }

    if (!pybind11::isinstance<pybind11::array_t<bool>>(out)) {
        throw std::invalid_argument("out wasn't a np.ndarray[np.bool_].");
    }
    auto buf = pybind11::cast<pybind11::array_t<bool>>(out);
    if (buf.ndim() != 1) {
        throw std::invalid_argument("len(out.shape) != 1");
    }
    if ((size_t)buf.shape(0) != n0) {
        std::stringstream ss;
        ss << "Expected out.shape == (" << n0 << ")";
        ss << " but got out.shape=(" << buf.shape(0) << ",).";
        throw std::invalid_argument(ss.str());
    }

    auto s0 = buf.strides(0);
    auto out_ptr = buf.mutable_data(0);
    for (size_t k0 = 0; k0 < items.size(); k0++) {
        *out_ptr = peek_single_bit(self, items[k0], shot_index);
        out_ptr += s0;
    }

    return out;
}

pybind11::object peek_across_shots_2d_np_bool(
    const PyInteractiveSimulator &self, const stride_span_z &items, pybind11::object out) {
    size_t n0 = items.size();
    size_t n1 = self.batch_size;
    if (out.is_none()) {
        auto numpy = pybind11::module::import("numpy");
        out = numpy.attr("empty")(pybind11::make_tuple(n0, n1), numpy.attr("bool_"));
    }

    if (!pybind11::isinstance<pybind11::array_t<bool>>(out)) {
        throw std::invalid_argument("out wasn't a np.ndarray[np.bool_].");
    }
    auto buf = pybind11::cast<pybind11::array_t<bool>>(out);
    if (buf.ndim() != 2) {
        throw std::invalid_argument("len(out.shape) != 2");
    }
    if ((size_t)buf.shape(0) != n0 || (size_t)buf.shape(1) != n1) {
        std::stringstream ss;
        ss << "Expected out.shape == (" << n0 << ", " << n1 << ")";
        ss << " but got out.shape=(" << buf.shape(0) << ", " << buf.shape(1) << ").";
        throw std::invalid_argument(ss.str());
    }

    auto s1 = buf.strides(1);
    for (size_t k0 = 0; k0 < items.size(); k0++) {
        auto out_ptr = buf.mutable_data(k0, 0);
        for (size_t k1_a = 0; k1_a < self.batch_size; k1_a += PY_SIM_WORD_BITS) {
            auto v = self.simulators[k1_a / PY_SIM_WORD_BITS].safe_read_val(items[k0]);
            size_t e = std::min(self.batch_size - k1_a, size_t{PY_SIM_WORD_BITS});
            for (size_t k1_b = 0; k1_b < e; k1_b++) {
                *out_ptr = v.bit(k1_b % PY_SIM_WORD_BITS);
                out_ptr += s1;
            }
        }
    }

    return out;
}

pybind11::object peek_within_shot_1d_np_bool(
    const PyInteractiveSimulator &self, const QubitOrBitOrBool &index, size_t shot_index, pybind11::object out) {
    if (out.is_none()) {
        auto numpy = pybind11::module::import("numpy");
        out = numpy.attr("empty")(pybind11::make_tuple(), numpy.attr("bool_"));
    }

    if (!pybind11::isinstance<pybind11::array_t<bool>>(out)) {
        throw std::invalid_argument("out wasn't a np.ndarray[np.bool_].");
    }
    auto buf = pybind11::cast<pybind11::array_t<bool>>(out);
    if (buf.ndim() != 0) {
        throw std::invalid_argument("len(out.shape) != 0");
    }

    auto out_ptr = buf.mutable_data();
    *out_ptr = peek_single_bit(self, index, shot_index);
    return out;
}

pybind11::object peek_across_shots_1d_np_bool(
    const PyInteractiveSimulator &self, const QubitOrBitOrBool &index, pybind11::object out) {
    size_t n1 = self.batch_size;
    if (out.is_none()) {
        auto numpy = pybind11::module::import("numpy");
        out = numpy.attr("empty")(pybind11::make_tuple(n1), numpy.attr("bool_"));
    }

    if (!pybind11::isinstance<pybind11::array_t<bool>>(out)) {
        throw std::invalid_argument("out wasn't a np.ndarray[np.bool_].");
    }
    auto buf = pybind11::cast<pybind11::array_t<bool>>(out);
    if (buf.ndim() != 1) {
        throw std::invalid_argument("len(out.shape) != 1");
    }
    if ((size_t)buf.shape(0) != n1) {
        std::stringstream ss;
        ss << "Expected out.shape == (" << n1 << ",)";
        ss << " but got out.shape=(" << buf.shape(0) << ",).";
        throw std::invalid_argument(ss.str());
    }

    auto s1 = buf.strides(0);
    auto out_ptr = buf.mutable_data(0);
    for (size_t k1_a = 0; k1_a < self.batch_size; k1_a += PY_SIM_WORD_BITS) {
        auto v = self.simulators[k1_a / PY_SIM_WORD_BITS].safe_read_val(index);
        size_t e = std::min(self.batch_size - k1_a, size_t{PY_SIM_WORD_BITS});
        for (size_t k1_b = 0; k1_b < e; k1_b++) {
            *out_ptr = v.bit(k1_b % PY_SIM_WORD_BITS);
            out_ptr += s1;
        }
    }

    return out;
}

pybind11::object peek_across_shots_1d_np_u64(
    PyInteractiveSimulator &self, const stride_span_z &items, pybind11::object out) {
    size_t n0 = items.size();
    size_t n1 = (self.batch_size + 63) / 64;
    if (out.is_none()) {
        auto numpy = pybind11::module::import("numpy");
        out = numpy.attr("empty")(pybind11::make_tuple(n0, n1), numpy.attr("uint64"));
    }

    if (!pybind11::isinstance<pybind11::array_t<uint64_t>>(out)) {
        throw std::invalid_argument("out wasn't a np.ndarray[np.uint64].");
    }
    auto buf = pybind11::cast<pybind11::array_t<uint64_t>>(out);
    if (buf.ndim() != 2) {
        throw std::invalid_argument("len(out.shape) != 2");
    }
    if ((size_t)buf.shape(0) != n0 || (size_t)buf.shape(1) != n1) {
        std::stringstream ss;
        ss << "Expected out.shape == (" << n0 << ", " << n1 << ")";
        ss << " but got out.shape=(" << buf.shape(0) << ", " << buf.shape(1) << ").";
        throw std::invalid_argument(ss.str());
    }

    auto s1 = buf.strides(1);
    for (size_t k0 = 0; k0 < items.size(); k0++) {
        auto out_ptr = buf.mutable_data(k0, 0);
        for (size_t k1_a = 0; k1_a < self.batch_size; k1_a += PY_SIM_WORD_BITS) {
            auto v = self.simulators[k1_a / PY_SIM_WORD_BITS].safe_read_val(items[k0]);
            size_t e = (std::min(self.batch_size - k1_a, size_t{PY_SIM_WORD_BITS}) + 63) / 64;
            for (size_t k1_b = 0; k1_b < e; k1_b++) {
                *out_ptr = v.u64(k1_b % PY_SIM_WORD_BITS);
                out_ptr += s1;
            }
        }
    }

    return out;
}

pybind11::object peek_within_shot_1d_int(PyInteractiveSimulator &self, QubitOrBitOrBool index, size_t shot) {
    if (shot >= self.batch_size) {
        throw pybind11::index_error("Need shot < sim.batch_size");
    }
    return pybind11::cast(peek_single_bit(self, index, shot));
}

pybind11::object peek_within_shot_2d_int(PyInteractiveSimulator &self, const stride_span_z &items, size_t shot) {
    self.ensure_byte_buf_can_store_bits(items.size());
    memset(self.byte_buf.data(), 0, self.byte_buf.size());
    for (size_t k = 0; k < items.size(); k++) {
        bool b = peek_single_bit(self, items[k], shot);
        if (b) {
            self.byte_buf[k / 8] ^= uint8_t{1} << (k % 8);
        }
    }
    return int_obj_from_bytes_with_max_bits(self.byte_buf.data(), items.size());
}

pybind11::object peek_across_shots_1d_int(PyInteractiveSimulator &self, QubitOrBitOrBool index) {
    uint8_t *out = self.byte_buf.data();
    for (const auto &sim : self.simulators) {
        auto v = sim.safe_read_val(index);
        memcpy(out, &v, sizeof(v));
        out += sizeof(v);
    }
    return int_obj_from_bytes_with_max_bits(self.byte_buf.data(), self.batch_size);
}

pybind11::object peek_across_shots_2d_int(PyInteractiveSimulator &self, const stride_span_z &items) {
    std::vector<pybind11::object> out;
    for (size_t k = 0; k < items.size(); k++) {
        out.push_back(peek_across_shots_1d_int(self, items[k]));
    }
    return pybind11::cast(out);
}

pybind11::object sim_read_phase(PyInteractiveSimulator &self, size_t shot_index) {
    if (shot_index >= self.batch_size) {
        throw pybind11::index_error("Need shot_index < sim.batch_size");
    }

    const auto &sim = self.simulators[shot_index / PY_SIM_WORD_BITS];
    size_t k2 = shot_index % PY_SIM_WORD_BITS;
    const auto &angle = sim.angles[k2];
    uint64_t w0 = angle.words[0];
    uint64_t w1 = angle.words[1];
    if (sim.global_phase_ref().bit(k2)) {
        w1 ^= uint64_t{1} << 63;
    }
    if (w0 == 0 && w1 == 0) {
        return pybind11::cast(0);
    }
    if (w0 == 0 && w1 == uint64_t{1} << 63) {
        return pybind11::cast(1);
    }

    // Pack into a fraction.
    return pybind11::module_::import("fractions")
        .attr("Fraction")(
            pybind11::cast(w0) | (pybind11::cast(w1) << pybind11::cast(64)), pybind11::cast(1) << pybind11::cast(127));
}

void sim_write_phase(PyInteractiveSimulator &self, size_t shot_index, pybind11::object new_value) {
    auto new_angle = fixed_precision_angle_from_half_turns_obj(new_value);

    if (shot_index >= self.batch_size) {
        throw pybind11::index_error("Need shot_index < sim.batch_size");
    }
    auto &sim = self.simulators[shot_index / PY_SIM_WORD_BITS];
    size_t k2 = shot_index % PY_SIM_WORD_BITS;
    sim.angles[k2] = new_angle;
    sim.global_phase_ref().set_bit(k2, false);
}

void write_across_shots_1d_int(PyInteractiveSimulator &self, QubitOrBit index, const pybind11::object &value_obj) {
    int_obj_to_bytes_with_max_bits(value_obj, self.byte_buf.data(), self.batch_size, "new_value", "batch_size");

    const uint8_t *in = self.byte_buf.data();
    self.ensure_big_enough_state_for(index);
    for (auto &e : self.simulators) {
        auto &v = e.val_for(index);
        memcpy(&v, in, sizeof(v));
        in += sizeof(v);
    }
}

void write_across_shots_1d_array(PyInteractiveSimulator &self, QubitOrBit index, const pybind11::object &value_obj) {
    auto array = pybind11::cast<pybind11::array_t<bool>>(value_obj);
    if (array.ndim() != 1) {
        throw std::invalid_argument("len(new_value.shape) != 1");
    }
    if ((size_t)array.shape(0) != self.batch_size) {
        std::stringstream ss;
        ss << "Expected new_value.shape == (" << self.batch_size << ",)";
        ss << " but got new_value.shape=(" << array.shape(0) << ",).";
        throw std::invalid_argument(ss.str());
    }
    auto stride = array.strides(0);
    auto in_ptr = array.data(0);
    for (size_t k = 0; k < self.batch_size; k++) {
        set_single_bit(self, index, k, *in_ptr);
        in_ptr += stride;
    }
}

ConvertedArrayXZ index_to_sim_items(const PyInteractiveSimulator &self, const pybind11::object &indices_obj);

void write_across_shots_2d_int(
    PyInteractiveSimulator &self, const stride_span<const QubitOrBit> &indices, const pybind11::object &value_obj) {
    size_t n = pybind11::len(value_obj);
    if (n != indices.size()) {
        std::stringstream ss;
        ss << "Expected len(new_value) == len(index) == " << indices.size();
        ss << " but got len(new_value)=" << n << ".";
        throw std::invalid_argument(ss.str());
    }

    size_t k = 0;
    for (const auto &item : value_obj) {
        write_across_shots_1d_int(self, indices[k++], pybind11::cast<pybind11::object>(item));
    }
}

void write_across_shots_2d_np_bool(
    PyInteractiveSimulator &self, const stride_span<const QubitOrBit> &indices, const pybind11::object &value_obj) {
    auto array = pybind11::cast<pybind11::array_t<bool>>(value_obj);
    if (array.ndim() != 2) {
        throw std::invalid_argument("len(new_value.shape) != 2");
    }
    if ((size_t)array.shape(0) != indices.size() || (size_t)array.shape(1) != self.batch_size) {
        std::stringstream ss;
        ss << "Expected new_value.shape == (" << indices.size() << ", " << self.batch_size << ")";
        ss << " but got new_value.shape=(" << array.shape(0) << ", " << array.shape(1) << ").";
        throw std::invalid_argument(ss.str());
    }

    for (auto q : indices) {
        self.ensure_big_enough_state_for(q);
    }
    auto s0 = array.strides(0);
    auto s1 = array.strides(1);
    auto base = array.data(0, 0);
    for (size_t k0 = 0; k0 < indices.size(); k0++) {
        auto in_ptr = base + (Py_ssize_t)k0 * s0;
        for (size_t k1 = 0; k1 < self.batch_size; k1++) {
            set_single_bit(self, indices[k0], k1, *in_ptr);
            in_ptr += s1;
        }
    }
}

void write_across_shots(
    PyInteractiveSimulator &self, const pybind11::object &index_obj, const pybind11::object &value_obj) {
    auto indices_conv = index_to_sim_items(self, index_obj);
    auto indices = indices_conv.span.checked_cast_to_qubit_or_bit("index");

    auto type_ptr = Py_TYPE(value_obj.ptr());
    bool is_int = type_ptr == &PyLong_Type || type_ptr == &PyBool_Type;
    bool is_bool_array = pybind11::isinstance<pybind11::array_t<bool>>(value_obj);

    if (indices_conv.was_singleton) {
        if (is_int) {
            write_across_shots_1d_int(self, indices[0], value_obj);
            return;
        }
        if (is_bool_array) {
            write_across_shots_1d_array(self, indices[0], value_obj);
            return;
        }
    } else {
        // Checked before the sequence case, because an ndarray is also a sequence.
        if (is_bool_array) {
            write_across_shots_2d_np_bool(self, indices, value_obj);
            return;
        }
        if (!is_int && PySequence_Check(value_obj.ptr())) {
            write_across_shots_2d_int(self, indices, value_obj);
            return;
        }
    }

    std::stringstream ss;
    ss << "Don't know how to write from ";
    ss << pybind11::repr(value_obj);
    if (indices_conv.was_singleton) {
        ss << " (expected an `int` or `np.ndarray` with `dtype=np.bool_`)";
    } else {
        ss << " (expected a sequence of `int` or an `np.ndarray` with `dtype=np.bool_`,";
        ss << " because len(index) == " << indices.size() << ")";
    }
    throw std::invalid_argument(ss.str());
}

ConvertedArrayXZ index_to_sim_items(const PyInteractiveSimulator &self, const pybind11::object &indices_obj) {
    if (pybind11::isinstance<pybind11::str>(indices_obj)) {
        for (size_t k = 0; k < self.register_data.size(); k++) {
            if (self.register_data[k].name == pybind11::cast<std::string_view>(indices_obj)) {
                return stride_span_z(self.register_data[k].contents);
            }
        }
        throw pybind11::index_error(pybind11::repr(indices_obj));
    }

    if (pybind11::isinstance<RegisterId>(indices_obj)) {
        auto r = pybind11::cast<kickmix::RegisterId>(indices_obj);
        if (r.id >= self.register_data.size()) {
            throw pybind11::index_error(pybind11::repr(indices_obj));
        }
        return stride_span_z(self.register_data[r.id].contents);
    }

    return ConvertedArrayXZ::from_obj(indices_obj, "indices");
}

pybind11::object read_phase_flipped_across_shots_int(PyInteractiveSimulator &self) {
    uint8_t *out = self.byte_buf.data();
    for (size_t k = 0; k < self.batch_size; k++) {
        auto r = k % PY_SIM_WORD_BITS;
        const auto &sim = self.simulators[k / PY_SIM_WORD_BITS];
        auto angle = sim.angles[r];
        if (sim.global_phase_ref().bit(r)) {
            angle = angle.rotated180();
        }
        bool bit = angle.is_closer_to_half_turn_than_no_turn();
        uint8_t &out_byte = out[k / 8];
        if (k % 8 == 0) {
            out_byte = 0;
        }
        out_byte |= uint8_t{bit} << (k % 8);
    }
    return int_obj_from_bytes_with_max_bits(self.byte_buf.data(), self.batch_size);
}

pybind11::object peek_phase_flipped_across_shots_np_bool(const PyInteractiveSimulator &self, pybind11::object out) {
    size_t n1 = self.batch_size;
    if (out.is_none()) {
        auto numpy = pybind11::module::import("numpy");
        out = numpy.attr("empty")(pybind11::make_tuple(n1), numpy.attr("bool_"));
    }

    if (!pybind11::isinstance<pybind11::array_t<bool>>(out)) {
        throw std::invalid_argument("out wasn't a np.ndarray[np.bool_].");
    }
    auto buf = pybind11::cast<pybind11::array_t<bool>>(out);
    if (buf.ndim() != 1) {
        throw std::invalid_argument("len(out.shape) != 1");
    }
    if ((size_t)buf.shape(0) != n1) {
        std::stringstream ss;
        ss << "Expected out.shape == (" << n1 << ",)";
        ss << " but got out.shape=(" << buf.shape(0) << ",).";
        throw std::invalid_argument(ss.str());
    }

    auto s1 = buf.strides(0);
    auto out_ptr = buf.mutable_data(0);
    size_t out_bit = 0;
    for (const auto &sim : self.simulators) {
        for (size_t k = 0; k < PY_SIM_WORD_BITS; k++) {
            if (out_bit >= self.batch_size) {
                break;
            }
            auto angle = sim.angles[k];
            if (sim.global_phase_ref().bit(k)) {
                angle = angle.rotated180();
            }

            bool bit = angle.is_closer_to_half_turn_than_no_turn();
            *out_ptr = bit;
            out_ptr += s1;
            out_bit++;
        }
    }
    return out;
}

pybind11::object read_within_shot(
    PyInteractiveSimulator &self,
    const pybind11::object &indices_obj,
    const pybind11::object &shot_index_obj,
    const pybind11::object &out) {
    auto indices_conv = index_to_sim_items(self, indices_obj);
    auto indices = indices_conv.span.checked_cast_to_qubit_or_bit_or_bool("indices");
    auto shot_index = pybind11::cast<size_t>(shot_index_obj);

    if (out.ptr() == (PyObject *)&PyLong_Type) {
        if (indices_conv.was_singleton) {
            return peek_within_shot_1d_int(self, indices[0], shot_index);
        } else {
            return peek_within_shot_2d_int(self, indices, shot_index);
        }
    }

    if (indices_conv.was_singleton) {
        return peek_within_shot_1d_np_bool(self, indices[0], shot_index, out);
    } else {
        return peek_within_shot_2d_np_bool(self, indices, shot_index, out);
    }
}

void write_within_shot_1d_int(
    PyInteractiveSimulator &self,
    const stride_span<const QubitOrBit> &items,
    size_t shot,
    const pybind11::object &new_value) {
    for (auto q : items) {
        self.ensure_big_enough_state_for(q);
    }
    // byte_buf is initially sized for batch_size bits, but this writes len(indices) bits into it.
    self.ensure_byte_buf_can_store_bits(items.size());
    int_obj_to_bytes_with_max_bits(new_value, self.byte_buf.data(), items.size(), "new_value", "len(indices)");
    for (size_t k = 0; k < items.size(); k++) {
        set_single_bit(self, items[k], shot, (self.byte_buf[k / 8] >> (k % 8)) & 1);
    }
}
void write_within_shot_1d_np_bool(
    PyInteractiveSimulator &self,
    const stride_span<const QubitOrBit> &indices,
    size_t shot,
    const pybind11::object &new_value) {
    for (auto q : indices) {
        self.ensure_big_enough_state_for(q);
    }

    auto array = pybind11::cast<pybind11::array_t<bool>>(new_value);
    if (array.ndim() != 1) {
        throw std::invalid_argument("len(new_value.shape) != 1");
    }
    if ((size_t)array.shape(0) != indices.size()) {
        std::stringstream ss;
        ss << "Expected new_value.shape == (" << indices.size() << ",)";
        ss << " but got new_value.shape=(" << array.shape(0) << ",).";
        throw std::invalid_argument(ss.str());
    }
    auto stride = array.strides(0);
    auto in_ptr = array.data(0);
    for (size_t k = 0; k < indices.size(); k++) {
        set_single_bit(self, indices[k], shot, *in_ptr);
        in_ptr += stride;
    }
}

void write_within_shot(
    PyInteractiveSimulator &self,
    const pybind11::object &indices_obj,
    const pybind11::object &shot_index_obj,
    const pybind11::object &new_value) {
    auto indices_conv = index_to_sim_items(self, indices_obj);
    auto indices = indices_conv.span.checked_cast_to_qubit_or_bit("indices");
    auto shot_index = pybind11::cast<size_t>(shot_index_obj);

    auto type_ptr = Py_TYPE(new_value.ptr());
    if (type_ptr == &PyLong_Type || type_ptr == &PyBool_Type) {
        write_within_shot_1d_int(self, indices, shot_index, new_value);
    } else if (pybind11::isinstance<pybind11::array_t<bool>>(new_value)) {
        write_within_shot_1d_np_bool(self, indices, shot_index, new_value);
    } else {
        std::stringstream ss;
        ss << "Don't know how to write from ";
        ss << pybind11::repr(new_value);
        ss << " (expected an `int` or `np.ndarray` with `dtype=np.bool_`)";
        throw std::invalid_argument(ss.str());
    }
}

void kickmix_py::register_interactive_simulator_methods(pybind11::class_<PyInteractiveSimulator> &c_sim) {
    c_sim.def(
        pybind11::init([](size_t batch_size) -> PyInteractiveSimulator {
            return PyInteractiveSimulator(batch_size);
        }),
        pybind11::arg("batch_size"),
        clean_doc_string(R"DOC(
            @signature def __init__(batch_size: int):
            Initializes an interactive simulator with the given batch size.

            Args:
                batch_size: Determines how many simultaneous shots are being tracked by the
                    simulator.
        )DOC")
            .data());

    c_sim.def("use_same_registers_as", [](PyInteractiveSimulator &self, const Circuit &circuit) {
        self.register_data = circuit.register_data;
    });

    c_sim.def(
        "do",
        [](PyInteractiveSimulator &self, const Circuit &circuit) {
            self.ensure_big_enough_state_for(circuit.num_qubits, circuit.num_bits);
            for (auto &sim : self.simulators) {
                sim.apply(circuit);
            }
        },
        pybind11::arg("circuit"),
        clean_doc_string(R"DOC(
            @signature def do(self, circuit: km.Circuit):
            Applies the instructions from the circuit to the simulator's state.

            Args:
                circuit: The circuit with instructions to apply.
        )DOC")
            .data());

    c_sim.def_property_readonly(
        "batch_size",
        [](PyInteractiveSimulator &self) {
            return self.batch_size;
        },
        clean_doc_string(R"DOC(
            The number of shots being tracked by the simulator.
        )DOC")
            .data());

    c_sim.def(
        "clear_for_shot",
        &sim_clear_for_shot,
        clean_doc_string(R"DOC(
            Zeroes all data that should be reset between shots from a circuit.

            When using the simulator to repeatedly sample from a circuit, it is important
            to call clear_for_shot between calls. Otherwise state modified by an earlier
            shot may end up affecting a later shot.

            This method clears:
            - tracked qubit values
            - tracked bit values
            - tracked phase values
            - the condition stack

            Examples:
                >>> import kickmix as km
                >>> sim = km.Simulator(batch_size=8)
                >>> sim.do(km.Circuit('''
                ...     X q0
                ...     PUSH_CONDITION if b0
                ... '''))
                >>> sim.read_across_shots(km.q(0), out=int)
                255

                >>> sim.clear_for_shot()
                >>> sim.read_across_shots(km.q(0), out=int)
                0
        )DOC")
            .data());

    c_sim.def(
        "read_shot_phase",
        &sim_read_phase,
        pybind11::arg("shot_index"),
        clean_doc_string(R"DOC(
            Returns the tracked global phase of a shot.

            The phase is affected by operations like Z, HMR, and Z_POW.

            Returns:
                The phase, in half turns, as an int or as a `fractions.Fraction`.

                If the shot is exactly unphased (0 radians), the int 0 is returned.
                If the shot is exactly phase flipped (pi radians), the int 1 is returned.
                If the phase is some other value, a fractions.Fraction with the exact
                phase value (in half turns) is returned.

            Examples:
                >>> import kickmix as km
                >>> sim = km.Simulator(batch_size=5)
                >>> sim.read_shot_phase(shot_index=4)
                0

                >>> sim.do(km.Circuit('''
                ...     X q0
                ...     Z q0
                ... '''))
                >>> sim.read_shot_phase(shot_index=4)
                1

                >>> sim.do(km.Circuit('''
                ...     Z_POW q0 1.25000000000000001387778780781445675529539585113525390625
                ... '''))
                >>> sim.read_shot_phase(shot_index=4)
                Fraction(18014398509481985, 72057594037927936)
        )DOC")
            .data());

    c_sim.def(
        "write_shot_phase",
        &sim_write_phase,
        pybind11::arg("shot_index"),
        pybind11::arg("new_phase_half_turns"),
        clean_doc_string(R"DOC(
            Sets the tracked global phase of a shot.

            The phase is affected by operations like Z, HMR, and Z_POW.

            Args:
                shot_index: The shot whose phase is being written.
                new_phase_half_turns: The new phase, as a rotation in half turn units.
                    This can be an int, a float, or a fractions.Fraction.
                    The value will be canonicalized into the range [0, 2) and then
                    rounded to the nearest multiple of 2**-127.

            Examples:
                >>> import kickmix as km
                >>> sim = km.Simulator(batch_size=5)

                >>> sim.write_shot_phase(4, 0.125)
                >>> sim.read_shot_phase(4)
                Fraction(1, 8)

                >>> sim.write_shot_phase(4, 1.0)
                >>> sim.read_shot_phase(4)
                1
        )DOC")
            .data());

    c_sim.def(
        "read_phase_flipped_across_shots",
        [](PyInteractiveSimulator &self, const pybind11::object &out) -> pybind11::object {
            if (out.ptr() == (PyObject *)&PyLong_Type) {
                return read_phase_flipped_across_shots_int(self);
            }
            return peek_phase_flipped_across_shots_np_bool(self, out);
        },
        pybind11::kw_only(),
        pybind11::arg("out") = pybind11::none(),
        clean_doc_string(R"DOC(
            Reads whether or not each shot is phase flipped.

            A shot is considered phase flipped if its tracked phase is closer to a
            half turn than to no turns.

            Args:
                out: Where to store the output. Defaults to None (allocate a new numpy
                    array). Can be set to a numpy array to write the results into, or
                    to `int` to return the result bit packed into an integer.

            Returns:
                The phase flip information.
         )DOC")
            .data());

    c_sim.def(
        "read_across_shots",
        [](PyInteractiveSimulator &self,
           const pybind11::object &indices_obj,
           const pybind11::object &out) -> pybind11::object {
            auto indices_conv = index_to_sim_items(self, indices_obj);
            auto indices = indices_conv.span.checked_cast_to_qubit_or_bit_or_bool("indices");

            if (out.ptr() == (PyObject *)&PyLong_Type) {
                if (indices_conv.was_singleton) {
                    return peek_across_shots_1d_int(self, indices[0]);
                } else {
                    return peek_across_shots_2d_int(self, indices);
                }
            }

            if (indices_conv.was_singleton) {
                return peek_across_shots_1d_np_bool(self, indices[0], out);
            } else {
                return peek_across_shots_2d_np_bool(self, indices, out);
            }
        },
        pybind11::arg("indices"),
        pybind11::kw_only(),
        pybind11::arg("out") = pybind11::none(),
        clean_doc_string(R"DOC(
         )DOC")
            .data());

    c_sim.def(
        "read_within_shot",
        &read_within_shot,
        pybind11::arg("indices"),
        pybind11::arg("shot_index"),
        pybind11::kw_only(),
        pybind11::arg("out") = pybind11::none(),
        clean_doc_string(R"DOC(
         )DOC")
            .data());

    c_sim.def(
        "write_across_shots",
        &write_across_shots,
        pybind11::arg("index"),
        pybind11::arg("new_value"),
        clean_doc_string(R"DOC(
            Sets the tracked value of the given qubits or bits, across all shots.

            Args:
                index: The qubit id or bit id to write to, or an array of them
                    (e.g. a register name).
                new_value: The value to write.

                    If `index` is a single qubit or bit, this is an `int` or an
                    `np.ndarray` with dtype=np.bool_ and shape=(sim.batch_size,).

                    If `index` is an array of length n, this is a sequence of n
                    `int`s or an `np.ndarray` with dtype=np.bool_ and
                    shape=(n, sim.batch_size).

                    Each `int` is bit packed in little-endian order, so the bit
                    for shot k is `(new_value >> k) & 1`. This matches the
                    layout returned by `read_across_shots(..., out=int)`.

            Raises:
                IndexError: The given index isn't a qubit id or bit id.
                ValueError: The given new_value isn't valid.
        )DOC")
            .data());

    c_sim.def(
        "write_within_shot",
        &write_within_shot,
        pybind11::arg("index"),
        pybind11::arg("shot_index"),
        pybind11::arg("new_value"),
        clean_doc_string(R"DOC(
        )DOC")
            .data());
}
