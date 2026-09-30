#include "kickmix/py/circuit/subs/gf_arithmetic.pybind.h"

#include <cstring>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <vector>

#include "kickmix/gen/gf_arithmetic/gen_gf_div.h"
#include "kickmix/gen/gf_arithmetic/gen_gf_iadd.h"
#include "kickmix/gen/gf_arithmetic/gen_gf_imul_classical.h"
#include "kickmix/gen/gf_arithmetic/gen_gf_inverse.h"
#include "kickmix/gen/gf_arithmetic/gen_gf_mul.h"
#include "kickmix/py/circuit/control_helper.pybind.h"
#include "kickmix/py/util.pybind.h"
#include "kickmix/py/val/array.pybind.h"
#include "kickmix/py/val/converted_array.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

namespace {

GF2Poly py_int_to_gf2_poly(const pybind11::int_ &val, const char *name) {
    if (val < pybind11::int_(0)) {
        std::stringstream ss;
        ss << name << " must be non-negative, got " << pybind11::repr(val);
        throw std::invalid_argument(ss.str());
    }
    size_t bit_len = pybind11::cast<size_t>(val.attr("bit_length")());
    if (bit_len > GF2Poly::MAX_BITS) {
        std::stringstream ss;
        ss << name << " has bit_length " << bit_len << " > " << GF2Poly::MAX_BITS;
        throw std::invalid_argument(ss.str());
    }
    if (bit_len <= 64) {
        return GF2Poly::from_u64(pybind11::cast<uint64_t>(val));
    }
    GF2Poly result;
    size_t num_bytes = (bit_len + 7) / 8;
    pybind11::bytes bytes_obj = val.attr("to_bytes")(num_bytes, "little");
    std::string_view bytes_view = bytes_obj;
    std::memcpy(result.words.data(), bytes_view.data(), num_bytes);
    return result;
}

pybind11::int_ gf2_poly_to_py_int(const GF2Poly &poly) {
    size_t bits = poly.num_bits_in_use();
    if (bits <= 64) {
        return pybind11::int_(poly.words[0]);
    }
    size_t num_bytes = (bits + 7) / 8;
    pybind11::bytes bytes_obj(reinterpret_cast<const char *>(poly.words.data()), num_bytes);
    return pybind11::int_(0).attr("from_bytes")(bytes_obj, "little");
}

GF2Field make_checked_gf2_field(size_t degree, const pybind11::int_ &modulus_int, const char *name) {
    GF2Poly modulus = py_int_to_gf2_poly(modulus_int, name);
    GF2Field field(degree, modulus);
    if (!gf2_is_irreducible(field.modulus())) {
        throw std::invalid_argument("GF2Field: modulus is not irreducible over GF(2)");
    }
    return field;
}

GF2Field resolve_gf2_field(const pybind11::object &field_obj, size_t expected_degree) {
    if (field_obj.is_none()) {
        return GF2Field(expected_degree);
    }
    if (pybind11::isinstance<GF2Field>(field_obj)) {
        const GF2Field &field = field_obj.cast<const GF2Field &>();
        if (field.degree() != expected_degree) {
            std::stringstream ss;
            ss << "field.degree (" << field.degree() << ") != len(target) (" << expected_degree << ")";
            throw std::invalid_argument(ss.str());
        }
        return field;
    }
    if (pybind11::isinstance<pybind11::int_>(field_obj) && !pybind11::isinstance<pybind11::bool_>(field_obj)) {
        return make_checked_gf2_field(expected_degree, pybind11::cast<pybind11::int_>(field_obj), "field");
    }
    throw std::invalid_argument("field must be None, an int modulus, or a km.GF2Field.");
}

// Allocates `n` fresh qubits and returns them as an owning km.array object.
pybind11::object alloc_qubit_array_obj(PyCircuitBuilder &self, size_t n) {
    if (self.allocated_qubits + n > self.max_qubits) {
        std::stringstream ss;
        ss << "Attempted to allocate more qubits than the maximum.";
        ss << "\n    allocated: " << self.allocated_qubits;
        ss << "\n    requested: " << n;
        ss << "\n    allocated+requested: " << (self.allocated_qubits + n);
        ss << "\n    max: " << self.max_qubits;
        throw std::invalid_argument(ss.str());
    }
    array_xz arr = array_xz::alloc_noinit(n, QXZTypeTag8::QUBIT_ID);
    for (size_t k = 0; k < n; k++) {
        arr[k] = self.alloc_qubit();
    }
    PyArrayXZ wrapped;
    wrapped.owner = std::make_shared<array_xz>(std::move(arr));
    wrapped.view = *wrapped.owner;
    return pybind11::cast(wrapped);
}

// Resolves an `init_` target, which may be the string "alloc" asking the
// builder to allocate an `n` qubit register instead of supplying one.
pybind11::object resolve_init_target_obj(PyCircuitBuilder &self, const pybind11::object &target_obj, size_t n) {
    if (pybind11::isinstance<pybind11::str>(target_obj)) {
        auto text = pybind11::cast<std::string>(target_obj);
        if (text != "alloc") {
            std::stringstream ss;
            ss << R"(target must be a register or the string "alloc", got )" << pybind11::repr(target_obj);
            throw std::invalid_argument(ss.str());
        }
        return alloc_qubit_array_obj(self, n);
    }
    return target_obj;
}

}  // namespace

pybind11::class_<GF2Field> kickmix_py::register_gf2_field_class(pybind11::module &m) {
    return pybind11::class_<GF2Field>(
        m,
        "GF2Field",
        clean_doc_string(R"DOC(
            A binary extension field GF(2^m) with polynomial basis arithmetic.

            Elements are polynomials of degree less than m over GF(2), represented
            in Python as non-negative integers where bit k is the coefficient of x^k.
            Addition is bitwise XOR, and multiplication is polynomial multiplication
            reduced modulo an irreducible polynomial of degree m.

            Examples:
                >>> import kickmix as km
                >>> field = km.GF2Field(4)
                >>> field.degree
                4
                >>> hex(field.modulus)
                '0x13'
                >>> field.mul(3, 5)
                15
        )DOC")
            .data());
}

void kickmix_py::register_gf2_field_methods(pybind11::class_<GF2Field> &c) {
    c.def(
        pybind11::init([](size_t degree, const pybind11::object &modulus_obj) -> GF2Field {
            if (modulus_obj.is_none()) {
                return GF2Field(degree);
            }
            if (pybind11::isinstance<pybind11::int_>(modulus_obj) &&
                !pybind11::isinstance<pybind11::bool_>(modulus_obj)) {
                return make_checked_gf2_field(degree, pybind11::cast<pybind11::int_>(modulus_obj), "modulus");
            }
            throw std::invalid_argument("modulus must be None or an int.");
        }),
        pybind11::arg("degree"),
        pybind11::arg("modulus") = pybind11::none(),
        clean_doc_string(R"DOC(
            @signature def __init__(self, degree: int, modulus: int | None = None) -> None:
            Creates the binary extension field GF(2^degree).

            Args:
                degree: The extension degree m (1 <= m <= 512).
                modulus: Optional irreducible reduction polynomial of degree m,
                    encoded as an int where bit k is the coefficient of x^k. If
                    None, a low-weight default irreducible polynomial is chosen.

            Examples:
                >>> import kickmix as km
                >>> field = km.GF2Field(4)
                >>> field.degree
                4
                >>> hex(field.modulus)
                '0x13'
                >>> custom_field = km.GF2Field(4, modulus=0x19)
                >>> hex(custom_field.modulus)
                '0x19'
        )DOC")
            .data());

    c.def_property_readonly(
        "degree",
        [](const GF2Field &self) -> size_t {
            return self.degree();
        },
        clean_doc_string(R"DOC(
            The extension degree m of the field GF(2^m).
        )DOC")
            .data());

    c.def_property_readonly(
        "modulus",
        [](const GF2Field &self) -> pybind11::int_ {
            return gf2_poly_to_py_int(self.modulus());
        },
        clean_doc_string(R"DOC(
            The irreducible reduction polynomial of degree m, including the x^m bit.
        )DOC")
            .data());

    c.def(
        "add",
        [](const GF2Field &self, const pybind11::int_ &a, const pybind11::int_ &b) -> pybind11::int_ {
            GF2Poly pa = self.mod(py_int_to_gf2_poly(a, "a"));
            GF2Poly pb = self.mod(py_int_to_gf2_poly(b, "b"));
            return gf2_poly_to_py_int(self.add(pa, pb));
        },
        pybind11::arg("a"),
        pybind11::arg("b"),
        clean_doc_string(R"DOC(
            @signature def add(self, a: int, b: int) -> int:
            Returns (a + b) mod modulus in GF(2^m), which is bitwise XOR.

            Examples:
                >>> import kickmix as km
                >>> field = km.GF2Field(4)
                >>> field.add(3, 5)
                6
        )DOC")
            .data());

    c.def(
        "mul",
        [](const GF2Field &self, const pybind11::int_ &a, const pybind11::int_ &b) -> pybind11::int_ {
            GF2Poly pa = py_int_to_gf2_poly(a, "a");
            GF2Poly pb = py_int_to_gf2_poly(b, "b");
            return gf2_poly_to_py_int(self.mul(pa, pb));
        },
        pybind11::arg("a"),
        pybind11::arg("b"),
        clean_doc_string(R"DOC(
            @signature def mul(self, a: int, b: int) -> int:
            Returns the product (a * b) mod modulus in GF(2^m).

            Examples:
                >>> import kickmix as km
                >>> field = km.GF2Field(4)
                >>> field.mul(3, 5)
                15
        )DOC")
            .data());

    c.def(
        "square",
        [](const GF2Field &self, const pybind11::int_ &a) -> pybind11::int_ {
            GF2Poly pa = py_int_to_gf2_poly(a, "a");
            return gf2_poly_to_py_int(self.square(pa));
        },
        pybind11::arg("a"),
        clean_doc_string(R"DOC(
            @signature def square(self, a: int) -> int:
            Returns a^2 mod modulus in GF(2^m).

            Examples:
                >>> import kickmix as km
                >>> field = km.GF2Field(4)
                >>> field.square(3)
                5
        )DOC")
            .data());

    c.def(
        "frobenius",
        [](const GF2Field &self, const pybind11::int_ &a, size_t k) -> pybind11::int_ {
            GF2Poly pa = py_int_to_gf2_poly(a, "a");
            return gf2_poly_to_py_int(self.frobenius(pa, k));
        },
        pybind11::arg("a"),
        pybind11::arg("k") = 1,
        clean_doc_string(R"DOC(
            @signature def frobenius(self, a: int, k: int = 1) -> int:
            Returns a^(2^k) mod modulus in GF(2^m).

            Examples:
                >>> import kickmix as km
                >>> field = km.GF2Field(4)
                >>> field.frobenius(3, 1)
                5
                >>> field.frobenius(3, 4) == 3  # in GF(2^4), a^(2^4) == a
                True
        )DOC")
            .data());

    c.def(
        "pow",
        [](const GF2Field &self, const pybind11::int_ &a, pybind11::int_ exp_int) -> pybind11::int_ {
            GF2Poly pa = self.mod(py_int_to_gf2_poly(a, "a"));
            if (exp_int < pybind11::int_(0)) {
                pa = self.invert(pa);
                exp_int = exp_int.attr("__neg__")();
            }
            size_t bit_len = pybind11::cast<size_t>(exp_int.attr("bit_length")());
            if (bit_len <= 64) {
                return gf2_poly_to_py_int(self.pow(pa, pybind11::cast<uint64_t>(exp_int)));
            }
            size_t num_bytes = (bit_len + 7) / 8;
            pybind11::bytes bytes_obj = exp_int.attr("to_bytes")(num_bytes, "little");
            std::string_view bytes_view = bytes_obj;
            FixedWidthInt fwi(bit_len);
            for (size_t k = 0; k < bit_len; k++) {
                fwi.bit_ref(k) = ((uint8_t)bytes_view[k / 8] >> (k % 8)) & 1;
            }
            return gf2_poly_to_py_int(self.pow(pa, fwi));
        },
        pybind11::arg("a"),
        pybind11::arg("exponent"),
        clean_doc_string(R"DOC(
            @signature def pow(self, a: int, exponent: int) -> int:
            Returns a**exponent mod modulus in GF(2^m). Negative exponents invert a.

            Examples:
                >>> import kickmix as km
                >>> field = km.GF2Field(4)
                >>> field.pow(3, 2)
                5
                >>> field.pow(3, -1)
                14
                >>> field.mul(3, 14)
                1
        )DOC")
            .data());

    c.def(
        "invert",
        [](const GF2Field &self, const pybind11::int_ &a) -> pybind11::int_ {
            GF2Poly pa = py_int_to_gf2_poly(a, "a");
            return gf2_poly_to_py_int(self.invert(pa));
        },
        pybind11::arg("a"),
        clean_doc_string(R"DOC(
            @signature def invert(self, a: int) -> int:
            Returns the multiplicative inverse of a in GF(2^m), or 0 when a is 0.

            Examples:
                >>> import kickmix as km
                >>> field = km.GF2Field(4)
                >>> inv = field.invert(3)
                >>> inv
                14
                >>> field.mul(3, inv)
                1
                >>> field.invert(0)
                0
        )DOC")
            .data());

    c.def(
        "div",
        [](const GF2Field &self, const pybind11::int_ &a, const pybind11::int_ &b) -> pybind11::int_ {
            GF2Poly pa = py_int_to_gf2_poly(a, "a");
            GF2Poly pb = py_int_to_gf2_poly(b, "b");
            return gf2_poly_to_py_int(self.div(pa, pb));
        },
        pybind11::arg("a"),
        pybind11::arg("b"),
        clean_doc_string(R"DOC(
            @signature def div(self, a: int, b: int) -> int:
            Returns (a / b) mod modulus in GF(2^m), or 0 when b is 0.

            Examples:
                >>> import kickmix as km
                >>> field = km.GF2Field(4)
                >>> q = field.div(6, 3)
                >>> q
                2
                >>> field.mul(q, 3)
                6
                >>> field.div(6, 0)
                0
        )DOC")
            .data());

    c.def(
        "mod",
        [](const GF2Field &self, const pybind11::int_ &a) -> pybind11::int_ {
            GF2Poly pa = py_int_to_gf2_poly(a, "a");
            return gf2_poly_to_py_int(self.mod(pa));
        },
        pybind11::arg("a"),
        clean_doc_string(R"DOC(
            @signature def mod(self, a: int) -> int:
            Reduces a polynomial mod the field's irreducible polynomial.

            Examples:
                >>> import kickmix as km
                >>> field = km.GF2Field(4)
                >>> field.mod(0x13)  # reduction modulo modulus (0x13)
                0
                >>> field.mod(0x15)
                6
        )DOC")
            .data());

    c.def(
        "is_element",
        [](const GF2Field &self, const pybind11::int_ &a) -> bool {
            if (a < pybind11::int_(0)) {
                return false;
            }
            size_t bit_len = pybind11::cast<size_t>(a.attr("bit_length")());
            return bit_len <= self.degree();
        },
        pybind11::arg("a"),
        clean_doc_string(R"DOC(
            @signature def is_element(self, a: int) -> bool:
            Returns True if a is a valid element of GF(2^m) (0 <= a < 2**degree).

            Examples:
                >>> import kickmix as km
                >>> field = km.GF2Field(4)
                >>> field.is_element(15)
                True
                >>> field.is_element(16)
                False
                >>> field.is_element(-1)
                False
        )DOC")
            .data());

    c.def_static(
        "is_irreducible",
        [](const pybind11::int_ &poly_int) -> bool {
            GF2Poly poly = py_int_to_gf2_poly(poly_int, "poly");
            return gf2_is_irreducible(poly);
        },
        pybind11::arg("poly"),
        clean_doc_string(R"DOC(
            @signature def is_irreducible(poly: int) -> bool:
            Returns True if poly is irreducible over GF(2).

            Examples:
                >>> import kickmix as km
                >>> km.GF2Field.is_irreducible(0x13)  # x^4 + x + 1
                True
                >>> km.GF2Field.is_irreducible(0x15)  # x^4 + x^2 + 1 = (x^2 + x + 1)^2
                False
        )DOC")
            .data());

    c.def(
        "__eq__",
        [](const GF2Field &self, const pybind11::object &other) -> bool {
            if (!pybind11::isinstance<GF2Field>(other)) {
                return false;
            }
            const GF2Field &o = other.cast<const GF2Field &>();
            return self.degree() == o.degree() && self.modulus() == o.modulus();
        },
        pybind11::arg("other"),
        clean_doc_string(R"DOC(
            @signature def __eq__(self, other: object) -> bool:
            Returns True if other is a GF2Field with the same degree and modulus.
        )DOC")
            .data());

    c.def(
        "__ne__",
        [](const GF2Field &self, const pybind11::object &other) -> bool {
            if (!pybind11::isinstance<GF2Field>(other)) {
                return true;
            }
            const GF2Field &o = other.cast<const GF2Field &>();
            return self.degree() != o.degree() || self.modulus() != o.modulus();
        },
        pybind11::arg("other"),
        clean_doc_string(R"DOC(
            @signature def __ne__(self, other: object) -> bool:
            Returns True if other is not a GF2Field with the same degree and modulus.
        )DOC")
            .data());

    c.def(
        "__hash__",
        [](const GF2Field &self) -> size_t {
            size_t h = self.degree();
            for (uint64_t w : self.modulus().words) {
                h ^= std::hash<uint64_t>{}(w) + 0x9e3779b9 + (h << 6) + (h >> 2);
            }
            return h;
        },
        clean_doc_string(R"DOC(
            @signature def __hash__(self) -> int:
            Returns a hash of the field's degree and modulus.
        )DOC")
            .data());

    c.def(
        "__repr__",
        [](const GF2Field &self) -> std::string {
            std::stringstream ss;
            ss << "km.GF2Field(" << self.degree();
            if (self.modulus() != gf2_default_irreducible_poly(self.degree())) {
                ss << ", modulus=" << self.modulus().str();
            }
            ss << ")";
            return ss.str();
        },
        clean_doc_string(R"DOC(
            @signature def __repr__(self) -> str:
            Returns a string representation of the field that evaluates to an equal field.
        )DOC")
            .data());

    c.def(
        "__str__",
        [](const GF2Field &self) -> std::string {
            std::stringstream ss;
            ss << "GF(2^" << self.degree() << ", modulus=" << self.modulus().str() << ")";
            return ss.str();
        },
        clean_doc_string(R"DOC(
            @signature def __str__(self) -> str:
            Returns a human-readable summary of the field.
        )DOC")
            .data());
}

void kickmix_py::append_gf2_iadd_obj(
    PyCircuitBuilder &self,
    const pybind11::object &offset_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj,
    const pybind11::object &control_obj) {
    auto target_conv = ConvertedArrayXZ::from_obj(target_obj, "target");
    auto target = target_conv.span.checked_cast_to_qubit_ids("target");
    if (!field_obj.is_none()) {
        (void)resolve_gf2_field(field_obj, target.size());
    }

    auto offset_conv =
        ConvertedArrayXZ::from_obj_or_int(offset_obj, target.size(), "offset", /*allow_twos_complement=*/false);
    auto offset = offset_conv.span.checked_cast_to_qubit_or_bit_or_bool("offset");
    throw_unless(target.size() == offset.size(), "gen_gf_iadd: Q_target.size() != Q_offset.size()");

    RaiiControlObjHelper control(self, control_obj, "control");
    if (!control.skip) {
        gen_gf_iadd(self.builder, CircuitGenCtx{{}}, target, offset, control);
    }
}

void kickmix_py::append_gf2_imul_obj(
    PyCircuitBuilder &self,
    const pybind11::object &constant_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj) {
    if (!pybind11::isinstance<pybind11::int_>(constant_obj) || pybind11::isinstance<pybind11::bool_>(constant_obj)) {
        PyErr_SetString(
            PyExc_NotImplementedError, "gf2_imul currently only supports classical constant integer factors.");
        throw pybind11::error_already_set();
    }
    auto target_conv = ConvertedArrayXZ::from_obj(target_obj, "target");
    auto target = target_conv.span.checked_cast_to_qubit_ids("target");
    GF2Field field = resolve_gf2_field(field_obj, target.size());
    GF2Poly constant_poly = py_int_to_gf2_poly(pybind11::reinterpret_borrow<pybind11::int_>(constant_obj), "constant");
    gen_gf_imul_classical(self.builder, CircuitGenCtx{{}}, field, target, constant_poly);
}

void kickmix_py::append_gf2_idiv_obj(
    PyCircuitBuilder &self,
    const pybind11::object &constant_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj) {
    if (!pybind11::isinstance<pybind11::int_>(constant_obj) || pybind11::isinstance<pybind11::bool_>(constant_obj)) {
        PyErr_SetString(
            PyExc_NotImplementedError, "gf2_idiv currently only supports classical constant integer divisors.");
        throw pybind11::error_already_set();
    }
    auto target_conv = ConvertedArrayXZ::from_obj(target_obj, "target");
    auto target = target_conv.span.checked_cast_to_qubit_ids("target");
    GF2Field field = resolve_gf2_field(field_obj, target.size());
    GF2Poly constant_poly = py_int_to_gf2_poly(pybind11::reinterpret_borrow<pybind11::int_>(constant_obj), "constant");
    gen_gf_idiv_classical(self.builder, CircuitGenCtx{{}}, field, target, constant_poly);
}

pybind11::object kickmix_py::append_init_gf2_mul_obj(
    PyCircuitBuilder &self,
    const pybind11::object &lhs_obj,
    const pybind11::object &rhs_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj,
    const pybind11::object &control_obj) {
    // Resolve the field from lhs rather than target, so that the target can be
    // allocated from the field degree when target is "alloc".
    auto lhs_conv = ConvertedArrayXZ::from_obj(lhs_obj, "lhs");
    auto lhs = lhs_conv.span.checked_cast_to_qubit_ids("lhs");
    GF2Field field = resolve_gf2_field(field_obj, lhs.size());
    pybind11::object resolved = resolve_init_target_obj(self, target_obj, field.degree());

    auto target_conv = ConvertedArrayXZ::from_obj(resolved, "target");
    auto target = target_conv.span.checked_cast_to_qubit_ids("target");
    self.builder.broadcast_reset(target);
    append_ixor_gf2_mul_obj(self, lhs_obj, rhs_obj, resolved, field_obj, control_obj);
    return resolved;
}

void kickmix_py::append_ixor_gf2_mul_obj(
    PyCircuitBuilder &self,
    const pybind11::object &lhs_obj,
    const pybind11::object &rhs_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj,
    const pybind11::object &control_obj) {
    auto target_conv = ConvertedArrayXZ::from_obj(target_obj, "target");
    auto lhs_conv = ConvertedArrayXZ::from_obj(lhs_obj, "lhs");
    auto rhs_conv = ConvertedArrayXZ::from_obj(rhs_obj, "rhs");
    auto target = target_conv.span.checked_cast_to_qubit_ids("target");
    auto lhs = lhs_conv.span.checked_cast_to_qubit_ids("lhs");
    auto rhs = rhs_conv.span.checked_cast_to_qubit_ids("rhs");
    GF2Field field = resolve_gf2_field(field_obj, target.size());
    size_t m = field.degree();
    throw_unless(lhs.size() == m, "gen_gf_mul: Q_lhs.size() != field.degree()");
    throw_unless(rhs.size() == m, "gen_gf_mul: Q_rhs.size() != field.degree()");

    RaiiControlObjHelper control(self, control_obj, "control");
    if (control.skip) {
        return;
    }
    QubitOrTrue ctrl_val = control;

    size_t needed_clean = ctrl_val.is_qubit() ? m : 0;
    size_t num_clean = self.num_free_qubits() >= needed_clean ? needed_clean : 0;
    array_z clean = array_z::alloc_noinit(num_clean, QXZTypeTag8::QUBIT_ID);
    for (size_t k = 0; k < clean.size(); k++) {
        clean[k] = self.alloc_qubit();
    }
    std::span<const QubitId> clean_span{(QubitId *)clean.items, clean.size()};
    gen_gf_mul(self.builder, CircuitGenCtx{clean_span}, field, target, lhs, rhs, ctrl_val);
    for (size_t k = clean.size(); k--;) {
        self.free_qubit((QubitId)clean[k]);
    }
}

void kickmix_py::append_del_gf2_mul_obj(
    PyCircuitBuilder &self,
    const pybind11::object &lhs_obj,
    const pybind11::object &rhs_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj,
    const pybind11::object &control_obj) {
    auto target_conv = ConvertedArrayXZ::from_obj(target_obj, "target");
    auto lhs_conv = ConvertedArrayXZ::from_obj(lhs_obj, "lhs");
    auto rhs_conv = ConvertedArrayXZ::from_obj(rhs_obj, "rhs");
    auto target = target_conv.span.checked_cast_to_qubit_ids("target");
    auto lhs = lhs_conv.span.checked_cast_to_qubit_ids("lhs");
    auto rhs = rhs_conv.span.checked_cast_to_qubit_ids("rhs");
    GF2Field field = resolve_gf2_field(field_obj, target.size());
    size_t m = field.degree();
    throw_unless(lhs.size() == m, "gen_gf_unmul: Q_lhs.size() != field.degree()");
    throw_unless(rhs.size() == m, "gen_gf_unmul: Q_rhs.size() != field.degree()");

    RaiiControlObjHelper control(self, control_obj, "control");
    if (control.skip) {
        return;
    }
    QubitOrTrue ctrl_val = control;

    size_t needed_clean = ctrl_val.is_qubit() ? m : 0;
    size_t num_clean = self.num_free_qubits() >= needed_clean ? needed_clean : 0;
    array_z clean = array_z::alloc_noinit(num_clean, QXZTypeTag8::QUBIT_ID);
    for (size_t k = 0; k < clean.size(); k++) {
        clean[k] = self.alloc_qubit();
    }
    std::span<const QubitId> clean_span{(QubitId *)clean.items, clean.size()};
    gen_gf_unmul(self.builder, CircuitGenCtx{clean_span}, field, target, lhs, rhs, ctrl_val);
    for (size_t k = clean.size(); k--;) {
        self.free_qubit((QubitId)clean[k]);
    }
}

pybind11::object kickmix_py::append_init_gf2_inverse_obj(
    PyCircuitBuilder &self,
    const pybind11::object &input_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj) {
    // Resolve the field from input rather than target, so that the target can
    // be allocated from the field degree when target is "alloc".
    auto input_conv = ConvertedArrayXZ::from_obj(input_obj, "input");
    auto input = input_conv.span.checked_cast_to_qubit_ids("input");
    GF2Field field = resolve_gf2_field(field_obj, input.size());
    pybind11::object resolved = resolve_init_target_obj(self, target_obj, field.degree());

    auto target_conv = ConvertedArrayXZ::from_obj(resolved, "target");
    auto target = target_conv.span.checked_cast_to_qubit_ids("target");

    size_t needed_clean = gf_inverse_workspace_size(field);
    size_t num_clean = self.num_free_qubits() >= needed_clean ? needed_clean : 0;
    array_z clean = array_z::alloc_noinit(num_clean, QXZTypeTag8::QUBIT_ID);
    for (size_t k = 0; k < clean.size(); k++) {
        clean[k] = self.alloc_qubit();
    }
    std::span<const QubitId> clean_span{(QubitId *)clean.items, clean.size()};
    self.builder.broadcast_reset(target);
    gen_gf_inverse(self.builder, CircuitGenCtx{clean_span}, field, target, input);
    for (size_t k = clean.size(); k--;) {
        self.free_qubit((QubitId)clean[k]);
    }
    return resolved;
}

pybind11::object kickmix_py::append_init_gf2_inverse_with_scaffold_obj(
    PyCircuitBuilder &self,
    const pybind11::object &input_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj) {
    auto input_conv = ConvertedArrayXZ::from_obj(input_obj, "input");
    auto input = input_conv.span.checked_cast_to_qubit_ids("input");
    GF2Field field = resolve_gf2_field(field_obj, input.size());
    pybind11::object resolved = resolve_init_target_obj(self, target_obj, field.degree());
    pybind11::object scaffold_obj = alloc_qubit_array_obj(self, gf_inverse_chain_size(field));

    auto target_conv = ConvertedArrayXZ::from_obj(resolved, "target");
    auto target = target_conv.span.checked_cast_to_qubit_ids("target");
    auto scaffold_conv = ConvertedArrayXZ::from_obj(scaffold_obj, "scaffold");
    auto scaffold = scaffold_conv.span.checked_cast_to_qubit_ids("scaffold");

    self.builder.broadcast_reset(target);
    if (scaffold.size() > 0) {
        self.builder.broadcast_reset(scaffold);
    }
    gen_gf_inverse(self.builder, CircuitGenCtx{{}}, field, target, input, scaffold);
    return pybind11::make_tuple(resolved, scaffold_obj);
}

void kickmix_py::append_del_gf2_inverse_obj(
    PyCircuitBuilder &self,
    const pybind11::object &input_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj) {
    auto target_conv = ConvertedArrayXZ::from_obj(target_obj, "target");
    auto input_conv = ConvertedArrayXZ::from_obj(input_obj, "input");
    auto target = target_conv.span.checked_cast_to_qubit_ids("target");
    auto input = input_conv.span.checked_cast_to_qubit_ids("input");
    GF2Field field = resolve_gf2_field(field_obj, target.size());
    size_t m = field.degree();
    throw_unless(input.size() == m, "gen_gf_uninverse: Q_input.size() != field.degree()");

    size_t needed_clean = gf_inverse_workspace_size(field);
    size_t num_clean = self.num_free_qubits() >= needed_clean ? needed_clean : 0;
    array_z clean = array_z::alloc_noinit(num_clean, QXZTypeTag8::QUBIT_ID);
    for (size_t k = 0; k < clean.size(); k++) {
        clean[k] = self.alloc_qubit();
    }
    std::span<const QubitId> clean_span{(QubitId *)clean.items, clean.size()};
    gen_gf_uninverse(self.builder, CircuitGenCtx{clean_span}, field, target, input);
    for (size_t k = clean.size(); k--;) {
        self.free_qubit((QubitId)clean[k]);
    }
}

void kickmix_py::append_del_gf2_inverse_with_scaffold_obj(
    PyCircuitBuilder &self,
    const pybind11::object &input_obj,
    const pybind11::object &target_obj,
    const pybind11::object &scaffold_obj,
    const pybind11::object &field_obj) {
    auto target_conv = ConvertedArrayXZ::from_obj(target_obj, "target");
    auto input_conv = ConvertedArrayXZ::from_obj(input_obj, "input");
    auto target = target_conv.span.checked_cast_to_qubit_ids("target");
    auto input = input_conv.span.checked_cast_to_qubit_ids("input");
    GF2Field field = resolve_gf2_field(field_obj, target.size());
    size_t m = field.degree();
    throw_unless(input.size() == m, "gen_gf_uninverse: Q_input.size() != field.degree()");

    auto scaffold_conv = ConvertedArrayXZ::from_obj(scaffold_obj, "scaffold");
    auto scaffold = scaffold_conv.span.checked_cast_to_qubit_ids("scaffold");
    throw_unless(
        scaffold.size() == gf_inverse_chain_size(field),
        "gen_gf_uninverse: Q_chain.size() != gf_inverse_chain_size(field)");
    gen_gf_uninverse(self.builder, CircuitGenCtx{{}}, field, target, input, scaffold);
}

pybind11::object kickmix_py::append_init_gf2_div_obj(
    PyCircuitBuilder &self,
    const pybind11::object &lhs_obj,
    const pybind11::object &rhs_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj) {
    // Resolve the field from lhs rather than target, so that the target can be
    // allocated from the field degree when target is "alloc".
    auto lhs_conv = ConvertedArrayXZ::from_obj(lhs_obj, "lhs");
    auto lhs = lhs_conv.span.checked_cast_to_qubit_ids("lhs");
    GF2Field field = resolve_gf2_field(field_obj, lhs.size());
    pybind11::object resolved = resolve_init_target_obj(self, target_obj, field.degree());

    auto target_conv = ConvertedArrayXZ::from_obj(resolved, "target");
    auto target = target_conv.span.checked_cast_to_qubit_ids("target");
    self.builder.broadcast_reset(target);
    append_ixor_gf2_div_obj(self, lhs_obj, rhs_obj, resolved, field_obj);
    return resolved;
}

pybind11::object kickmix_py::append_init_gf2_div_with_scaffold_obj(
    PyCircuitBuilder &self,
    const pybind11::object &lhs_obj,
    const pybind11::object &rhs_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj) {
    auto lhs_conv = ConvertedArrayXZ::from_obj(lhs_obj, "lhs");
    auto rhs_conv = ConvertedArrayXZ::from_obj(rhs_obj, "rhs");
    auto lhs = lhs_conv.span.checked_cast_to_qubit_ids("lhs");
    auto rhs = rhs_conv.span.checked_cast_to_qubit_ids("rhs");
    GF2Field field = resolve_gf2_field(field_obj, lhs.size());
    size_t m = field.degree();
    throw_unless(lhs.size() == m, "gen_gf_div: Q_lhs.size() != field.degree()");
    throw_unless(rhs.size() == m, "gen_gf_div: Q_rhs.size() != field.degree()");
    pybind11::object resolved = resolve_init_target_obj(self, target_obj, m);

    auto target_conv = ConvertedArrayXZ::from_obj(resolved, "target");
    auto target = target_conv.span.checked_cast_to_qubit_ids("target");
    self.builder.broadcast_reset(target);

    pybind11::object scaffold_obj = alloc_qubit_array_obj(self, m + gf_inverse_chain_size(field));
    auto scaffold_conv = ConvertedArrayXZ::from_obj(scaffold_obj, "scaffold");
    auto scaffold = scaffold_conv.span.checked_cast_to_qubit_ids("scaffold");
    gen_gf_div(self.builder, CircuitGenCtx{{}}, field, target, lhs, rhs, scaffold);
    return pybind11::make_tuple(resolved, scaffold_obj);
}

void kickmix_py::append_ixor_gf2_div_obj(
    PyCircuitBuilder &self,
    const pybind11::object &lhs_obj,
    const pybind11::object &rhs_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj) {
    auto target_conv = ConvertedArrayXZ::from_obj(target_obj, "target");
    auto lhs_conv = ConvertedArrayXZ::from_obj(lhs_obj, "lhs");
    auto rhs_conv = ConvertedArrayXZ::from_obj(rhs_obj, "rhs");
    auto target = target_conv.span.checked_cast_to_qubit_ids("target");
    auto lhs = lhs_conv.span.checked_cast_to_qubit_ids("lhs");
    auto rhs = rhs_conv.span.checked_cast_to_qubit_ids("rhs");
    GF2Field field = resolve_gf2_field(field_obj, target.size());
    size_t m = field.degree();
    throw_unless(lhs.size() == m, "gen_gf_div: Q_lhs.size() != field.degree()");
    throw_unless(rhs.size() == m, "gen_gf_div: Q_rhs.size() != field.degree()");

    size_t needed_clean = gf_div_workspace_size(field);
    size_t num_clean = self.num_free_qubits() >= needed_clean ? needed_clean : 0;
    array_z clean = array_z::alloc_noinit(num_clean, QXZTypeTag8::QUBIT_ID);
    for (size_t k = 0; k < clean.size(); k++) {
        clean[k] = self.alloc_qubit();
    }
    std::span<const QubitId> clean_span{(QubitId *)clean.items, clean.size()};
    gen_gf_div(self.builder, CircuitGenCtx{clean_span}, field, target, lhs, rhs);
    for (size_t k = clean.size(); k--;) {
        self.free_qubit((QubitId)clean[k]);
    }
}

void kickmix_py::append_del_gf2_div_obj(
    PyCircuitBuilder &self,
    const pybind11::object &lhs_obj,
    const pybind11::object &rhs_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj) {
    auto target_conv = ConvertedArrayXZ::from_obj(target_obj, "target");
    auto lhs_conv = ConvertedArrayXZ::from_obj(lhs_obj, "lhs");
    auto rhs_conv = ConvertedArrayXZ::from_obj(rhs_obj, "rhs");
    auto target = target_conv.span.checked_cast_to_qubit_ids("target");
    auto lhs = lhs_conv.span.checked_cast_to_qubit_ids("lhs");
    auto rhs = rhs_conv.span.checked_cast_to_qubit_ids("rhs");
    GF2Field field = resolve_gf2_field(field_obj, target.size());
    size_t m = field.degree();
    throw_unless(lhs.size() == m, "gen_gf_undiv: Q_lhs.size() != field.degree()");
    throw_unless(rhs.size() == m, "gen_gf_undiv: Q_rhs.size() != field.degree()");

    size_t needed_clean = gf_div_workspace_size(field);
    size_t num_clean = self.num_free_qubits() >= needed_clean ? needed_clean : 0;
    array_z clean = array_z::alloc_noinit(num_clean, QXZTypeTag8::QUBIT_ID);
    for (size_t k = 0; k < clean.size(); k++) {
        clean[k] = self.alloc_qubit();
    }
    std::span<const QubitId> clean_span{(QubitId *)clean.items, clean.size()};
    gen_gf_undiv(self.builder, CircuitGenCtx{clean_span}, field, target, lhs, rhs);
    for (size_t k = clean.size(); k--;) {
        self.free_qubit((QubitId)clean[k]);
    }
}

void kickmix_py::append_del_gf2_div_with_scaffold_obj(
    PyCircuitBuilder &self,
    const pybind11::object &lhs_obj,
    const pybind11::object &rhs_obj,
    const pybind11::object &target_obj,
    const pybind11::object &scaffold_obj,
    const pybind11::object &field_obj) {
    auto target_conv = ConvertedArrayXZ::from_obj(target_obj, "target");
    auto lhs_conv = ConvertedArrayXZ::from_obj(lhs_obj, "lhs");
    auto rhs_conv = ConvertedArrayXZ::from_obj(rhs_obj, "rhs");
    auto target = target_conv.span.checked_cast_to_qubit_ids("target");
    auto lhs = lhs_conv.span.checked_cast_to_qubit_ids("lhs");
    auto rhs = rhs_conv.span.checked_cast_to_qubit_ids("rhs");
    GF2Field field = resolve_gf2_field(field_obj, target.size());
    size_t m = field.degree();
    throw_unless(lhs.size() == m, "gen_gf_undiv: Q_lhs.size() != field.degree()");
    throw_unless(rhs.size() == m, "gen_gf_undiv: Q_rhs.size() != field.degree()");

    auto scaffold_conv = ConvertedArrayXZ::from_obj(scaffold_obj, "scaffold");
    auto scaffold = scaffold_conv.span.checked_cast_to_qubit_ids("scaffold");
    gen_gf_undiv(self.builder, CircuitGenCtx{{}}, field, target, lhs, rhs, scaffold);
}

void kickmix_py::append_gf2_phase_by_product_obj(
    PyCircuitBuilder &self,
    const pybind11::object &lhs_obj,
    const pybind11::object &rhs_obj,
    const pybind11::object &mask_obj,
    const pybind11::object &field_obj) {
    auto mask_conv = ConvertedArrayXZ::from_obj(mask_obj, "mask");
    auto lhs_conv = ConvertedArrayXZ::from_obj(lhs_obj, "lhs");
    auto rhs_conv = ConvertedArrayXZ::from_obj(rhs_obj, "rhs");
    auto mask = mask_conv.span.checked_cast_to_bit_ids("mask");
    auto lhs = lhs_conv.span.checked_cast_to_qubit_ids("lhs");
    auto rhs = rhs_conv.span.checked_cast_to_qubit_ids("rhs");
    GF2Field field = resolve_gf2_field(field_obj, lhs.size());
    size_t m = field.degree();
    throw_unless(mask.size() == m, "gen_gf_phase_by_product: B_mask.size() != field.degree()");
    throw_unless(rhs.size() == m, "gen_gf_phase_by_product: Q_rhs.size() != field.degree()");

    gen_gf_phase_by_product(self.builder, CircuitGenCtx{{}}, field, mask, lhs, rhs);
}
