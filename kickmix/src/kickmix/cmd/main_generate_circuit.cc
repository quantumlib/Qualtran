#include <fstream>
#include <iostream>

#include "kickmix/gen/adders/gen_iadd.h"
#include "kickmix/gen/adders/gen_iadd_classical.h"
#include "kickmix/gen/comparators/gen_cmp.h"
#include "kickmix/gen/lookup/gen_lookup.h"
#include "kickmix/gen/modular_arithmetic/gen_imul2_mod.h"
#include "kickmix/util/arg_parse.h"
#include "main_util.h"

using namespace kickmix;

int kickmix::main_generate_circuit(int argc, const char **argv) {
    check_for_unknown_arguments(
        {
            "--mod",
            "--btol",
            "--flame_out",
            "--n",
            "--w",
            "--kind",
            "--kmb",
            "--diagram",
            "--controlled",
            "--minimize_qubits",
            "--clean",
            "--dirty",
            "--validate",
            "--skip_output",
            "--out",
        },
        {},
        "kickmix",
        "circuit",
        argc,
        argv);

    auto kind = find_enum_argument(
        "--kind",
        nullptr,
        std::map<std::string_view, std::string_view>{
            {"iadd", "iadd"},
            {"flip_if_lt", "flip_if_lt"},
            {"iadd_classical", "iadd_classical"},
            {"iadd_classical_simple", "iadd_classical_simple"},
            {"imul2_mod", "imul2_mod"},
            {"lookup", "lookup"},
            {"unlookup", "unlookup"},
        },
        argc,
        argv);
    auto diagram = find_bool_argument("--diagram", argc, argv);
    auto kmb = find_bool_argument("--kmb", argc, argv);
    auto num_clean = find_int64_argument("--clean", -1, -1, 1000000, argc, argv);
    auto num_dirty = find_int64_argument("--dirty", -1, -1, 1000000, argc, argv);
    auto w = find_int64_argument("--w", -1, -1, 1000000, argc, argv);
    auto n = find_int64_argument("--n", -1, -1, 1000000, argc, argv);
    auto mod_arg = find_argument("--mod", argc, argv);
    auto controlled = find_bool_argument("--controlled", argc, argv);
    auto btol = find_double_argument("--btol", INFINITY, 0, INFINITY, argc, argv);
    auto minimize_qubits = find_bool_argument("--minimize_qubits", argc, argv);

    CircuitBuilder builder;
    builder.skip_validation = !find_bool_argument("--validate", argc, argv);

    if (kind == "flip_if_lt") {
        if (n == -1) {
            std::cerr << "Need '--n #'.\n";
            return 1;
        }
        auto lhs = builder.append_register(n);
        auto rhs = builder.append_register(n);
        auto clean = builder.reserve_qubits(num_clean == -1 ? n : num_clean);
        QubitOrTrue control = true;
        if (controlled) {
            control = builder.append_register(1)[0];
        }
        auto out = builder.append_register(1)[0];
        auto dirty = builder.reserve_qubits(num_dirty < 0 ? 0 : num_dirty);
        CircuitGenCtx ctx{.clean_workspace = clean};
        ctx = ctx.with_more_dirty_qubits(dirty);
        ctx.minimize_qubits = minimize_qubits;
        gen_flip_if_lt(builder, ctx, lhs, rhs, out, false, control, btol);
    } else if (kind == "iadd") {
        if (n == -1) {
            std::cerr << "Need '--n #'.\n";
            return 1;
        }
        auto target = builder.append_register(n);
        auto offset = builder.append_register(n);
        auto clean = builder.reserve_qubits(num_clean == -1 ? 0 : num_clean);
        QubitOrTrue control = true;
        if (controlled) {
            control = builder.append_register(1)[0];
        }
        auto dirty = builder.reserve_qubits(num_dirty < 0 ? 0 : num_dirty);
        CircuitGenCtx ctx{.clean_workspace = clean};
        ctx = ctx.with_more_dirty_qubits(dirty);
        ctx.minimize_qubits = minimize_qubits;
        gen_iadd(builder, ctx, target, offset, control);
    } else if (kind == "iadd_classical") {
        if (n == -1) {
            std::cerr << "Need '--n #'.\n";
            return 1;
        }
        if (num_clean == -1) {
            num_clean = n;
        }
        if (num_dirty == -1) {
            num_dirty = n;
        }
        auto target = builder.append_register(n, "target");
        auto offset = builder.append_classical_register_qcarray_result(n, "offset");
        QubitOrBitOrBool carry = false;  // builder.append_register(1, "carry_in")[0];
        auto clean = builder.reserve_qubits(num_clean == -1 ? 0 : num_clean);
        auto dirty = builder.reserve_qubits(num_dirty);
        CircuitGenCtx ctx{.clean_workspace = clean, .minimize_qubits = minimize_qubits};
        ctx = ctx.with_more_dirty_qubits(dirty);
        QubitOrTrue control = true;
        if (controlled) {
            control = builder.append_register(1)[0];
        }
        gen_iadd_classical(builder, ctx, target, offset, carry, control, btol);
    } else if (kind == "iadd_classical_simple") {
        if (n == -1) {
            std::cerr << "Need '--n #'.\n";
            return 1;
        }
        if (num_clean == -1) {
            num_clean = n;
        }
        auto target = builder.append_register(n);
        auto offset = builder.append_classical_register_qcarray_result(n);
        auto carry = builder.append_register(1)[0];
        auto clean = builder.reserve_qubits(num_clean == -1 ? 0 : num_clean);
        auto dirty = builder.reserve_qubits(num_dirty < 0 ? 0 : num_dirty);
        CircuitGenCtx ctx{.clean_workspace = clean};
        ctx = ctx.with_more_dirty_qubits(dirty);
        ctx.minimize_qubits = minimize_qubits;
        QubitOrTrue control = true;
        if (controlled) {
            control = builder.append_register(1)[0];
        }
        gen_iadd_classical_simple(builder, ctx, target, offset, carry, control);
    } else if (kind == "lookup") {
        if (n == -1) {
            n = 3;
        }
        if (w == -1) {
            w = 2;
        }
        if (num_clean == -1) {
            num_clean = n;
        }
        std::vector<QubitOrBitOrBool> mod_bits;
        std::vector<QubitId> target;
        auto address = builder.append_register(n);
        auto output = builder.append_register(w == -1 ? 2 : static_cast<size_t>(w));
        auto table_bits = builder.append_classical_register(w << n);
        auto clean = builder.reserve_qubits(static_cast<size_t>(num_clean));
        QubitOrTrue control = true;
        if (controlled) {
            control = builder.append_register(1)[0];
        }
        auto dirty = builder.reserve_qubits(num_dirty < 0 ? 0 : num_dirty);
        CircuitGenCtx ctx{.clean_workspace = clean};
        ctx = ctx.with_more_dirty_qubits(dirty);
        ctx.minimize_qubits = minimize_qubits;
        gen_lookup(builder, ctx, table_bits, address, output, 'X', control);
    } else if (kind == "unlookup") {
        if (n == -1) {
            n = 3;
        }
        if (w == -1) {
            w = 2;
        }
        if (num_clean == -1) {
            num_clean = std::max(static_cast<size_t>(n), size_t{1} << (n / 2));
        }
        if (controlled) {
            std::cerr << "--controlled not supported with unlookup\n";
            return 1;
        }
        std::vector<QubitOrBitOrBool> mod_bits;
        std::vector<QubitId> target;
        auto address = builder.append_register(n);
        auto output = builder.append_register(w == -1 ? 2 : static_cast<size_t>(w));
        auto table_bits = builder.append_classical_register(w << n);
        auto clean = builder.reserve_qubits(num_clean == -1 ? n : static_cast<size_t>(num_clean));
        auto dirty = builder.reserve_qubits(num_dirty < 0 ? 0 : num_dirty);
        CircuitGenCtx ctx{.clean_workspace = clean};
        ctx = ctx.with_more_dirty_qubits(dirty);
        ctx.minimize_qubits = minimize_qubits;
        gen_unlookup(builder, ctx, table_bits, address, output);
    } else if (kind == "imul2_mod") {
        array_z mod_bits;
        std::vector<QubitId> target;
        if (mod_arg == nullptr && n > 1) {
            target = builder.append_register(n);
            mod_bits = builder.append_classical_register_qcarray_result(n);
            mod_bits.front() = true;
            mod_bits.back() = true;
            mod_bits.common_type = QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE;
        } else if (mod_arg != nullptr && n == -1) {
            mod_bits = array_z::copy_of(FixedWidthInt(mod_arg));
            n = mod_bits.size();
            target = builder.append_register(n);
        } else {
            std::cerr << "Specify '--mod #' xor '--n #'.\n";
            return 1;
        }
        QubitOrTrue control = true;
        if (controlled) {
            control = builder.append_register(1)[0];
        }
        auto clean = builder.reserve_qubits(num_clean == -1 ? n : static_cast<size_t>(num_clean));
        auto dirty = builder.reserve_qubits(num_dirty < 0 ? 0 : num_dirty);
        CircuitGenCtx ctx{.clean_workspace = clean};
        ctx = ctx.with_more_dirty_qubits(dirty);
        ctx.minimize_qubits = minimize_qubits;
        gen_imul2_mod(builder, ctx, target, mod_bits, control, btol);
    } else {
        std::cerr << "Unrecognized kind: " << kind << '\n';
        return 1;
    }

    const char *out = find_argument("--out", argc, argv);
    FILE *out_file = stdout;
    if (out != nullptr) {
        out_file = fopen(out, "wb");
        if (out_file == nullptr) {
            throw std::invalid_argument("Failed to open " + std::string(out));
        }
    }

    const char *flame_out = find_argument("--flame_out", argc, argv);
    if (flame_out != nullptr) {
        std::ofstream flame_out_file(flame_out);
        builder.write_analysis_svg_to(flame_out_file, n >= 0 ? (size_t)n : 0);
    }

    if (find_bool_argument("--skip_output", argc, argv)) {
        return 0;
    }

    if (diagram) {
        builder.finish_circuit().write_text_diagram_to(out_file, true);
        putc('\n', out_file);
    } else if (kmb) {
        builder.finish_circuit().write_kmb_to(out_file);
    } else {
        builder.finish_circuit().write_kmx_to(out_file);
    }

    if (out_file != stdout) {
        fclose(out_file);
    }
    return 0;
}
