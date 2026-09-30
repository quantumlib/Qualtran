#include "gen_lookup.h"

using namespace kickmix;

void kickmix::gen_binary_to_unary(CircuitBuilder &builder, CircuitGenCtx ctx, std::span<const QubitId> target) {
    auto mark = builder.raii_mark_block_entry("binary_to_unary");

    size_t num_bits = 0;
    while ((size_t{1} << num_bits) < target.size()) {
        num_bits++;
    }
    if (target.size() != (size_t{1} << num_bits)) {
        throw std::invalid_argument("gen_binary_to_unary not implemented: non power of 2 target size");
    }
    builder.broadcast_reset(target.subspan(num_bits));

    for (size_t k = num_bits; k--;) {
        builder.swap(target[k], target[(2 << k) - 1]);
    }
    builder.x(target[0]);
    for (size_t k = 0; k < num_bits; k++) {
        size_t m = 1 << k;
        size_t c = 2 * m - 1;
        for (size_t j = 0; j < m - 1; j++) {
            builder.cswap(target[c], target[j], target[j + m]);
        }
        for (size_t j = 0; j < m - 1; j++) {
            builder.cx(target[j + m], target[c]);
        }
        builder.cx(target[c], target[m - 1]);
    }
}

void kickmix::gen_unary_to_binary(CircuitBuilder &builder, CircuitGenCtx ctx, std::span<const QubitId> target) {
    auto mark = builder.raii_mark_block_entry("unary_to_binary");

    size_t num_bits = 0;
    while ((size_t{1} << num_bits) < target.size()) {
        num_bits++;
    }
    if (target.size() != (size_t{1} << num_bits)) {
        throw std::invalid_argument("gen_unary_to_binary not implemented: non power of 2 target size");
    }

    std::vector<QubitId> swapped_targets;
    for (auto e : target) {
        swapped_targets.push_back(e);
    }
    for (size_t k = 0; k < num_bits; k++) {
        size_t k2 = (2 << k) - 1;
        builder.swap(target[k], target[k2]);
    }
    for (size_t k = num_bits; k--;) {
        size_t k2 = (2 << k) - 1;
        std::swap(swapped_targets[k], swapped_targets[k2]);
    }

    for (size_t k = num_bits; k--;) {
        size_t m = 1 << k;
        size_t c = 2 * m - 1;
        builder.cx(swapped_targets[c], swapped_targets[m - 1]);
        for (size_t j = m - 1; j--;) {
            builder.cx(swapped_targets[j + m], swapped_targets[c]);
        }
        for (size_t j = m - 1; j--;) {
            builder.cx(swapped_targets[j + m], swapped_targets[j]);
            builder.ccx(swapped_targets[c], swapped_targets[j], builder.hmr_raii_xbit(swapped_targets[j + m]));
        }
    }
    builder.x(builder.hmr_raii_xbit(swapped_targets[0]));
}

void kickmix::gen_unlookup(
    CircuitBuilder &builder,
    CircuitGenCtx ctx,
    const stride_span_z &table_bits,
    stride_span<const QubitId> address,
    stride_span<const QubitId> output) {
    if (output.empty()) {
        return;
    }

    auto mark = builder.raii_mark_block_entry("unlookup");

    if (table_bits.size() % output.size()) {
        std::stringstream ss;
        ss << "gen_unlookup: ";
        ss << "table_bits.size()=" << table_bits.size();
        ss << "% output.size()=" << output.size();
        ss << " != 0";
        throw std::invalid_argument(ss.str());
    }
    size_t max_n = 1 << address.size();
    size_t n = table_bits.size() / output.size();
    size_t w = output.size();
    if (n > max_n) {
        std::stringstream ss;
        ss << "gen_unlookup: ";
        ss << "table_bits.size()=" << table_bits.size();
        ss << " > (output.size()=" << output.size();
        ss << " << address.size()" << address.size();
        ss << ")";
        throw std::invalid_argument(ss.str());
    }

    std::vector<CircuitBuilderRaiiBit> raii_phase_lookup_table;
    std::vector<BitId> phase_lookup_table;
    for (size_t k = 0; k < (size_t{1} << address.size()); k++) {
        raii_phase_lookup_table.push_back(builder.alloc_clean_raii_bit());
        phase_lookup_table.push_back(raii_phase_lookup_table.back().bit);
    }
    for (auto e : phase_lookup_table) {
        builder.bit_store0(e);
    }

    for (size_t j = 0; j < output.size(); j++) {
        CircuitBuilderRaiiXBit mx = builder.hmr_raii_xbit(output[j]);
        builder.push_condition(mx.bit);
        for (size_t k = 0; k < n; k++) {
            auto v = table_bits[k * w + j];
            if (v.is_bit()) {
                builder.cx((BitId)v, phase_lookup_table[k]);
            } else if (v.is_bool()) {
                builder.cx((bool)v, phase_lookup_table[k]);
            } else {
                throw std::invalid_argument("quantum data in table");
            }
        }
        builder.pop_condition();
    }

    std::vector<QubitId> unfolded_targets;
    size_t h = address.size() / 2;
    for (size_t k = 0; k < h; k++) {
        unfolded_targets.push_back(address[k]);
    }
    size_t n2 = 1 << unfolded_targets.size();
    for (auto e : output) {
        if (unfolded_targets.size() >= n2) {
            break;
        }
        unfolded_targets.push_back(e);
    }
    while (unfolded_targets.size() < n2) {
        unfolded_targets.push_back(ctx.take_clean(1)[0]);
    }

    gen_binary_to_unary(builder, ctx, unfolded_targets);
    gen_lookup(builder, ctx, phase_lookup_table, address.skip(h), unfolded_targets, 'Z');
    gen_unary_to_binary(builder, ctx, unfolded_targets);
}

void kickmix::gen_lookup(
    CircuitBuilder &builder,
    CircuitGenCtx ctx,
    const stride_span_z &table_bits,
    stride_span<const QubitId> address,
    stride_span<const QubitId> output,
    char pauli_type,
    QubitOrTrue control) {
    if (output.empty()) {
        return;
    }

    auto mark = builder.raii_mark_block_entry("lookup");

    if (table_bits.size() % output.size()) {
        std::stringstream ss;
        ss << "gen_lookup: ";
        ss << "table_bits.size()=" << table_bits.size();
        ss << "% output.size()=" << output.size();
        ss << " != 0";
        throw std::invalid_argument(ss.str());
    }
    size_t max_n = 1 << address.size();
    size_t n = table_bits.size() / output.size();
    if (n > max_n) {
        std::stringstream ss;
        ss << "gen_lookup: ";
        ss << "table_bits.size()=" << table_bits.size();
        ss << " > (output.size()=" << output.size();
        ss << " << address.size()=" << address.size();
        ss << ")=" << (max_n * output.size());
        throw std::invalid_argument(ss.str());
    }

    if (address.empty()) {
        if (pauli_type == 'X') {
            builder.broadcast_ccx(control, table_bits, output);
        } else if (pauli_type == 'Z') {
            builder.broadcast_ccz(control, table_bits, output);
        } else {
            throw std::invalid_argument("gen_lookup: unrecognized pauli type (not 'X' or 'Z')");
        }
        return;
    }
    size_t half_n = max_n / 2;

    auto anc = ctx.take_clean(1)[0];
    builder.x(address.back());
    builder.reset_and(control, address.back(), anc);
    builder.x(address.back());
    auto address_rem = address.keep(address.size() - 1);
    if (n > half_n) {
        gen_lookup(builder, ctx, table_bits.keep(output.size() * half_n), address_rem, output, pauli_type, anc);
        builder.cx(control, anc);
        gen_lookup(builder, ctx, table_bits.skip(output.size() * half_n), address_rem, output, pauli_type, anc);
    } else {
        gen_lookup(builder, ctx, table_bits, address_rem, output, pauli_type, anc);
    }
    builder.del_and(control, address.back(), anc);
}
