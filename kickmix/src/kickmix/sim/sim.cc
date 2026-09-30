#include "kickmix/sim/sim.h"

using namespace kickmix;

std::ostream &kickmix::operator<<(std::ostream &out, const SimInitInstruction &rhs) {
    out << rhs.target_register;
    out << (rhs.randomize ? '<' : '=');
    out << rhs.value.decimal();
    return out;
}

SimInitInstruction SimInitInstruction::from_str(std::string_view text) {
    std::string_view rest = text;
    SimInitInstruction result;
    if (rest.starts_with('r')) {
        rest = rest.substr(1);
        uint64_t value = 0;
        while (!rest.empty() && rest[0] >= '0' && rest[0] <= '9' && value <= UINT32_MAX) {
            value *= 10;
            value += rest[0] - '0';
            rest = rest.substr(1);
        }
        if (value > UINT32_MAX) {
            throw std::invalid_argument("Register id is out of range in '" + std::string(text) + "'");
        }
        result.target_register.id = static_cast<uint32_t>(value);
    } else {
        throw std::invalid_argument("Failed to parse init instruction: '" + std::string(text) + "'");
    }
    if (rest == "=*") {
        result.value = FixedWidthInt(0);
        result.randomize = true;
    } else if (rest.starts_with('=')) {
        result.value = FixedWidthInt::from_str(rest.substr(1));
        result.randomize = false;
    } else if (rest.starts_with('<')) {
        result.value = FixedWidthInt::from_str(rest.substr(1));
        result.randomize = true;
    } else {
        throw std::invalid_argument("Failed to parse init instruction: '" + std::string(text) + "'");
    }
    return result;
}

std::vector<SimInitInstruction> SimInitInstruction::from_str_many(std::string_view text) {
    try {
        std::vector<SimInitInstruction> result;
        size_t start = 0;
        for (size_t k = 0; k < text.size(); k++) {
            if (text[k] == ',') {
                result.push_back(from_str(text.substr(start, k - start)));
                start = k + 1;
            }
        }
        if (start != text.size()) {
            result.push_back(from_str(text.substr(start)));
        }
        return result;
    } catch (const std::invalid_argument &ex) {
        std::stringstream ss;
        ss << ex.what();
        ss << "\n    Examples of valid init strings: r0=5, r1<0x500, r2=*";
        throw std::invalid_argument(ss.str());
    }
}
