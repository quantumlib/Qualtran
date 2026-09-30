#ifndef KICKMIX_REGISTER_ID_H
#define KICKMIX_REGISTER_ID_H

#include <cstdint>
#include <ostream>

namespace kickmix {
struct RegisterId {
    uint32_t id;
    inline bool operator==(RegisterId other) const {
        return id == other.id;
    }
    inline bool operator<(RegisterId other) const {
        return id < other.id;
    }
    std::string str() const {
        std::string result;
        result.push_back('r');
        result.append(std::to_string(id));
        return result;
    }
};
std::ostream &operator<<(std::ostream &out, RegisterId v);
}  // namespace kickmix

#endif
