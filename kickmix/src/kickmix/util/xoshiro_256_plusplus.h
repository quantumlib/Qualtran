#ifndef KICKMIX_UTIL_XOSHIRO_H
#define KICKMIX_UTIL_XOSHIRO_H

namespace kickmix {

template <typename TWord>
inline TWord rol64(const TWord &x, int k) {
    return x.u64_left_shift(k) | x.u64_right_shift(64 - k);
}

template <typename TWord>
struct Xoshiro256PlusPlus {
    TWord s[4];
    TWord next() {
        auto result = rol64<TWord>(s[0].u64_add(s[3]), 23).u64_add(s[0]);
        auto t = s[1].u64_left_shift(17);

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;
        s[3] = rol64<TWord>(s[3], 45);

        return result;
    }
};

}  // namespace kickmix

#endif
