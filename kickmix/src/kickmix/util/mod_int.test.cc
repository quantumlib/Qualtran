#include "mod_int.h"

#include "gtest/gtest.h"

#include "test.test.h"

using namespace kickmix;

TEST(mod_int, fixed_val_move_constructor) {
    std::shared_ptr<FixedWidthInt> mod = std::make_shared<FixedWidthInt>("503770467140461526195448602895");

    {
        ModInt v("267758050305935043618143975821", mod);
        ASSERT_EQ(v.value, FixedWidthInt("267758050305935043618143975821"));
        ASSERT_EQ(v.modulus, mod);
        v.verify_invariants();
    }

    {
        ModInt v("134888598083240094192267100302247074140311888793175110601800", mod);
        ASSERT_EQ(v.value, 5);
        ASSERT_EQ(v.modulus, mod);
        v.verify_invariants();
    }

    {
        ModInt v("503770467140461526195448602896", mod);
        ASSERT_EQ(v.value, 1);
        ASSERT_EQ(v.modulus, mod);
        v.verify_invariants();
    }
}

TEST(mod_int, move_constructor) {
    std::shared_ptr<FixedWidthInt> mod = std::make_shared<FixedWidthInt>("503770467140461526195448602895");
    auto r = FixedWidthInt("267758050305935043618143975821");
    ModInt v(r, mod);
    ModInt v2(std::move(v));
    v2.verify_invariants();
    ASSERT_EQ(v2.value, r);
    ASSERT_EQ(v2.modulus, mod);
}

TEST(mod_int, copy_constructor) {
    std::shared_ptr<FixedWidthInt> mod = std::make_shared<FixedWidthInt>("503770467140461526195448602895");
    auto r = FixedWidthInt("267758050305935043618143975821");
    ModInt v(r, mod);
    ModInt v2(v);
    v.verify_invariants();
    v2.verify_invariants();
    ASSERT_EQ(v.value, r);
    ASSERT_EQ(v.modulus, mod);
    ASSERT_EQ(v2.value, r);
    ASSERT_EQ(v2.modulus, mod);
}

TEST(mod_int, move_assignment) {
    std::shared_ptr<FixedWidthInt> mod = std::make_shared<FixedWidthInt>("503770467140461526195448602895");
    std::shared_ptr<FixedWidthInt> mod2 =
        std::make_shared<FixedWidthInt>("69992199261667865755875935508998456568559495868298186351733");

    {
        ModInt v("267758050305935043618143975821", mod);
        ModInt v2("267758050305935043618143975820", mod);
        v2 = std::move(v);
        ASSERT_EQ(v2.value, FixedWidthInt("267758050305935043618143975821"));
        ASSERT_EQ(v2.modulus, mod);
        v2.verify_invariants();
    }

    {
        ModInt v("267758050305935043618143975821", mod);
        ModInt v2("267758050305935043618143975820", mod2);
        v2 = std::move(v);
        ASSERT_EQ(v2.value, FixedWidthInt("267758050305935043618143975821"));
        ASSERT_EQ(v2.modulus, mod);
        v2.verify_invariants();
    }
}

TEST(mod_int, fixed_val_copy_constructor) {
    std::shared_ptr<FixedWidthInt> mod = std::make_shared<FixedWidthInt>("503770467140461526195448602895");

    {
        auto r = FixedWidthInt("267758050305935043618143975821");
        ModInt v(r, mod);
        ASSERT_EQ(v.value, r);
        ASSERT_EQ(v.modulus, mod);
        v.verify_invariants();
    }

    {
        auto r = FixedWidthInt("134888598083240094192267100302247074140311888793175110601800");
        ModInt v(r, mod);
        ASSERT_NE(v.value, r);
        ASSERT_EQ(v.value, 5);
        ASSERT_EQ(v.modulus, mod);
        v.verify_invariants();
    }

    {
        auto r = FixedWidthInt("503770467140461526195448602896");
        ModInt v(r, mod);
        ASSERT_NE(v.value, r);
        ASSERT_EQ(v.value, 1);
        ASSERT_EQ(v.modulus, mod);
        v.verify_invariants();
    }
}

TEST(mod_int, copy_assignment) {
    std::shared_ptr<FixedWidthInt> mod = std::make_shared<FixedWidthInt>("503770467140461526195448602895");
    std::shared_ptr<FixedWidthInt> mod2 =
        std::make_shared<FixedWidthInt>("69992199261667865755875935508998456568559495868298186351733");

    auto r = FixedWidthInt("267758050305935043618143975821");
    auto r2 = FixedWidthInt("267758050305935043618143975823");
    ModInt v(r, mod);
    ModInt v2(r2, mod2);
    v = v2;
    v.verify_invariants();
    v2.verify_invariants();
    ASSERT_EQ(v.value, r2);
    ASSERT_EQ(v.modulus, mod2);
    ASSERT_EQ(v2.value, r2);
    ASSERT_EQ(v2.modulus, mod2);
}

TEST(mod_int, zero_init) {
    std::shared_ptr<FixedWidthInt> mod = std::make_shared<FixedWidthInt>("503770467140461526195448602895");

    ModInt v(mod);
    v.verify_invariants();
    ASSERT_EQ(v.value, 0);
    ASSERT_EQ(v.modulus, mod);
}

TEST(mod_int, random) {
    std::shared_ptr<FixedWidthInt> mod = std::make_shared<FixedWidthInt>("503770467140461526195448602895");
    auto rng = INDEPENDENT_TEST_RNG();

    ModInt v = ModInt::random(rng, mod);
    v.verify_invariants();
    ASSERT_EQ(v.modulus, mod);
    auto v2 = v;

    v.randomize(rng);
    v.verify_invariants();
    ASSERT_NE(v, v2);  // With *exremely* high probability.
}

TEST(mod_int, bool_val) {
    std::shared_ptr<FixedWidthInt> mod = std::make_shared<FixedWidthInt>("503770467140461526195448602895");
    auto m = ModInt(mod);
    auto m2 = ModInt("1", mod);
    ASSERT_TRUE(m2);
    ASSERT_FALSE(m);
}

TEST(mod_int, eq) {
    std::shared_ptr<FixedWidthInt> mod = std::make_shared<FixedWidthInt>("503770467140461526195448602895");
    auto m = ModInt(mod);
    ASSERT_EQ(m, ModInt(mod));
    ASSERT_EQ(m, ModInt(std::make_shared<FixedWidthInt>("1")));
    ASSERT_NE(m, ModInt("1", mod));
}

TEST(mod_int, times_by) {
    std::shared_ptr<FixedWidthInt> mod = std::make_shared<FixedWidthInt>("503770467140461526195448602895");
    auto v1 = ModInt("267758050305935043618143975821", mod);
    auto v2 = ModInt("403712566440672721379549381870", mod);
    v1 *= v2;
    ASSERT_EQ(v1.modulus, mod);
    ASSERT_EQ(v1.value, FixedWidthInt("53249894652784811511808232255"));
}

TEST(mod_int, divide_by_not_invertible) {
    std::shared_ptr<FixedWidthInt> mod = std::make_shared<FixedWidthInt>("503770467140461526195448602895");
    auto v1 = ModInt("267758050305935043618143975821", mod);
    auto v2 = ModInt("403712566440672721379549381870", mod);
    ASSERT_THROW({ v1 /= v2; }, std::invalid_argument);
}

TEST(mod_int, divide_by) {
    std::shared_ptr<FixedWidthInt> mod = std::make_shared<FixedWidthInt>("503770467140461526195448602895");
    auto v1 = ModInt("267758050305935043618143975821", mod);
    auto v2 = ModInt("393836260909983489752333563576", mod);
    v1 /= v2;
    ASSERT_EQ(v1.modulus, mod);
    ASSERT_EQ(v1.value, FixedWidthInt("145548978695648329449912064561"));
    v1.verify_invariants();
    v2.verify_invariants();
}

TEST(mod_int, divide_by_move) {
    std::shared_ptr<FixedWidthInt> mod = std::make_shared<FixedWidthInt>("503770467140461526195448602895");
    auto v1 = ModInt("267758050305935043618143975821", mod);
    auto v2 = ModInt("393836260909983489752333563576", mod);
    v1 /= std::move(v2);
    ASSERT_EQ(v1.modulus, mod);
    ASSERT_EQ(v1.value, FixedWidthInt("145548978695648329449912064561"));
    v1.verify_invariants();
}

TEST(mod_int, plus_by) {
    std::shared_ptr<FixedWidthInt> mod = std::make_shared<FixedWidthInt>("503770467140461526195448602895");
    auto v1 = ModInt("267758050305935043618143975821", mod);
    auto v2 = ModInt("393836260909983489752333563576", mod);
    v1 += v2;
    ASSERT_EQ(v1.modulus, mod);
    ASSERT_EQ(v1.value, FixedWidthInt("157823844075457007175028936502"));
    v1.verify_invariants();
    v2.verify_invariants();
    v1 += ModInt("1", mod);
    ASSERT_EQ(v1.modulus, mod);
    ASSERT_EQ(v1.value, FixedWidthInt("157823844075457007175028936503"));
    v1.verify_invariants();
    v2.verify_invariants();
}

TEST(mod_int, sub_by) {
    std::shared_ptr<FixedWidthInt> mod = std::make_shared<FixedWidthInt>("503770467140461526195448602895");
    auto v1 = ModInt("267758050305935043618143975821", mod);
    auto v2 = ModInt("393836260909983489752333563576", mod);
    v1 -= v2;
    ASSERT_EQ(v1.modulus, mod);
    ASSERT_EQ(v1.value, FixedWidthInt("377692256536413080061259015140"));
    v1.verify_invariants();
    v2.verify_invariants();
    v1 -= ModInt("1", mod);
    ASSERT_EQ(v1.modulus, mod);
    ASSERT_EQ(v1.value, FixedWidthInt("377692256536413080061259015139"));
    v1.verify_invariants();
    v2.verify_invariants();
}

TEST(mod_int, negate) {
    std::shared_ptr<FixedWidthInt> mod = std::make_shared<FixedWidthInt>("503770467140461526195448602895");
    auto v1 = ModInt("267758050305935043618143975821", mod);
    v1.negate();
    v1.verify_invariants();
    ASSERT_EQ(v1.modulus, mod);
    ASSERT_EQ(v1.value, FixedWidthInt("236012416834526482577304627074"));

    v1 = ModInt(mod);
    v1.negate();
    v1.verify_invariants();
    ASSERT_EQ(v1.modulus, mod);
    ASSERT_EQ(v1.value, FixedWidthInt("0"));

    v1 = ModInt("1", mod);
    v1.negate();
    v1.verify_invariants();
    ASSERT_EQ(v1.modulus, mod);
    ASSERT_EQ(v1.value, FixedWidthInt("503770467140461526195448602894"));
    v1.negate();
    v1.verify_invariants();
    ASSERT_EQ(v1.modulus, mod);
    ASSERT_EQ(v1.value, FixedWidthInt("1"));
}

TEST(mod_int, invert) {
    std::shared_ptr<FixedWidthInt> mod = std::make_shared<FixedWidthInt>("503770467140461526195448602895");
    auto v1 = ModInt("267758050305935043618143975821", mod);
    v1.invert();
    v1.verify_invariants();
    ASSERT_EQ(v1.modulus, mod);
    ASSERT_EQ(v1.value, FixedWidthInt("51509054617542776874239733256"));

    v1.invert();
    v1.verify_invariants();
    ASSERT_EQ(v1.modulus, mod);
    ASSERT_EQ(v1.value, FixedWidthInt("267758050305935043618143975821"));

    v1 = ModInt(mod);
    ASSERT_THROW({ v1.invert(); }, std::invalid_argument);

    v1 = ModInt("1", mod);
    v1.invert();
    v1.verify_invariants();
    ASSERT_EQ(v1.modulus, mod);
    ASSERT_EQ(v1.value, FixedWidthInt("1"));

    v1 = ModInt("2", mod);
    v1.invert();
    v1.verify_invariants();
    ASSERT_EQ(v1.modulus, mod);
    ASSERT_EQ(v1.value, FixedWidthInt("251885233570230763097724301448"));
    v1.invert();
    v1.verify_invariants();
    ASSERT_EQ(v1.modulus, mod);
    ASSERT_EQ(v1.value, FixedWidthInt("2"));
}

TEST(mod_int, left_shift) {
    std::shared_ptr<FixedWidthInt> mod = std::make_shared<FixedWidthInt>("503770467140461526195448602895");
    auto v1 = ModInt("267758050305935043618143975821", mod);
    v1 <<= int64_t{1};
    v1.verify_invariants();
    ASSERT_EQ(v1.value, FixedWidthInt("31745633471408561040839348747"));
    ASSERT_EQ(v1.modulus, mod);
    v1 <<= uint64_t{1};
    v1.verify_invariants();
    ASSERT_EQ(v1.value, FixedWidthInt("63491266942817122081678697494"));
    ASSERT_EQ(v1.modulus, mod);
    v1 <<= uint32_t{1};
    v1.verify_invariants();
    ASSERT_EQ(v1.value, FixedWidthInt("126982533885634244163357394988"));
    ASSERT_EQ(v1.modulus, mod);
    v1 <<= int32_t{1};
    v1.verify_invariants();
    ASSERT_EQ(v1.value, FixedWidthInt("253965067771268488326714789976"));
    ASSERT_EQ(v1.modulus, mod);
    v1 <<= int32_t{-1};
    v1.verify_invariants();
    ASSERT_EQ(v1.value, FixedWidthInt("126982533885634244163357394988"));
    ASSERT_EQ(v1.modulus, mod);
    v1 <<= int64_t{-2};
    v1.verify_invariants();
    ASSERT_EQ(v1.value, FixedWidthInt("31745633471408561040839348747"));
    ASSERT_EQ(v1.modulus, mod);
    v1 <<= int64_t{2};
    v1.verify_invariants();
    ASSERT_EQ(v1.value, FixedWidthInt("126982533885634244163357394988"));
    ASSERT_EQ(v1.modulus, mod);
}

TEST(mod_int, right_shift) {
    std::shared_ptr<FixedWidthInt> mod = std::make_shared<FixedWidthInt>("503770467140461526195448602895");
    auto v1 = ModInt("267758050305935043618143975821", mod);
    v1 >>= int64_t{1};
    v1.verify_invariants();
    ASSERT_EQ(v1.value, FixedWidthInt("385764258723198284906796289358"));
    ASSERT_EQ(v1.modulus, mod);
    v1 >>= uint64_t{1};
    v1.verify_invariants();
    ASSERT_EQ(v1.value, FixedWidthInt("192882129361599142453398144679"));
    ASSERT_EQ(v1.modulus, mod);
    v1 >>= uint32_t{1};
    v1.verify_invariants();
    ASSERT_EQ(v1.value, FixedWidthInt("348326298251030334324423373787"));
    ASSERT_EQ(v1.modulus, mod);
    v1 >>= int32_t{1};
    v1.verify_invariants();
    ASSERT_EQ(v1.value, FixedWidthInt("426048382695745930259935988341"));
    ASSERT_EQ(v1.modulus, mod);
    v1 >>= int32_t{-1};
    v1.verify_invariants();
    ASSERT_EQ(v1.value, FixedWidthInt("348326298251030334324423373787"));
    ASSERT_EQ(v1.modulus, mod);
    v1 >>= int64_t{-2};
    v1.verify_invariants();
    ASSERT_EQ(v1.value, FixedWidthInt("385764258723198284906796289358"));
    ASSERT_EQ(v1.modulus, mod);
    v1 >>= int64_t{2};
    v1.verify_invariants();
    ASSERT_EQ(v1.value, FixedWidthInt("348326298251030334324423373787"));
    ASSERT_EQ(v1.modulus, mod);
}
