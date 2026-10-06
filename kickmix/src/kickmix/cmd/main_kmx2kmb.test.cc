#include "gtest/gtest.h"

#include "main_util.h"
#include "test.test.h"

using namespace kickmix;

TEST(kmx2kmb, round_trip_incrementer) {
    RaiiTempNamedFile in(R"CIRCUIT(APPEND_TO_REGISTER q0 r0
APPEND_TO_REGISTER q1 r0
APPEND_TO_REGISTER q2 r0
CCX q0 q1 q2
CX q0 q1
X q0
)CIRCUIT");

    RaiiTempNamedFile out;
    main_kmx2kmb(
        6,
        std::vector<const char *>{
            "kickmix",
            "kmx2kmb",
            "--in",
            in.path.c_str(),
            "--out",
            out.path.c_str(),
        }
            .data());

    RaiiTempNamedFile round_trip;
    main_kmb2kmx(
        6,
        std::vector<const char *>{
            "kickmix",
            "kmb2kmx",
            "--in",
            out.path.c_str(),
            "--out",
            round_trip.path.c_str(),
        }
            .data());

    ASSERT_EQ(round_trip.read_contents(), in.read_contents());
}

TEST(kmx2kmb, round_trip_empty) {
    RaiiTempNamedFile in("");

    RaiiTempNamedFile out;
    main_kmx2kmb(
        6,
        std::vector<const char *>{
            "kickmix",
            "kmx2kmb",
            "--in",
            in.path.c_str(),
            "--out",
            out.path.c_str(),
        }
            .data());

    RaiiTempNamedFile round_trip;
    main_kmb2kmx(
        6,
        std::vector<const char *>{
            "kickmix",
            "kmb2kmx",
            "--in",
            out.path.c_str(),
            "--out",
            round_trip.path.c_str(),
        }
            .data());

    ASSERT_EQ(round_trip.read_contents(), in.read_contents());
}
