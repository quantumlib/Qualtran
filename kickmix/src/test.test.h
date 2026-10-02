#ifndef KICKMIX_UTIL_TEST_H
#define KICKMIX_UTIL_TEST_H

#include <random>

#include "gtest/gtest.h"

std::mt19937_64 externally_seeded_rng();

std::mt19937_64 INDEPENDENT_TEST_RNG();

std::string rewind_read_close(FILE *f);

std::string resolve_testdata_file_path(std::string_view name);

struct RaiiTempNamedFile {
    int descriptor;
    std::string path;
    RaiiTempNamedFile();
    ~RaiiTempNamedFile();
    explicit RaiiTempNamedFile(std::string_view contents);
    std::string read_contents();
    std::vector<uint8_t> read_bytes();
    void write_contents(std::string_view contents);
};

#endif
