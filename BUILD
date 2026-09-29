load("@rules_python//python:packaging.bzl", "py_wheel")

package(default_visibility = ["//visibility:public"])

SOURCE_FILES_NO_MAIN = glob(
    [
        "kickmix/src/**/*.cc",
        "kickmix/src/**/*.h",
        "kickmix/src/**/*.inl",
    ],
    exclude = glob([
        "kickmix/src/**/*.test.cc",
        "kickmix/src/**/*.test.h",
        "kickmix/src/**/*.perf.cc",
        "kickmix/src/**/*.perf.h",
        "kickmix/src/**/*.pybind.cc",
        "kickmix/src/**/*.pybind.h",
        "kickmix/src/**/main.cc",
    ]),
)

TEST_FILES = glob(
    [
        "kickmix/src/**/*.test.cc",
        "kickmix/src/**/*.test.h",
    ],
)

PERF_FILES = glob(
    [
        "kickmix/src/**/*.perf.cc",
        "kickmix/src/**/*.perf.h",
    ],
)

PYBIND_FILES = glob(
    [
        "kickmix/src/**/*.pybind.cc",
        "kickmix/src/**/*.pybind.h",
    ],
)

cc_binary(
    name = "kickmix",
    srcs = SOURCE_FILES_NO_MAIN + ["kickmix/src/kickmix/main.cc"],
    copts = [
        "-O3",
        "-std=c++20",
        "-fno-strict-aliasing",
        "-march=native",
    ],
    includes = ["kickmix/src/"],
    deps = [
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "kickmix_perf",
    srcs = SOURCE_FILES_NO_MAIN + PERF_FILES,
    copts = [
        "-O3",
        "-std=c++20",
        "-fno-strict-aliasing",
        "-march=native",
    ],
    data = glob(["testdata/**"]),
    includes = ["kickmix/src/"],
)

cc_test(
    name = "kickmix_test",
    srcs = SOURCE_FILES_NO_MAIN + TEST_FILES,
    copts = [
        "-O1",
        "-std=c++20",
        "-fno-strict-aliasing",
        "-fno-omit-frame-pointer",
        "-fsanitize=undefined",
        "-fsanitize=address",
        "-march=native",
    ],
    data = glob(["testdata/**"]),
    includes = ["kickmix/src/"],
    linkopts = [
        "-fsanitize=undefined",
        "-fsanitize=address",
    ],
    deps = [
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "kickmix.so",
    srcs = SOURCE_FILES_NO_MAIN + PYBIND_FILES,
    copts = [
        "-O3",
        "-std=c++20",
        "-fvisibility=hidden",
        "-fno-strict-aliasing",
        "-march=native",
    ],
    includes = ["kickmix/src/"],
    # Python supplies the extension's Python API symbols when importing it.
    linkopts = select({
        "@platforms//os:osx": ["-Wl,-undefined,dynamic_lookup"],
        "//conditions:default": [],
    }),
    linkshared = 1,
    deps = ["@pybind11"],
)

py_wheel(
    name = "kickmix_dev_wheel",
    distribution = "kickmix",
    requires = ["numpy"],
    version = "0.0.dev0",
    deps = [
        ":kickmix.so",
    ],
)
