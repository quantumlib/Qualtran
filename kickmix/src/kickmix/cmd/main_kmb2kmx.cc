#include <iostream>
#include <random>

#include "kickmix/circuit/circuit.h"
#include "kickmix/cmd/main_util.h"
#include "kickmix/util/arg_parse.h"

using namespace kickmix;

int kickmix::main_kmb2kmx(int argc, const char **argv) {
    try {
        check_for_unknown_arguments(
            {
                "--in",
                "--out",
            },
            {},
            "kickmix",
            "kmb2kmx",
            argc,
            argv);
    } catch (const std::invalid_argument &ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }

    FILE *in = find_open_file_argument("--in", stdin, "rb", argc, argv);
    FILE *out = find_open_file_argument("--out", stdout, "wb", argc, argv);

    Circuit circuit = Circuit::from_kmb_file(in);
    circuit.write_kmx_to(out);

    if (in != stdin) {
        fclose(in);
    }
    if (out != stdout) {
        fclose(out);
    }

    return 0;
}
