#include <iostream>
#include <random>

#include "kickmix/circuit/circuit.h"
#include "kickmix/cmd/main_util.h"
#include "kickmix/util/arg_parse.h"

using namespace kickmix;

int kickmix::main_kmx2kmb(int argc, const char **argv) {
    check_for_unknown_arguments(
        {
            "--in",
            "--out",
        },
        {},
        "kickmix",
        "kmx2kmb",
        argc,
        argv);

    FILE *in = find_open_file_argument("--in", stdin, "rb", argc, argv);
    FILE *out = find_open_file_argument("--out", stdout, "wb", argc, argv);

    Circuit circuit = Circuit::from_kmx_file(in);
    circuit.write_kmb_to(out);

    if (in != stdin) {
        fclose(in);
    }
    if (out != stdout) {
        fclose(out);
    }
    return 0;
}
