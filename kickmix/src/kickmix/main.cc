#include <cstring>
#include <iostream>

#include "cmd/main_util.h"

using namespace kickmix;

int main(int argc, const char **argv) {
    if (argc > 1) {
        try {
            if (strcmp(argv[1], "circuit") == 0) {
                return main_generate_circuit(argc, argv);
            } else if (strcmp(argv[1], "count_operations") == 0) {
                return main_count_operations(argc, argv);
            } else if (strcmp(argv[1], "kmb2kmx") == 0) {
                return main_kmb2kmx(argc, argv);
            } else if (strcmp(argv[1], "kmx2kmb") == 0) {
                return main_kmx2kmb(argc, argv);
            } else if (strcmp(argv[1], "count_operations_sampled") == 0) {
                return main_count_operations_sampled(argc, argv);
            } else if (strcmp(argv[1], "sample") == 0) {
                return main_sample(argc - 1, argv + 1);
            } else if (strcmp(argv[1], "spin") == 0) {
                return main_spin(argc, argv);
            } else if (strcmp(argv[1], "diagram") == 0) {
                return main_diagram(argc, argv);
            }
        } catch (const std::invalid_argument &ex) {
            std::cerr << ex.what() << "\n";
            return 1;
        }
    }

    if (argc > 1) {
        std::cerr << "Unrecognized subcommand. Known subcommands are:\n";
    } else {
        std::cerr << "Specify a subcommand. Known subcommands are:\n";
    }
    std::cerr << "    kickmix circuit                       # generate a circuit\n";
    std::cerr << "    kickmix diagram                       # make a picture of a circuit\n";
    std::cerr << "    kickmix count_operations              # count instructions in a circuit\n";
    std::cerr << "    kickmix count_operations_sampled      # count instructions executed while simulating a circuit\n";
    std::cerr << "    kickmix kmb2kmx                       # reads a kmb file and writes out an equivalent kmx file\n";
    std::cerr << "    kickmix kmx2kmb                       # reads a kmx file and writes out an equivalent kmb file\n";
    std::cerr << "    kickmix sample                        # simulate a circuit\n";
    std::cerr << "    kickmix spin                          # time how long it takes to simulate a circuit\n";
    return 1;
}
