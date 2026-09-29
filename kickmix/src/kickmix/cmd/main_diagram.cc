#include <iostream>

#include "kickmix/circuit/circuit.h"
#include "kickmix/util/arg_parse.h"
#include "main_util.h"

using namespace kickmix;

int kickmix::main_diagram(int argc, const char **argv) {
    try {
        check_for_unknown_arguments(
            {
                "--in",
                "--out",
                "--kind",
            },
            {},
            "kickmix",
            "diagram",
            argc,
            argv);
    } catch (const std::invalid_argument &ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }

    std::string_view kind = find_enum_argument<std::string_view>(
        "--kind",
        "text",
        {
            {"text", "text"},
            {"html", "html"},
            {"svg", "svg"},
        },
        argc,
        argv);

    FILE *in = find_open_file_argument("--in", stdin, "rb", argc, argv);
    FILE *out = find_open_file_argument("--out", stdout, "wb", argc, argv);
    Circuit circuit = Circuit::from_kmx_or_kmb_file(in);

    std::stringstream ss;
    if (kind == "html") {
        circuit.write_svg_or_html_diagram_to(ss, true);
    } else if (kind == "svg") {
        circuit.write_svg_or_html_diagram_to(ss, false);
    } else {
        circuit.write_text_diagram_to(ss);
    }
    ss << "\n";
    std::string text = ss.str();
    bool failed = fwrite(text.data(), text.size(), 1, out) != 1;
    if (failed) {
        throw std::invalid_argument("Error while writing output.");
    }

    if (in != stdin) {
        fclose(in);
    }
    if (out != stdout) {
        fclose(out);
    }

    return 0;
}
