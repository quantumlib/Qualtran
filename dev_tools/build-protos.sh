#!/bin/bash

#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


################################################################################
# Generates python from protobuf definitions, including mypy stubs.
#
# Usage:
#     dev_tools/build-protos.sh [--check]
################################################################################

DOC="\
usage: dev_tools/build-protos.sh [options]

Options:

  --check       check if generated code is in sync with existing source code
                without overwriting any file.
  -h, --help    display this message and exit.
"

# process global options -----------------------------------------------------

opt_check=0

while (( $# )); do
    case "$1" in
        --check)
            opt_check=1
            ;;
        -h|--help)
            echo "$DOC"
            exit 0
            ;;
        *)
            echo "Unsupported positional argument '$1'" >&2
            exit 2
            ;;
    esac
    shift
done

# change to the src directory
thisdir="$(dirname "${BASH_SOURCE[0]}")" || exit $?
topdir="$(git -C "${thisdir}" rev-parse --show-toplevel)" || exit $?
cd "${topdir}/" || exit $?

# determine the output directory
outdir=.
if (( opt_check )); then
    outdir="$(mktemp -d /tmp/nqcr-build-protos.XXXXXXXX)" || exit $?
    echo "Using temporary directory ${outdir}" >&2
fi

# compile protos or bail out in case of error

python -m grpc_tools.protoc \
    -I=. --python_out="${outdir}" --mypy_out="${outdir}" qualtran/protos/*.proto ||
    exit $?

# handle the check mode if we get here
my_exit_code=0

if (( opt_check )); then
    typeset -a new_generated_files
    mapfile -t new_generated_files < <( \
        cd "${outdir}" && find . -name '*_pb2.py*' -type f | sed 's,^[.]/,,'
    )
    typeset -a old_generated_files
    mapfile -t old_generated_files < <( git ls-files '*_pb2.py*' )
    if (( ${#new_generated_files[@]} != ${#old_generated_files[@]} )); then
        printf "Unequal number of generated files:\n  %i in %s\n  %i in %s\n" \
            ${#new_generated_files[@]} "${outdir}" \
            ${#old_generated_files[@]} "${PWD}" >&2
        my_exit_code=1
    fi
    for f in "${new_generated_files[@]}"; do
        cmp "${outdir}/${f}" "${topdir}/src/${f}" || my_exit_code=1
    done
    # clean up after a successful check
    if (( my_exit_code == 0 )); then
        echo "Protobuf-generated python code is up to date." >&2
        # double check we only delete the temporary files
        if [[ "${outdir}" == /tmp/nqcr-build-protos* ]]; then
            rm -r "${outdir}"
        fi
    fi
fi

exit "${my_exit_code}"
