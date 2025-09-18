#  Copyright 2025 Google LLC
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

import argparse
import json
import sys


def escape_lines(input_stream, output_stream):
    """
    Reads each line from the input stream, escapes it for a textproto bytes
    field, and writes the result to the output stream.
    """
    for line in input_stream:
        # json.dumps is perfect for this task. It:
        # 1. Encloses the string in double quotes.
        # 2. Escapes backslashes (\\), double quotes (\"), etc.
        # 3. Converts non-printing chars like newlines to escape codes (\n).
        escaped_line = json.dumps(line)
        output_stream.write(escaped_line + '\n')


def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description='Escape each line of a file to be used in a textproto bytes field.'
    )
    parser.add_argument(
        'infile',
        nargs='?',
        type=argparse.FileType('r'),
        default=sys.stdin,
        help='Input file to process (reads from stdin if not provided).',
    )
    parser.add_argument(
        'outfile',
        nargs='?',
        type=argparse.FileType('w'),
        default=sys.stdout,
        help='Output file to write to (writes to stdout if not provided).',
    )
    args = parser.parse_args()

    try:
        escape_lines(args.infile, args.outfile)
    finally:
        # Close files if they aren't stdin/stdout
        if args.infile is not sys.stdin:
            args.infile.close()
        if args.outfile is not sys.stdout:
            args.outfile.close()


if __name__ == '__main__':
    main()
