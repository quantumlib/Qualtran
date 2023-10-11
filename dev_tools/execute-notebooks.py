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

import argparse

from qualtran_dev_tools.notebook_execution import execute_and_export_notebooks


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output-nbs', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--output-html', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--only-out-of-date', action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    execute_and_export_notebooks(
        output_nbs=args.output_nbs,
        output_html=args.output_html,
        only_out_of_date=args.only_out_of_date,
    )


if __name__ == '__main__':
    parse_args()
