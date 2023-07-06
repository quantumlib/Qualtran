import argparse

from qualtran_dev_tools.notebook_execution import execute_and_export_notebooks


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output-nbs', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--output-html', action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args()
    execute_and_export_notebooks(output_nbs=args.output_nbs, output_html=args.output_html)


if __name__ == '__main__':
    parse_args()
