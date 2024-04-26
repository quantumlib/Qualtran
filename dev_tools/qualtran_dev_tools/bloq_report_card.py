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
import time
import warnings
from typing import Any, Dict, Iterable, List, Optional, Set, Type

import pandas as pd
import pandas.io.formats.style

from qualtran import Bloq, BloqExample
from qualtran.testing import (
    BloqCheckResult,
    check_bloq_example_decompose,
    check_bloq_example_make,
    check_bloq_example_qtyping,
    check_bloq_example_serializes,
    check_equivalent_bloq_example_counts,
)

from .bloq_finder import get_bloq_classes, get_bloq_examples


def _get_package(bloq_cls: Type[Bloq]) -> str:
    """The package name for a bloq class"""
    return '.'.join(bloq_cls.__module__.split('.')[:-1])


def color_status(v: BloqCheckResult):
    """Used to style the dataframe."""
    if v is BloqCheckResult.PASS:
        return 'background-color:lightgreen'
    if v is BloqCheckResult.MISSING:
        return 'background-color:lightyellow'
    if v is BloqCheckResult.NA:
        return 'background-color:lightgrey'
    if v is BloqCheckResult.UNVERIFIED:
        return 'background-color:lightblue'

    return 'background-color:red'


def format_status(v: BloqCheckResult):
    """Used to format the dataframe."""
    return v.name.lower()


def bloq_classes_with_no_examples(
    bclasses: Iterable[Type[Bloq]], bexamples: Iterable[BloqExample]
) -> Set[Type[Bloq]]:
    ks = set(bclasses)
    for be in bexamples:
        try:
            ks.remove(be.bloq_cls)
        except KeyError:
            pass

    return ks


IDCOLS = ['package', 'bloq_cls', 'name']
CHECKCOLS = ['make', 'decomp', 'counts', 'serialize', 'qtyping']


def record_for_class_with_no_examples(k: Type[Bloq]) -> Dict[str, Any]:
    return {'bloq_cls': k.__name__, 'package': _get_package(k), 'name': '-'} | {
        check_name: BloqCheckResult.MISSING for check_name in CHECKCOLS
    }


def record_for_bloq_example(be: BloqExample) -> Dict[str, Any]:
    start = time.perf_counter()
    record = {
        'bloq_cls': be.bloq_cls.__name__,
        'package': _get_package(be.bloq_cls),
        'name': be.name,
        'make': check_bloq_example_make(be)[0],
        'decomp': check_bloq_example_decompose(be)[0],
        'counts': check_equivalent_bloq_example_counts(be)[0],
        'serialize': check_bloq_example_serializes(be)[0],
        'qtyping': check_bloq_example_qtyping(be)[0],
    }
    dur = time.perf_counter() - start
    if dur > 1.0:
        warnings.warn(f"{be.name} took {dur} to check.")
    return record


def show_bloq_report_card(df: pd.DataFrame) -> pandas.io.formats.style.Styler:
    return df.style.map(color_status, CHECKCOLS).format(format_status, CHECKCOLS)


def get_bloq_report_card(
    bclasses: Optional[Iterable[Type[Bloq]]] = None,
    bexamples: Optional[Iterable[BloqExample]] = None,
    package_prefix: str = 'qualtran.bloqs.',
) -> pd.DataFrame:

    if bclasses is None:
        bclasses = get_bloq_classes()
    if bexamples is None:
        bexamples = get_bloq_examples()
        # Default exclusions: pass explicit bexamples to override.
        skips = ['qubitization_qpe_hubbard_model_small', 'qubitization_qpe_hubbard_model_large']
        bexamples = [bex for bex in bexamples if bex.name not in skips]

    records: List[Dict[str, Any]] = []
    missing_bclasses = bloq_classes_with_no_examples(bclasses, bexamples)
    records.extend(record_for_class_with_no_examples(k) for k in missing_bclasses)
    records.extend(record_for_bloq_example(be) for be in bexamples)

    df = pd.DataFrame(records)
    df['package'] = df['package'].str.removeprefix(package_prefix)
    return df.sort_values(by=IDCOLS).loc[:, IDCOLS + CHECKCOLS].reindex()


def summarize_results(report_card: pd.DataFrame) -> pd.DataFrame:
    """Take a `report_card` data frame and return the number of times each status was noted."""
    summary = (
        pd.DataFrame([report_card[k].value_counts().rename(k) for k in CHECKCOLS])
        .fillna(0)
        .astype(int)
    )
    summary.columns = [v.name.lower() for v in summary.columns]
    return summary
