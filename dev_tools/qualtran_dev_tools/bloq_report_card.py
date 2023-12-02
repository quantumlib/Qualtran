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
from typing import Any, Dict, Iterable, Optional, Set, Type

import pandas as pd
import pandas.io.formats.style

from qualtran import Bloq, BloqExample
from qualtran.testing import BloqCheckResult, check_bloq_example_decompose, check_bloq_example_make

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
CHECKCOLS = ['make', 'decomp']


def record_for_class_with_no_examples(k: Type[Bloq]) -> Dict[str, Any]:
    return {
        'bloq_cls': k.__name__,
        'package': _get_package(k),
        'name': '-',
        'make': BloqCheckResult.MISSING,
        'decomp': BloqCheckResult.MISSING,
        # 'counts': BloqCheckResult.MISSING,
    }


def records_for_bloq_example(be: BloqExample) -> Dict[str, Any]:
    return {
        'bloq_cls': be.bloq_cls.__name__,
        'package': _get_package(be.bloq_cls),
        'name': be.name,
        'make': check_bloq_example_make(be)[0],
        'decomp': check_bloq_example_decompose(be)[0],
        # 'counts': check_equivalent_bloq_example_counts(be)[0],
    }


def show_bloq_report_card(df: pd.DataFrame) -> pandas.io.formats.style.Styler:
    return df.style.applymap(color_status, CHECKCOLS).format(format_status, CHECKCOLS)


def get_bloq_report_card(
    bclasses: Optional[Iterable[Type[Bloq]]] = None,
    bexamples: Optional[Iterable[BloqExample]] = None,
) -> pd.DataFrame:
    if bclasses is None:
        bclasses = get_bloq_classes()
    if bexamples is None:
        bexamples = get_bloq_examples()

    records = []
    missing_bclasses = bloq_classes_with_no_examples(bclasses, bexamples)
    records.extend(record_for_class_with_no_examples(k) for k in missing_bclasses)
    records.extend(records_for_bloq_example(be) for be in bexamples)

    return pd.DataFrame(records).sort_values(by=IDCOLS).loc[:, IDCOLS + CHECKCOLS]
