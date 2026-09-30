#  Copyright 2026 Google LLC
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

"""
Iterates over modules and classes, listing their attributes and methods in markdown.
"""

import sys

from util_gen_stub_file import generate_documentation

import kickmix


def main():
    version = kickmix.__version__
    if "dev" in version or version == "VERSION_INFO" or "-dev" in sys.argv:
        version = "(Development Version)"
    else:
        version = "v" + version
    objects = [
        obj
        for obj in generate_documentation(obj=kickmix, full_name="kickmix", level=0)
        if all('[DEPRECATED]' not in line for line in obj.lines)
    ]

    print(f"# kickmix {version} API Reference")
    print()
    print("## Index")
    for obj in objects:
        level = obj.level
        print((level - 1) * "    " + f"- [`{obj.full_name}`](#{obj.full_name})")

    print('''
```python
# Types used by the method definitions.
from typing import overload, TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
import io
import pathlib
import numpy as np
```
'''.strip())

    for obj in objects:
        print()
        print(f'<a name="{obj.full_name}"></a>')
        print("```python")
        print(f'# {obj.full_name}')
        print()
        if len(obj.full_name.split('.')) > 2:
            print(f'# (in class {".".join(obj.full_name.split(".")[:-1])})')
        else:
            print('# (at top-level in the kickmix module)')
        print('\n'.join(obj.lines))
        print("```")


if __name__ == '__main__':
    main()
