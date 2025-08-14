#  Copyright 2024 Google LLC
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
from typing import Dict


def get_aliases_str(pref_dotpath: str, *, aliases_d: Dict[str, str]) -> str:
    """From a complete dictionary of aliases, return a formatted string with our aliases.

    This will return just `pref_dotpath` if there are none,
    otherwise '{pref} **Alias(es):** {...}'
    """
    aliases = [k for k, v in aliases_d.items() if v == pref_dotpath and k != pref_dotpath]
    s = f'`{pref_dotpath}`'
    alias_str = ', '.join(f'`{a}`' for a in aliases)
    if len(aliases) > 1:
        s += f". **Aliases:** {alias_str}"
    elif len(aliases) == 1:
        s += f". **Alias:** {alias_str}"
    return s
