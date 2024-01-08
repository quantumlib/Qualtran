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

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Qualtran'
copyright = '2023, Google LLC'
author = 'Google Quantum AI'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_nb']

nb_execution_mode = 'off'
myst_enable_extensions = ['dollarmath', 'amsmath', 'deflist']
myst_dmath_double_inline = True

suppress_warnings = [
    # The markdown parser myst gets mad when you skip to small headers
    # e.g. <h4> for "parameters" section.
    "myst.header",
    # In `show_counts_sigma`, we use a <h4> inside an output cell.
    # Myst handles this well now:
    # https://myst-parser.readthedocs.io/en/v0.17.2/develop/_changelog.html#internal-improvements
    # but still emits a warning
    "myst.nested_header",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ['fixes.css']
