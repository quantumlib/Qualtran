# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

######## Configuration ########

# Before we go any further, test if certain programs are available.
PROGRAMS_NEEDED = sphinx-build sphinx-autobuild
TEST := $(foreach p,$(PROGRAMS_NEEDED),\
	  $(if $(shell which $(p)),_,$(error Cannot find program "$(p)")))

# The first two can be overridden via env vars.
SPHINXOPTS     ?=
SPHINXBUILD    ?= sphinx-build
SPHINXAUTO     = sphinx-autobuild
SPHINXAUTOOPTS = --open-browser -j auto
SOURCEDIR      = .
BUILDDIR       = _build

.PHONY: help Makefile

######## Targets ########

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

auto autobuild live livehtml:
	@$(SPHINXAUTO) "$(SOURCEDIR)" "$(BUILDDIR)"/html $(SPHINXOPTS) $(SPHINXAUTOOPTS) $(O)

# The next one is for working on a theme, when it's best to disable incremental builds using "-a".
# The build times are much slower, but CSS and template changes are picked up, and that's critical.
# C.f. https://github.com/sphinx-doc/sphinx-autobuild#working-on-a-sphinx-html-theme
auto-dev:
	@$(SPHINXAUTO) -a "$(SOURCEDIR)" "$(BUILDDIR)"/html $(SPHINXOPTS) $(SPHINXAUTOOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
