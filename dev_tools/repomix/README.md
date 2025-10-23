# Qualtran Repomix

We use the [repomix](https://github.com/yamadashy/repomix) tool to concatenate relevant
context so LLMs can write Qualtran code.

The repository is too large to naively include everything in the context window, so we're
a little selective. The packed output contains:

 - Reference docs for core modules and subpackages, via markdown. You must first run
   `dev_tools/build-reference-docs-2.py`.
 - Documentation notebooks (bloq notebooks and concept notebooks), via markdown. You must
   first run `dev_tools/execute-notebooks.py --output-md`.
 - Selected `*.py` files in `qualtran/bloqs/**/*.py`. The list is enumerated in
   `repomix-driver.py`.

After ensuring the docs are rendered in markdown format, run `python repomix-driver.py`. This
will run the repomix program, which requires Node.js.


--------------

You may want to run `git clean -ndX` (change `n` to `f`) in the `docs/` directory before 
running the markdown-export scripts to get a fresh build of the docs. `repomix-driver.py` will 
include  anything with a `*.md` extension, including any vestigial files. 