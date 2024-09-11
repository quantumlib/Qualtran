# Releasing Qualtran

The steps are

 1. Bump the version number to the desired release version
 2. Tag the release
 3. Follow the Python packaging guide to generate a distribution and upload it
 4. Bump the version number to the next "dev" version

The packaging guide instructions can be found at:
https://packaging.python.org/en/latest/tutorials/packaging-projects/#generating-distribution-archives

## Set the version number

Open a PR to edit `qualtran/_version.py` to change the version string to "x.y.z" (without any dev specifier), which
is the desired version number.
It should have previously been set to "x.y.z.dev0".
The PR should be merged to the `main` branch and care should be taken not to merge any other PRs until
the entire process is completed.
Draft the release notes and circulate them for review.

## Tag the repository

Use GitHub "Releases" to tag the release with release notes. Tags for "x.y.0" minor releases should go on `main`.
Patch releases (z > 0) should be tagged against a `vx.y` branch unless the patch immediately follows a minor release.

Follow the convention for tag names. Include the "v" in the tag name. 

## Package the release

Make sure you're on `main` and have pulled the version-bump commit.
It's recommended to carefully run `git clean -ndx` (change n to f to actually do it) to prepare
a pristine repository state.
Then, follow the guidance at 
https://packaging.python.org/en/latest/tutorials/packaging-projects/#generating-distribution-archives.

Modify the command to upload to the main pypi repository (rather than the test repository used for the tutorial).

## Bump the version number to dev

Edit `_version.py` again, and change the version number from "x.y.z" to "x.(y+1).0.dev0".

## Communicate

Send an email to [qualtran-announce@googlegroups.com](https://groups.google.com/g/qualtran-announce) announcing 
the release. Congrats!