# Regular-package marker: `setup.py` uses `find_packages()`, which only walks directories
# that have one, so without this file this directory is absent from the built wheel.
#
# `data_utils/` does NOT get one: the README tells the reader to `git clone` a
# third-party LAFAN repository into it, and making it a package would sweep that clone
# into any wheel built from a working copy that followed those instructions.
