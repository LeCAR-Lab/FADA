# Regular-package marker. `setup.py` discovers packages with `find_packages()`, which
# only walks directories that have one, so without this file this directory and
# everything under it is absent from the built wheel. Editable installs are unaffected
# because they read the checkout directly.
