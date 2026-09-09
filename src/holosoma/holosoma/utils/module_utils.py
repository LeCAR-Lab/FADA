import os

import holosoma.utils


def get_holosoma_root() -> str:
    # Use holosoma.utils since holosoma sometimes returns a namespaced module, and its __file__ is None
    return os.path.dirname(os.path.dirname(holosoma.utils.__file__))
