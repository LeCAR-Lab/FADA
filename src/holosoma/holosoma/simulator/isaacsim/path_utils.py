"""Path transformation utilities for USD scene loading.

This module provides utilities for transforming USD prim paths to IsaacLab
spawned path patterns and normalizing path cases for consistent lookup.
"""

from typing import List


def transform_usd_path_to_spawned_path(usd_prim_path: str, strip_prefixes: List[str]) -> str:
    """Transform USD prim path to IsaacLab spawned path pattern.

    Converts USD prim paths to the pattern format expected by IsaacLab's
    environment system, stripping specified prefixes and adding environment patterns.

    Parameters
    ----------
    usd_prim_path : str
        Original USD path like "/world/obj0_0".
    strip_prefixes : List[str]
        Prefixes to strip, e.g., ["/world/", "/scene/world/"].

    Returns
    -------
    str
        Spawned path pattern like "/World/envs/env_.*/obj0_0".
    """
    clean_path = usd_prim_path

    # Try each prefix in order
    for prefix in strip_prefixes:
        if clean_path.startswith(prefix):
            clean_path = clean_path[len(prefix) :]
            break

    # Add IsaacLab environment prefix
    return f"/World/envs/env_.*/{clean_path}"


def normalize_usd_path_case(config_path: str) -> str:
    """Normalize USD path case for consistent lookup.

    Converts path cases to a standardized format for consistent configuration
    lookup and path matching operations.

    Parameters
    ----------
    config_path : str
        Path from config like "/World/obj0_0".

    Returns
    -------
    str
        Normalized path like "/world/obj0_0".
    """
    return config_path.replace("/World/", "/world/")
