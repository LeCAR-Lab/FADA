"""Pytest configuration to ensure proper import order for isaacgym compatibility."""

# Import torch safely before any isaacgym imports during test collection
from holosoma.utils.safe_torch_import import torch  # noqa: F401


def pytest_configure(config):
    """Register custom markers for pytest."""
    config.addinivalue_line(
        "markers", "isaacsim: marks tests as requiring Isaac Sim (deselect with '-m \"not isaacsim\"')"
    )
    config.addinivalue_line(
        "markers", "multi_gpu: marks tests as requiring multiple GPUs (deselect with '-m \"not multi_gpu\"')"
    )
