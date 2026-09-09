import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from loguru import logger


class InferenceLogger:
    """Lightweight logger that mirrors eval callback output for inference runs."""

    def __init__(self, output_dir: str, filename: str = "state_log"):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        suffix = "" if filename.endswith(".npz") else ".npz"
        self.filepath = self.output_dir / f"{filename}{suffix}"
        # Avoid overwriting existing file by adding timestamp if needed
        if self.filepath.exists():
            self.filepath = self.output_dir / f"{filename}_{timestamp}{suffix}"

        self.state_log = defaultdict(list)

    def reset_session(self):
        """Clear in-memory state for a new session while keeping the same output path."""
        self.state_log = defaultdict(list)

    def log_states(self, data: dict):
        for key, value in data.items():
            self.state_log[key].append(value)

    def save(self):
        if not self.state_log:
            logger.info("No inference states to save.")
            return None

        save_dict = {}
        for key, values in self.state_log.items():
            if not values:
                continue
            first = values[0]
            if torch.is_tensor(first):
                stacked = torch.stack([v if torch.is_tensor(v) else torch.as_tensor(v) for v in values])
                save_dict[key] = stacked.cpu().numpy()
            else:
                save_dict[key] = np.array(values)

        np.savez_compressed(self.filepath, **save_dict)
        logger.info(f"Inference log saved to {self.filepath}")
        return self.filepath
