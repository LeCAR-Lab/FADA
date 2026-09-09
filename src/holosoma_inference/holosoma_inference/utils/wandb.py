from __future__ import annotations

import re
from pathlib import Path

import wandb

_WANDB_PREFIX = "wandb://"
_WANDB_HTTPS_PATTERN = re.compile(r"https://[^/]+/([^/]+)/([^/]+)/runs/([^/]+)/files/(.+)")


def load_checkpoint(
    wandb_run_path: str | None,
    checkpoint: str,
    log_dir: str,
) -> Path:
    """Download checkpoint from W&B or use local checkpoint.

    Parameters
    ----------
    wandb_run_path : str | None
        Path to the W&B run (e.g., 'username/project/run_id'). If None, checkpoint must be provided.
    checkpoint : str
        Name of checkpoint file in W&B run or path to local checkpoint file.
    log_dir : str
        Directory to save downloaded checkpoint.

    Returns
    -------
    Path
        Path to the downloaded or local checkpoint file.
    """

    if checkpoint.startswith(_WANDB_PREFIX):
        try:
            wandb_entity, wandb_project, wandb_run_id, checkpoint = checkpoint[len(_WANDB_PREFIX) :].split("/", 3)
        except ValueError:
            raise ValueError(
                f"Invalid wandb checkpoint path: {checkpoint}. "
                f"Expected format: {_WANDB_PREFIX}<entity>/<project>/<run_id>/<checkpoint_name>"
            )
        wandb_run_path = f"{wandb_entity}/{wandb_project}/{wandb_run_id}"
    elif match := _WANDB_HTTPS_PATTERN.match(checkpoint):
        wandb_entity, wandb_project, wandb_run_id, checkpoint = match.groups()
        wandb_run_path = f"{wandb_entity}/{wandb_project}/{wandb_run_id}"

    if wandb_run_path is not None:
        api = wandb.Api()
        run = api.run(wandb_run_path)
        # Create log dir
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        # Download checkpoint to log_dir
        checkpoint_file = run.file(checkpoint)  # Get the specific checkpoint file
        checkpoint_file.download(root=log_dir, replace=True)
        print(f"Finished downloading checkpoint {checkpoint} to {log_dir} from W&B run {wandb_run_path}")
        checkpoint_path = log_dir_path / checkpoint
    else:
        checkpoint_path = Path(checkpoint)
    return checkpoint_path
