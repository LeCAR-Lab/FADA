"""
Data Collector for Deploy/Evaluation
Supports both single and multi-environment modes
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from loguru import logger

# Values of the H5 file attribute `collection_status`. Written once at file creation and
# rewritten once at close(), so the value a reader finds says how the writing session ended:
#   in_progress            -- the writer never reached close(): SIGKILL, OOM, a crashed host.
#   complete               -- closed, and it recorded at least the steps it planned to.
#   truncated              -- closed cleanly, but short of its planned step count.
#   closed_unknown_length  -- closed cleanly; the caller supplied no planned step count, so
#                             completeness is not a question this file can answer.
COLLECTION_STATUS_IN_PROGRESS = "in_progress"
COLLECTION_STATUS_COMPLETE = "complete"
COLLECTION_STATUS_TRUNCATED = "truncated"
COLLECTION_STATUS_CLOSED_UNKNOWN_LENGTH = "closed_unknown_length"

# JSON list attribute: one entry per collection session that has ever written to this
# file, oldest first.
#
# The four attributes above are FILE-level and are rewritten every time the file is
# reopened, so in a file that two sessions appended to they describe only the second,
# while the file still holds every episode of the first. This list accumulates instead,
# one entry per session. Each episode group also carries `collection_session`, the
# 0-based index into this list, attributing the episode to the session that produced it.
#
# Reporting only: nothing here changes `dones`, the episode datasets, or which windows
# the finetune trains on.
COLLECTION_SESSIONS_ATTR = "collection_sessions"

# A run given `--task.max-steps N` records N-1 steps: the loop's stop check fires
# before the final iteration is recorded. `planned_steps` holds the number passed on
# the command line, so completeness comparisons allow the collected count to fall this
# many steps short of it. Readers of the attributes must apply the same allowance.
PLANNED_STEPS_TOLERANCE = 1


class DataCollector:
    """
    Data collector for both eval and deploy modes.
    
    Supports:
    - Single or multiple parallel environments
    - Variable-length episodes (no fixed step count)
    - Automatic episode management per environment
    - Individual observation groups saved separately
    - HDF5 format with batch saving
    """

    def __init__(
        self,
        output_dir: str,
        dataset_name: str = "inference_dataset",
        compress: bool = True,
        batch_size: int = 10,
        robot_type: str = "unknown",
        simulator: str = "unknown",
        policy_checkpoint: str | None = None,
        num_envs: int = 1,
        obs_dict: dict[str, list[str]] | None = None,
        skip_obs_keys: tuple[str, ...] = ("actor_obs", "critic_obs"),
        planned_steps: int | None = None,
        planned_steps_source: str = "unknown",
    ):
        """
        Initialize data collector.
        
        Args:
            output_dir: Directory to save dataset
            dataset_name: Name of the dataset
            compress: Whether to compress HDF5 data
            batch_size: Number of episodes to batch before saving
            robot_type: Type of robot
            simulator: Simulator name
            policy_checkpoint: Path to policy checkpoint
            num_envs: Number of parallel environments (1 for deploy, N for eval)
            obs_dict: Observation dictionary mapping group names to observation components.
                     If provided, will save each observation group separately.
                     Example: {"base_lin_vel_obs": ["base_lin_vel"], "base_ang_vel_obs": ["base_ang_vel"]}
            skip_obs_keys: Observation group keys to skip from saving.
            planned_steps: How many control steps this collection session intends to record,
                     if the caller knows (e.g. `--task.max-steps`). Recorded in the file.
                     None or <=0 means "not knowable", and is recorded as such.
            planned_steps_source: Where `planned_steps` came from, recorded verbatim.
        """
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.compress = compress
        self.batch_size = batch_size
        self.robot_type = robot_type
        self.simulator = simulator
        self.policy_checkpoint = policy_checkpoint
        self.num_envs = num_envs
        self.skip_obs_keys = set(skip_obs_keys)
        # Filter out keys we never want to save (e.g., actor/critic obs) up front
        self.obs_dict = {k: v for k, v in (obs_dict or {}).items() if k not in self.skip_obs_keys}
        
        # Track environments that are done (for marking dones=True)
        # We continue collecting to the same episode after reset,
        # but mark reset transitions as dones=True for filtering
        # Initialize as [num_envs, 1] boolean array, all False
        self.reset_envs: np.ndarray = np.zeros((self.num_envs, 1), dtype=bool)
        
        # Track whether rewards are available (set on first collect_step call)
        self.has_rewards: bool | None = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Episode tracking: single episode for all environments (batch format [num_envs, ...])
        # For inference (num_envs=1), format is [1, ...]
        self.current_episode: dict[str, list] | None = None
        self.episode_id_counter = 0
        self.episodes_buffer: list[dict] = []
        
        # File handles
        self.h5_file: h5py.File | None = None
        self.h5_episodes_group: h5py.Group | None = None
        # Total episodes present in the H5 file (existing episodes from a prior run that
        # this session appended to, plus whatever this session itself saves).
        self.saved_episode_count = 0
        # Episodes/steps produced *by this session* -- i.e. since this DataCollector
        # instance was constructed (or since the last reset_session()). Unlike
        # saved_episode_count, these start at 0 even when appending to an existing H5 file,
        # so callers can tell "collected nothing this run" apart from "the file already had
        # data from a previous run".
        self.new_episode_count = 0
        self.new_step_count = 0

        # Completion provenance. A collection stopped early -- SIGTERM, a killed pipeline
        # stage, a user's Ctrl-C -- takes the policy's `finally` path, so it flushes and
        # closes cleanly and leaves a structurally valid H5 indistinguishable from a
        # complete one by layout alone. `planned_steps` is what the run intended,
        # `collected_steps` is what it recorded, and `collection_status` is written
        # "in_progress" at file creation and only rewritten at close() -- so a hard kill
        # (SIGKILL, OOM, power) leaves "in_progress" behind, and a clean early stop leaves
        # "truncated".
        self.planned_steps = int(planned_steps) if planned_steps is not None and int(planned_steps) > 0 else None
        self.planned_steps_source = str(planned_steps_source)

        # Initialize HDF5 file
        self._init_h5_file()
        
        logger.info(f"Data collector initialized: {self.output_dir / self.dataset_name}.h5")
        logger.info(f"  Mode: {'Multi-env' if num_envs > 1 else 'Single-env'} ({num_envs} environments)")
        if self.obs_dict:
            logger.info(f"  Observation groups: {list(self.obs_dict.keys())}")

    def _init_h5_file(self):
        """Initialize HDF5 file and groups."""
        save_path = self.output_dir / f"{self.dataset_name}.h5"
        
        # Create file if it doesn't exist
        if not save_path.exists():
            self.h5_file = h5py.File(save_path, 'w')
            
            # Save metadata
            metadata = {
                "robot_type": self.robot_type,
                "simulator": self.simulator,
                "policy_checkpoint": str(self.policy_checkpoint) if self.policy_checkpoint else "none",
                "output_format": "h5",
                "num_envs": self.num_envs,
                "has_rewards": False,  # Will be updated on first collect_step if rewards are available
            }
            
            # Save observation dictionary if provided
            if self.obs_dict:
                metadata["obs_dict"] = self.obs_dict
            
            for key, value in metadata.items():
                if isinstance(value, (dict, list)):
                    self.h5_file.attrs[key] = json.dumps(value)
                else:
                    self.h5_file.attrs[key] = value
            
            # Create episodes group
            self.h5_episodes_group = self.h5_file.create_group("episodes")
        else:
            # Append to existing file
            self.h5_file = h5py.File(save_path, 'a')
            if "episodes" in self.h5_file:
                self.h5_episodes_group = self.h5_file["episodes"]
                # Count existing episodes
                self.saved_episode_count = len(self.h5_episodes_group.keys())
                # Seed episode_id_counter past any episode ids already in the file so this
                # session's new episodes don't collide with ones saved by a previous run.
                existing_ids = [
                    int(name[len("episode_"):])
                    for name in self.h5_episodes_group.keys()
                    if name.startswith("episode_") and name[len("episode_"):].isdigit()
                ]
                if existing_ids:
                    self.episode_id_counter = max(existing_ids) + 1
            else:
                self.h5_episodes_group = self.h5_file.create_group("episodes")

        # Claim a slot in the file's session history BEFORE the first stamp, so the
        # file-level attrs and the per-session entry are always written together.
        self._session_index = len(self._read_session_history())
        self._session_episode_ids: list[int] = []

        # Stamped for both branches, including append-to-an-existing-dataset: the file-level
        # verdict describes this session, so a resumed file does not keep the previous one's.
        self._write_session_provenance(COLLECTION_STATUS_IN_PROGRESS)

    def _read_session_history(self) -> list[dict[str, Any]]:
        """Session records already in the file, oldest first; `[]` if it has none.

        Never raises: a missing or unparseable attribute reads back as `[]`.
        """
        if self.h5_file is None:
            return []
        raw = self.h5_file.attrs.get(COLLECTION_SESSIONS_ATTR)
        if raw is None:
            return []
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return []
        return parsed if isinstance(parsed, list) else []

    def _write_session_provenance(self, status: str) -> None:
        """Record what this session intended, what it has, and whether it finished.

        Re-stamped on every batch save, which keeps `collected_steps` roughly current in a
        file whose writer is killed rather than closed.

        Writes both the file-level attributes (this session's verdict) and this session's
        entry in the accumulating `collection_sessions` list.
        """
        if self.h5_file is None:
            return
        try:
            self.h5_file.attrs["collection_status"] = status
            self.h5_file.attrs["planned_steps"] = int(self.planned_steps) if self.planned_steps else -1
            self.h5_file.attrs["planned_steps_source"] = self.planned_steps_source
            self.h5_file.attrs["collected_steps"] = int(self.new_step_count)

            history = self._read_session_history()
            record = {
                "session_index": int(self._session_index),
                "collection_status": status,
                "planned_steps": int(self.planned_steps) if self.planned_steps else -1,
                "planned_steps_source": self.planned_steps_source,
                "collected_steps": int(self.new_step_count),
                "episodes": int(self.new_episode_count),
                "episode_ids": [int(ep_id) for ep_id in self._session_episode_ids],
            }
            # Rewrite our own slot in place; never touch an earlier session's entry. If the
            # list is shorter than our index (a truncated/rewritten attribute), pad with
            # explicit "lost" placeholders rather than renumbering sessions.
            while len(history) < self._session_index:
                history.append({"session_index": len(history), "collection_status": None})
            if len(history) == self._session_index:
                history.append(record)
            else:
                history[self._session_index] = record
            self.h5_file.attrs[COLLECTION_SESSIONS_ATTR] = json.dumps(history)

            self.h5_file.flush()
        except Exception as exc:  # recording provenance never raises out of this method
            logger.warning(f"Could not record collection provenance: {exc}")

    # A run that asks for `--task.max-steps N` records N-1 steps: the policy loop's stop
    # check fires before the final iteration is recorded. `planned_steps` keeps the number
    # the user typed, so this tolerance absorbs that one step in the completeness comparison.
    _PLANNED_STEPS_TOLERANCE = PLANNED_STEPS_TOLERANCE
    def _final_collection_status(self) -> str:
        """The verdict written at close(). See `_write_session_provenance` for the third one."""
        if self.planned_steps is None:
            return COLLECTION_STATUS_CLOSED_UNKNOWN_LENGTH
        if self.new_step_count >= self.planned_steps - self._PLANNED_STEPS_TOLERANCE:
            return COLLECTION_STATUS_COMPLETE
        return COLLECTION_STATUS_TRUNCATED

    def collection_provenance(self) -> dict[str, Any]:
        """This session's completion provenance, as it stands right now.

        `session_index` is which session of the file this is (0 for a fresh dataset), so a
        caller can tell "this is the whole file" apart from "this is the latest of several".
        """
        closed = self.h5_file is None
        return {
            "collection_status": self._final_collection_status() if closed else COLLECTION_STATUS_IN_PROGRESS,
            "planned_steps": int(self.planned_steps) if self.planned_steps else -1,
            "planned_steps_source": self.planned_steps_source,
            "collected_steps": int(self.new_step_count),
            "session_index": int(getattr(self, "_session_index", 0)),
        }

    def start_episode(self):
        """
        Start new episode for all environments (batch format [num_envs, ...]).
        """
        # Finalize previous episode if exists
        if self.current_episode is not None:
            self._finalize_episode()
        
        # Reset reset_envs to all False when starting a new episode
        self.reset_envs[:] = False
        
        # Initialize new episode with all observation groups
        episode = {
            "episode_id": self.episode_id_counter,
            "episode_length": 0,
            "actions": [],
            "dones": [],
        }
        
        # Only add rewards if we know rewards are available
        if self.has_rewards is True:
            episode["rewards"] = []
        
        # Add observation groups based on obs_dict
        for obs_group_key in self.obs_dict.keys():
            episode[obs_group_key] = []
        
        self.current_episode = episode
        self.episode_id_counter += 1

    def collect_step(
        self,
        obs_dict: dict[str, torch.Tensor | np.ndarray],
        actions: torch.Tensor | np.ndarray,
        dones: torch.Tensor | np.ndarray,
        rewards: torch.Tensor | np.ndarray | None = None,
    ):
        """
        Collect data for one step for all environments (batch format [num_envs, ...]).
        
        Args:
            obs_dict: Dictionary of observations, shape [num_envs, obs_dim] or [1, obs_dim] for single env,
                     or [num_envs] or [1] (1D, will be converted to 2D).
            actions: Actions taken, shape [num_envs, action_dim] or [1, action_dim] for single env,
                    or [num_envs] or [1] (1D, will be converted to 2D).
            rewards: Rewards (optional), 1D array [num_envs] or scalar for single env (will be converted to 2D).
            dones: Done flags, 1D array [num_envs] or scalar for single env (will be converted to 2D).
        
        Raises:
            ValueError: If obs_dict/actions are not 2D+ arrays, or batch dimension doesn't match num_envs.
        """
        # Helper to convert to numpy and ensure batch dimension
        def to_numpy_batch(data, name: str = "data"):
            """Convert to numpy array and ensure batch dimension [num_envs, ...].
            
            Automatically converts 1D arrays to 2D: [num_envs] -> [num_envs, 1]
            Handles batch dimension and broadcasting: [1, ...] -> [num_envs, ...]
            
            Args:
                data: Input data (torch.Tensor or np.ndarray)
                name: Name of the data for error messages
            
            Returns:
                numpy array with shape [num_envs, ...] (always 2D+)
            
            Raises:
                ValueError: If shape mismatch or invalid dimensions
            """
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            
            # Handle 1D arrays: convert to 2D [num_envs] -> [num_envs, 1]
            if data.ndim == 1:
                if len(data) == self.num_envs:
                    return data[:, np.newaxis]
                elif len(data) == 1 and self.num_envs > 1:
                    # Single env: [1] -> [num_envs, 1] (broadcast)
                    return np.broadcast_to(data[:, np.newaxis], (self.num_envs, 1)).copy()
                else:
                    raise ValueError(
                        f"{name} 1D array length mismatch: got length {len(data)}, "
                        f"expected {self.num_envs} or 1."
                    )
            
            # Require 2D+ arrays
            if data.ndim < 2:
                raise ValueError(
                    f"{name} must be a 1D or 2D+ array (got shape {data.shape}, ndim={data.ndim})."
                )
            
            # Handle single env broadcasting: [1, ...] -> [num_envs, ...]
            if data.shape[0] == 1 and self.num_envs > 1:
                return np.broadcast_to(data, (self.num_envs,) + data.shape[1:]).copy()
            
            # Verify batch dimension matches num_envs
            if data.shape[0] != self.num_envs and data.shape[0] != 1:
                raise ValueError(
                    f"{name} batch dimension mismatch: got shape {data.shape}, "
                    f"expected first dimension to be {self.num_envs} or 1 (for single env)."
                )
            
            return data
        
        # Ensure episode is started
        if self.current_episode is None:
            self.start_episode()
        
        # Track rewards availability (set on first call)
        if self.has_rewards is None:
            self.has_rewards = (rewards is not None)
            if self.h5_file is not None:
                self.h5_file.attrs["has_rewards"] = self.has_rewards
            if self.has_rewards and "rewards" not in self.current_episode:
                self.current_episode["rewards"] = []
        
        # Update reset_envs: set corresponding indices to True
        dones_np = to_numpy_batch(dones, name="dones")
        # Ensure boolean type for bitwise_or operation
        self.reset_envs[:, 0] |= dones_np[:, 0].astype(bool)
        
        # Collect data for all environments in batch format [num_envs, ...]
        step_data = {}
        
        # Extract observations (batch format, automatically converts 1D to 2D)
        if self.obs_dict:
            for obs_group_key in self.obs_dict.keys():
                if obs_group_key in obs_dict:
                    step_data[obs_group_key] = to_numpy_batch(obs_dict[obs_group_key], name=f"obs_dict['{obs_group_key}']")
        
        # Extract actions (batch format, automatically converts 1D to 2D)
        step_data["actions"] = to_numpy_batch(actions, name="actions")
        
        # Extract rewards if available (batch format, automatically converts 1D to 2D)
        if rewards is not None:
            step_data["rewards"] = to_numpy_batch(rewards, name="rewards")
        
        # Record current reset_envs as dones (2D format [num_envs, 1])
        step_data["dones"] = self.reset_envs.copy()

        # Zero-out data for environments that are already marked done/reset
        done_mask = self.reset_envs[:, 0].astype(bool)
        if done_mask.any():
            for key, value in step_data.items():
                if key == "dones":
                    continue
                # value shape: [num_envs, ...]
                value = value.copy()
                value[done_mask] = 0
                step_data[key] = value
        
        # Append to episode (all environments at once)
        for key, value in step_data.items():
            if key not in self.current_episode:
                self.current_episode[key] = []
            self.current_episode[key].append(value)
        
        self.current_episode["episode_length"] += 1
        self.new_step_count += 1

    def _finalize_episode(self):
        """Finalize current episode (all environments in batch format)."""
        if self.current_episode is None:
            return
        
        if self.current_episode["episode_length"] == 0:
            self.current_episode = None
            return
        
        # Convert lists to numpy arrays
        # Each value_list contains arrays of shape [num_envs, ...] for each timestep
        # Stack along time dimension: [episode_length, num_envs, ...]
        finalized_episode = {}
        for key, value_list in self.current_episode.items():
            if key in ["episode_id", "episode_length"]:
                finalized_episode[key] = value_list
            else:
                # Stack along time dimension: [episode_length, num_envs, ...]
                finalized_episode[key] = np.stack(value_list, axis=0)
        
        # Save episode
        self.episodes_buffer.append(finalized_episode)
        
        # Batch save if buffer is full
        if len(self.episodes_buffer) >= self.batch_size:
            self._save_episodes_batch_h5()
        
        logger.debug(f"Episode {finalized_episode['episode_id']} finalized: {finalized_episode['episode_length']} steps (num_envs={self.num_envs})")
        
        # Clear current episode
        self.current_episode = None

    def _save_episodes_batch_h5(self):
        """Save a batch of episodes to HDF5."""
        if not self.episodes_buffer or self.h5_episodes_group is None:
            return
        
        for episode in self.episodes_buffer:
            ep_id = episode["episode_id"]
            ep_group = self.h5_episodes_group.create_group(f"episode_{ep_id}")
            
            for key, value in episode.items():
                if key not in ["episode_id", "episode_length"]:
                    try:
                        compression = "gzip" if self.compress else None
                        chunks = None
                        if self.compress and value.size > 1000000:
                            chunks = True
                        if compression == "gzip" and value.size > 10000000:
                            compression = "lzf"
                        
                        ep_group.create_dataset(key, data=value, compression=compression, chunks=chunks)
                    except Exception as e:
                        logger.error(f"Error saving {key} for episode {ep_id}: {e}")
                        raise
            
            # Save has_rewards flag as attribute for this episode
            ep_group.attrs["has_rewards"] = self.has_rewards if self.has_rewards is not None else False
            
            ep_group.attrs["episode_length"] = episode["episode_length"]
            ep_group.attrs["num_envs"] = self.num_envs
            # Which collection session produced this episode: the 0-based index into the
            # file's `collection_sessions` list.
            ep_group.attrs["collection_session"] = int(getattr(self, "_session_index", 0))
            self._session_episode_ids.append(int(ep_id))

        self.saved_episode_count += len(self.episodes_buffer)
        self.new_episode_count += len(self.episodes_buffer)
        # Keep `collected_steps` current for the reader of a file whose writer is later killed.
        self._write_session_provenance(COLLECTION_STATUS_IN_PROGRESS)
        logger.info(f"Saved batch of {len(self.episodes_buffer)} episodes (total: {self.saved_episode_count})")
        self.episodes_buffer.clear()
        # Force a full GC only for multi-env or batched saves, not the single-episode path.
        if self.num_envs > 1 or self.batch_size > 1:
            gc.collect()

    def flush(self):
        """Flush all remaining episodes to disk."""
        # Finalize current episode
        if self.current_episode is not None:
            self._finalize_episode()
        
        # Save any remaining episodes in buffer
        if self.episodes_buffer:
            self._save_episodes_batch_h5()

    def reset_session(self):
        """Discard the current dataset and start a fresh session on the same path."""
        save_path = self.output_dir / f"{self.dataset_name}.h5"
        if self.h5_file is not None:
            self.h5_file.close()
        self.h5_file = None
        self.h5_episodes_group = None
        if save_path.exists():
            save_path.unlink()

        self.reset_envs = np.zeros((self.num_envs, 1), dtype=bool)
        self.has_rewards = None
        self.current_episode = None
        self.episode_id_counter = 0
        self.episodes_buffer = []
        self.saved_episode_count = 0
        self.new_episode_count = 0
        self.new_step_count = 0
        # The file was unlinked above, so the session history it carried is gone with it.
        # `_init_h5_file` re-derives `_session_index` from the (now empty) new file.
        self._session_episode_ids = []

        self._init_h5_file()
        logger.info(f"Data collector session reset: {save_path}")


    def close(self):
        """Close file handles and finalize."""
        self.flush()

        if self.h5_file is not None:
            final_status = self._final_collection_status()
            self._write_session_provenance(final_status)
            if final_status == COLLECTION_STATUS_TRUNCATED:
                logger.warning(
                    f"[collection] this session recorded {self.new_step_count} of the "
                    f"{self.planned_steps} step(s) it planned ({self.planned_steps_source}); the "
                    "dataset is marked collection_status=truncated. It is still a valid H5 and the "
                    "FADA finetune will train on it -- on a short rollout."
                )
            elif final_status == COLLECTION_STATUS_CLOSED_UNKNOWN_LENGTH:
                logger.info(
                    "[collection] no planned step count was supplied, so this dataset records "
                    "collection_status=closed_unknown_length: it closed cleanly, but whether it is "
                    "as long as it was meant to be is not recorded."
                )
            else:
                logger.info(
                    f"[collection] recorded {self.new_step_count}/{self.planned_steps} planned step(s) "
                    f"({self.planned_steps_source}); collection_status=complete."
                )
            self.h5_file.close()
            self.h5_file = None
            self.h5_episodes_group = None
            
            save_path = self.output_dir / f"{self.dataset_name}.h5"
            if save_path.exists():
                logger.info(f"Dataset saved to {save_path}")
                logger.info(f"Dataset size: {os.path.getsize(save_path) / 1e6:.2f} MB")
                logger.info(f"Total episodes: {self.saved_episode_count}")
