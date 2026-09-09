"""Minimal unit test for ONNX export functionality."""

import tempfile
from pathlib import Path

import onnx
import torch
from torch import nn

from holosoma.agents.modules.module_utils import setup_ppo_actor_module
from holosoma.config_types.algo import LayerConfig, ModuleConfig
from holosoma.utils.inference_helpers import export_policy_as_onnx


class ActorWrapper(nn.Module):
    """Wrapper matching PPO's actor_onnx_wrapper pattern."""

    def __init__(self, actor: nn.Module):
        super().__init__()
        self.actor = actor

    def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        return self.actor.act_inference({"actor_obs": actor_obs})


def test_export_policy_as_onnx():
    """Test ONNX export, load, and dimension verification."""
    OBS_DIM, ACT_DIM = 10, 5

    # Minimal config for PPOActor
    module_config = ModuleConfig(
        type="MLP",
        input_dim=["actor_obs"],
        output_dim=[ACT_DIM],
        layer_config=LayerConfig(
            hidden_dims=[64],
            activation="ReLU",
            dropout_prob=0.0,
        ),
        min_noise_std=None,
        min_mean_noise_std=None,
    )

    # Create PPOActor
    actor = setup_ppo_actor_module(
        obs_dim_dict={"actor_obs": OBS_DIM},
        module_config=module_config,
        num_actions=ACT_DIM,
        init_noise_std=0.1,
        device="cpu",
        history_length={"actor_obs": 1},
    )
    wrapper = ActorWrapper(actor)
    wrapper.eval()

    # Export to a temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = str(Path(tmpdir) / "test_policy.onnx")
        example_obs = torch.zeros(1, OBS_DIM)

        export_policy_as_onnx(
            wrapper=wrapper,
            onnx_file_path=onnx_path,
            example_obs_dict={"actor_obs": example_obs},
        )

        # Load and verify
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)

        # Check input/output dims
        assert len(model.graph.input) == 1
        assert len(model.graph.output) == 1

        input_shape = model.graph.input[0].type.tensor_type.shape
        output_shape = model.graph.output[0].type.tensor_type.shape

        assert input_shape.dim[1].dim_value == OBS_DIM
        assert output_shape.dim[1].dim_value == ACT_DIM


if __name__ == "__main__":
    test_export_policy_as_onnx()
