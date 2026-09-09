"""Tests for weighted_horizon_mse loss utility."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from holosoma.fada.planner_idm.loss_utils import weighted_horizon_mse


def _make(B: int = 4, K: int = 5, D: int = 8, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator()
    rng.manual_seed(seed)
    pred = torch.randn(B, K, D, generator=rng)
    target = torch.randn(B, K, D, generator=rng)
    return pred, target


def test_gamma_1_equals_mse_loss():
    pred, target = _make()
    expected = F.mse_loss(pred, target)
    got = weighted_horizon_mse(pred, target, gamma=1.0)
    assert torch.allclose(got, expected, atol=1e-6), f"gamma=1: {got} vs {expected}"


def test_gamma_1_exact_path():
    # gamma==1.0 takes the fast path; result must still be a scalar
    pred, target = _make(K=1)
    result = weighted_horizon_mse(pred, target, gamma=1.0)
    assert result.shape == ()


def test_gamma_decay_weights():
    gamma = 0.9
    K = 3
    B, D = 2, 4
    pred = torch.zeros(B, K, D)
    # Build target so each horizon step has a known per-step MSE
    # step 0: error=1, step 1: error=2, step 2: error=3
    target = torch.zeros(B, K, D)
    target[:, 0, :] = 1.0
    target[:, 1, :] = 2.0 ** 0.5  # mse = 2
    target[:, 2, :] = 3.0 ** 0.5  # mse = 3

    # manually compute expected
    mse_per_step = torch.tensor([1.0, 2.0, 3.0])
    weights = torch.tensor([1.0, gamma, gamma**2])
    expected = (weights * mse_per_step).sum() / weights.sum()

    got = weighted_horizon_mse(pred, target, gamma=gamma)
    assert torch.allclose(got, expected, atol=1e-5), f"decay: {got} vs {expected}"


def test_gamma_less_than_1_weights_first_step_more():
    """Early horizon steps should contribute more than late steps for gamma<1."""
    K = 4
    pred, _ = _make(K=K)
    # Target that has large error only at last step
    target_late = pred.clone()
    target_late[:, -1, :] += 10.0
    # Target that has large error only at first step
    target_early = pred.clone()
    target_early[:, 0, :] += 10.0

    loss_late = weighted_horizon_mse(pred, target_late, gamma=0.5)
    loss_early = weighted_horizon_mse(pred, target_early, gamma=0.5)
    assert loss_early > loss_late, "gamma<1 should weight step-0 error higher than step-K error"


def test_config_fallback_default():
    """FADAConfig() without explicit gamma should fall back to the dataclass default.

    The default is 0.0 -- "supervise only the first horizon step".
    """
    import dataclasses

    from holosoma.fada.planner_idm.config import FADAConfig

    cfg = FADAConfig()
    assert hasattr(cfg, "idm_action_loss_horizon_gamma"), "field must exist"
    assert cfg.idm_action_loss_horizon_gamma == 0.0, f"default must be 0.0, got {cfg.idm_action_loss_horizon_gamma}"


def test_config_field_survives_asdict_round_trip():
    """asdict + FADAConfig(**d) must preserve the gamma field (checkpoint compat)."""
    import dataclasses

    from holosoma.fada.planner_idm.config import FADAConfig

    cfg = FADAConfig(idm_action_loss_horizon_gamma=0.9)
    d = dataclasses.asdict(cfg)
    assert d["idm_action_loss_horizon_gamma"] == 0.9

    # Simulate loading a *old* checkpoint dict that lacks the key — must not crash
    old_dict = {k: v for k, v in d.items() if k != "idm_action_loss_horizon_gamma"}
    cfg2 = FADAConfig(**old_dict)
    assert cfg2.idm_action_loss_horizon_gamma == 0.0


def test_gamma_zero_only_first_step():
    """gamma=0 must supervise only the first horizon step (0^0==1 in PyTorch)."""
    B, K, D = 4, 5, 8
    pred = torch.zeros(B, K, D)
    target = torch.zeros(B, K, D)
    target[:, 0, :] = 1.0    # step-0 error = 1
    target[:, 1:, :] = 10.0  # later steps have huge error (should be ignored)
    loss = weighted_horizon_mse(pred, target, gamma=0.0)
    expected_mse_step0 = torch.tensor(1.0)
    assert torch.allclose(loss, expected_mse_step0, atol=1e-6), f"gamma=0: {loss} vs {expected_mse_step0}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_result_on_cuda():
    pred, target = _make()
    pred, target = pred.cuda(), target.cuda()
    result = weighted_horizon_mse(pred, target, gamma=0.8)
    assert result.device.type == "cuda"
    assert result.shape == ()
