from __future__ import annotations

import types

import pytest

from holosoma.fada.common import compact_obs
from holosoma.utils.safe_torch_import import torch


def _build_dummy_env(device: str = "cpu") -> types.SimpleNamespace:
    return types.SimpleNamespace(
        base_ang_vel=torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32, device=device),
        dof_pos=torch.tensor([[0.1, -0.2]], dtype=torch.float32, device=device),
        dof_vel=torch.tensor([[4.0, -5.0]], dtype=torch.float32, device=device),
        projected_gravity=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=device),
    )


@pytest.fixture
def patch_obs_terms(monkeypatch):
    monkeypatch.setattr(compact_obs.obs_terms, "base_ang_vel", lambda env: env.base_ang_vel)
    monkeypatch.setattr(compact_obs.obs_terms, "dof_pos", lambda env: env.dof_pos)
    monkeypatch.setattr(compact_obs.obs_terms, "dof_vel", lambda env: env.dof_vel)
    monkeypatch.setattr(compact_obs.obs_terms, "projected_gravity", lambda env: env.projected_gravity)


def test_extract_compact_obs_scaled_order_and_dim(patch_obs_terms):
    env = _build_dummy_env()
    out = compact_obs.extract_compact_obs(env, term_scale=None)
    assert out.shape == (1, 10)
    expected = torch.tensor(
        [[0.25, -0.5, 0.75, 0.1, -0.2, 0.2, -0.25, 0.0, 0.0, -1.0]],
        dtype=torch.float32,
    )
    assert torch.allclose(out.cpu(), expected)


def test_extract_compact_obs_is_deterministic(patch_obs_terms):
    # extract_compact_obs takes no noise toggle and is unconditionally deterministic.
    # The separate, always-on augment_obs_noise training-batch augmentation is a
    # different mechanism and is not exercised here.
    env = _build_dummy_env()
    out1 = compact_obs.extract_compact_obs(env, term_scale=None)
    out2 = compact_obs.extract_compact_obs(env, term_scale=None)
    assert torch.allclose(out1, out2)


def test_get_compact_obs_preprocess_keeps_term_noise_metadata():
    # noise_enabled/term_noise_effective are fixed constants (add_noise is always False),
    # but term_noise is a real pass-through value: augment_obs_noise reuses it as its
    # noise-magnitude source.
    preprocess = compact_obs.get_compact_obs_preprocess(
        term_scale=None,
        term_noise=None,
    )
    assert preprocess["term_scale"]["base_ang_vel"] == pytest.approx(0.25)
    assert preprocess["term_scale"]["dof_vel"] == pytest.approx(0.05)
    assert preprocess["term_noise"]["dof_vel"] == pytest.approx(1.0)
    assert preprocess["term_noise_effective"]["dof_vel"] == pytest.approx(0.0)
    assert preprocess["noise_enabled"] is False
