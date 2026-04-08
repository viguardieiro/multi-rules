"""Regression tests for layer filtering in src.attention_hook."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attention_hook import register_boost_hooks, unregister_boost_hooks, update_bias_mask
from src.boost_config import BoostConfig, TokenSubset


class DummyAttn(nn.Module):
    """Minimal attention-like module that calls softmax in forward."""

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return F.softmax(scores, dim=-1)


class DummyModel(nn.Module):
    """Model with two leaf modules named 'attn' so discovery finds both."""

    def __init__(self):
        super().__init__()
        self.block0 = nn.Module()
        self.block0.attn = DummyAttn()
        self.block1 = nn.Module()
        self.block1.attn = DummyAttn()
        # Ensure next(model.parameters()) works in update_bias_mask.
        self.anchor = nn.Parameter(torch.zeros(1))


def test_layer_filter_persists_after_update_bias_mask():
    model = DummyModel()
    # Boost only token index 0 and only layer 1.
    cfg = BoostConfig(
        subsets=[TokenSubset(name="rule", indices=[0], bias=5.0)],
        layers=[1],
    )

    handle = register_boost_hooks(model, cfg, input_length=2)
    try:
        update_bias_mask(handle, seq_length=2)

        scores = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        out0 = model.block0.attn(scores)
        out1 = model.block1.attn(scores)

        # Layer 0 is not selected: should remain uniform softmax.
        assert torch.allclose(out0, torch.tensor([[0.5, 0.5]]), atol=1e-6)

        # Layer 1 is selected: index 0 should dominate due to added bias.
        assert out1[0, 0] > out1[0, 1]
        assert out1[0, 0] > 0.99
    finally:
        unregister_boost_hooks(handle)
