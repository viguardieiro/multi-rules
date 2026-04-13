"""Synthetic integration test for dynamic boosting subset switching."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import register_boost_hooks, unregister_boost_hooks, update_bias_mask
from src.dynamic_boost import BoundaryChecker, BoundaryConfig, DynamicBoostController, TokenStepOutput
from src.dynamic_boost.types import SelectorDecision, SelectorRequest
from src.ifeval_dynamic.runner import build_active_boost_config


class _DummyAttn(nn.Module):
    """Minimal attention-like module that runs softmax over score logits."""

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return F.softmax(scores, dim=-1)


class _DummyModel(nn.Module):
    """Model exposing a leaf `attn` module so hook discovery can patch it."""

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Module()
        self.block.attn = _DummyAttn()
        self.anchor = nn.Parameter(torch.zeros(1))


class _ProgrammedSelector:
    """Selector that emits a fixed decision sequence per call."""

    def __init__(self) -> None:
        self.calls = 0

    def select(self, _request: SelectorRequest) -> SelectorDecision:
        self.calls += 1
        if self.calls == 1:
            return SelectorDecision(
                decision="switch",
                active_instruction_ids=["i2"],
                confidence=1.0,
                reason="step1->switch-to-i2",
            )
        return SelectorDecision(
            decision="add",
            active_instruction_ids=["i3"],
            confidence=1.0,
            reason="step2->add-i3",
        )


def test_dynamic_boost_switches_attention_targets_across_steps():
    model = _DummyModel()
    instruction_token_indices = {
        "i1": [0],
        "i2": [1],
        "i3": [2],
    }
    boost_bias = 6.0
    initial_active = ["i1"]

    initial_cfg = build_active_boost_config(
        instruction_token_indices,
        initial_active,
        boost_bias=boost_bias,
    )

    handle = register_boost_hooks(model, initial_cfg, input_length=3)
    update_bias_mask(handle, seq_length=3, device=torch.device("cpu"))

    selector = _ProgrammedSelector()
    boundary_checker = BoundaryChecker(
        BoundaryConfig(
            min_tokens_between_checks=1,
            max_tokens_without_check=99,
            boundary_markers=(".",),
            rolling_buffer_chars=16,
        )
    )

    attention_per_step: list[torch.Tensor] = []

    def step_fn(_active_ids: list[str], step_index: int) -> TokenStepOutput:
        scores = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        attn = model.block.attn(scores).detach().clone()
        attention_per_step.append(attn)

        if step_index <= 2:
            return TokenStepOutput(text="x.")
        return TokenStepOutput(text="x", is_eos=True)

    def request_builder(
        current_generation: str,
        active_ids: list[str],
        generation_token_count: int,
        step_index: int,
    ) -> SelectorRequest:
        return SelectorRequest(
            sample_id="synthetic_1",
            base_prompt="q",
            candidate_instruction_ids=["i1", "i2", "i3"],
            instruction_text_by_id={"i1": "A", "i2": "B", "i3": "C"},
            currently_active_instruction_ids=active_ids,
            current_generation=current_generation,
            generation_token_count=generation_token_count,
            step_index=step_index,
            metadata={},
        )

    def on_selector_update(next_active_ids: list[str], _decision: SelectorDecision, _event: object) -> None:
        handle.config = build_active_boost_config(
            instruction_token_indices,
            next_active_ids,
            boost_bias=boost_bias,
        )
        update_bias_mask(handle, seq_length=3, device=torch.device("cpu"))

    controller = DynamicBoostController(
        model_name="dummy",
        selector_backend="synthetic",
        selector=selector,
        boundary_checker=boundary_checker,
        step_fn=step_fn,
        request_builder=request_builder,
        on_selector_update=on_selector_update,
        decode_config={"max_new_tokens": 3},
    )

    try:
        result = controller.run(
            sample_id="synthetic_1",
            initial_active_instruction_ids=initial_active,
            max_new_tokens=3,
        )
    finally:
        unregister_boost_hooks(handle)

    assert len(attention_per_step) == 3

    # Step 1 uses initial active set [i1] -> token 0 dominates.
    s1 = attention_per_step[0][0]
    assert s1[0] > s1[1]
    assert s1[0] > s1[2]
    assert float(s1[0]) > 0.99

    # Step 2 occurs after switch to [i2] -> token 1 dominates.
    s2 = attention_per_step[1][0]
    assert s2[1] > s2[0]
    assert s2[1] > s2[2]
    assert float(s2[1]) > 0.99

    # Step 3 occurs after add i3 to active [i2, i3] -> tokens 1 and 2 dominate similarly.
    s3 = attention_per_step[2][0]
    assert s3[1] > s3[0]
    assert s3[2] > s3[0]
    assert abs(float(s3[1] - s3[2])) < 1e-3

    assert result.final_active_instruction_ids == ["i2", "i3"]
    assert result.trace.selector_calls == 2
    assert len(result.trace.boundary_events) == 2
