"""Tests for attention_capture module using GPT-2."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.attention_hook import find_attention_modules
from src.attention_capture import (
    SegmentAttentionMap,
    AttentionCaptureHandle,
    register_capture_hooks,
    unregister_capture_hooks,
    reset_capture_data,
    get_capture_results,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpt2_model():
    # Must use eager attention so softmax is explicit and interceptable
    model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()
    return model


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def simple_segment_map():
    """A segment map with 2 segments covering different token ranges."""
    return SegmentAttentionMap(
        segment_names=["seg_a", "seg_b"],
        segment_token_sets=[[1, 2, 3], [5, 6]],
        segment_is_applicable=[True, False],
    )


# ---------------------------------------------------------------------------
# Tests for find_attention_modules
# ---------------------------------------------------------------------------

class TestFindAttentionModules:
    def test_finds_gpt2_modules(self, gpt2_model):
        modules = find_attention_modules(gpt2_model)
        # GPT-2 has 12 attention layers
        assert len(modules) == 12

    def test_returns_name_module_tuples(self, gpt2_model):
        modules = find_attention_modules(gpt2_model)
        for name, module in modules:
            assert isinstance(name, str)
            assert isinstance(module, torch.nn.Module)
            assert "attn" in name.lower()

    def test_excludes_projection_modules(self, gpt2_model):
        modules = find_attention_modules(gpt2_model)
        for name, _ in modules:
            name_lower = name.lower()
            assert "q_proj" not in name_lower
            assert "k_proj" not in name_lower
            assert "v_proj" not in name_lower
            assert "c_proj" not in name_lower


# ---------------------------------------------------------------------------
# Tests for SegmentAttentionMap validation
# ---------------------------------------------------------------------------

class TestSegmentAttentionMap:
    def test_valid_creation(self, simple_segment_map):
        assert len(simple_segment_map.segment_names) == 2

    def test_mismatched_token_sets_length(self):
        with pytest.raises(ValueError, match="segment_token_sets length"):
            SegmentAttentionMap(
                segment_names=["a", "b"],
                segment_token_sets=[[1, 2]],
                segment_is_applicable=[True, False],
            )

    def test_mismatched_applicability_length(self):
        with pytest.raises(ValueError, match="segment_is_applicable length"):
            SegmentAttentionMap(
                segment_names=["a", "b"],
                segment_token_sets=[[1, 2], [3, 4]],
                segment_is_applicable=[True],
            )


# ---------------------------------------------------------------------------
# Tests for register / unregister
# ---------------------------------------------------------------------------

class TestRegisterUnregister:
    def test_register_installs_hooks(self, gpt2_model, simple_segment_map):
        handle = register_capture_hooks(gpt2_model, simple_segment_map)
        try:
            assert len(handle.patched_modules) == 12
            assert handle.num_layers == 12
        finally:
            unregister_capture_hooks(handle)

    def test_unregister_restores_originals(self, gpt2_model, simple_segment_map):
        # Save original forward references
        modules_before = find_attention_modules(gpt2_model)
        originals = {name: module.forward for name, module in modules_before}

        handle = register_capture_hooks(gpt2_model, simple_segment_map)

        # Forwards should be changed
        for name, module in find_attention_modules(gpt2_model):
            if name in handle.patched_modules:
                assert module.forward is not originals[name]

        unregister_capture_hooks(handle)

        # Forwards should be restored
        for name, module in find_attention_modules(gpt2_model):
            assert module.forward is originals[name]

    def test_register_with_layer_filter(self, gpt2_model, simple_segment_map):
        handle = register_capture_hooks(
            gpt2_model, simple_segment_map, layers=[0, 5, 11]
        )
        try:
            assert len(handle.patched_modules) == 3
        finally:
            unregister_capture_hooks(handle)


# ---------------------------------------------------------------------------
# Tests for capture during forward pass
# ---------------------------------------------------------------------------

class TestCaptureDuringForward:
    def test_single_forward_produces_data(
        self, gpt2_model, gpt2_tokenizer, simple_segment_map
    ):
        handle = register_capture_hooks(gpt2_model, simple_segment_map)
        try:
            inputs = gpt2_tokenizer("Hello world, this is a test", return_tensors="pt")
            with torch.no_grad():
                gpt2_model(**inputs)

            results = get_capture_results(handle)
            assert results["num_steps"] == 1
            assert results["num_segments"] == 2

            # Check that data exists for each layer
            for layer_idx in range(12):
                assert layer_idx in results["data"]
                assert 0 in results["data"][layer_idx]
                seg_attn = results["data"][layer_idx][0]
                assert len(seg_attn) == 2
                # Values should be non-negative (attention weights)
                assert all(v >= 0 for v in seg_attn)
        finally:
            unregister_capture_hooks(handle)

    def test_data_is_python_floats(
        self, gpt2_model, gpt2_tokenizer, simple_segment_map
    ):
        handle = register_capture_hooks(gpt2_model, simple_segment_map)
        try:
            inputs = gpt2_tokenizer("Hello world, this is a test", return_tensors="pt")
            with torch.no_grad():
                gpt2_model(**inputs)

            results = get_capture_results(handle)
            for layer_idx in results["data"]:
                for step in results["data"][layer_idx]:
                    for val in results["data"][layer_idx][step]:
                        assert isinstance(val, float)
        finally:
            unregister_capture_hooks(handle)


# ---------------------------------------------------------------------------
# Tests for capture during generate
# ---------------------------------------------------------------------------

class TestCaptureDuringGenerate:
    def test_multi_step_generation(
        self, gpt2_model, gpt2_tokenizer, simple_segment_map
    ):
        handle = register_capture_hooks(gpt2_model, simple_segment_map)
        try:
            inputs = gpt2_tokenizer("Hello world", return_tensors="pt")
            with torch.no_grad():
                gpt2_model.generate(
                    **inputs, max_new_tokens=5, do_sample=False
                )

            results = get_capture_results(handle)
            # Should have 1 prefill step + 5 generation steps = 6
            # (unless KV cache means some are merged)
            assert results["num_steps"] >= 2  # At least prefill + 1 gen
            assert results["num_segments"] == 2

            # All layers should have the same number of steps
            steps_per_layer = set()
            for layer_idx in results["data"]:
                steps_per_layer.add(len(results["data"][layer_idx]))
            assert len(steps_per_layer) == 1
        finally:
            unregister_capture_hooks(handle)


# ---------------------------------------------------------------------------
# Tests for reset
# ---------------------------------------------------------------------------

class TestResetClearsData:
    def test_reset_empties_data(
        self, gpt2_model, gpt2_tokenizer, simple_segment_map
    ):
        handle = register_capture_hooks(gpt2_model, simple_segment_map)
        try:
            inputs = gpt2_tokenizer("Hello world", return_tensors="pt")
            with torch.no_grad():
                gpt2_model(**inputs)

            results_before = get_capture_results(handle)
            assert results_before["num_steps"] == 1

            reset_capture_data(handle)
            results_after = get_capture_results(handle)
            assert results_after["num_steps"] == 0
            assert results_after["data"] == {}
        finally:
            unregister_capture_hooks(handle)

    def test_hooks_work_after_reset(
        self, gpt2_model, gpt2_tokenizer, simple_segment_map
    ):
        handle = register_capture_hooks(gpt2_model, simple_segment_map)
        try:
            inputs = gpt2_tokenizer("Hello world", return_tensors="pt")

            # First forward
            with torch.no_grad():
                gpt2_model(**inputs)
            assert get_capture_results(handle)["num_steps"] == 1

            # Reset and run again
            reset_capture_data(handle)
            with torch.no_grad():
                gpt2_model(**inputs)
            assert get_capture_results(handle)["num_steps"] == 1
        finally:
            unregister_capture_hooks(handle)


# ---------------------------------------------------------------------------
# Tests for attention sum
# ---------------------------------------------------------------------------

class TestSegmentAttentionSums:
    def test_all_tokens_attention_sums_to_one(self, gpt2_model, gpt2_tokenizer):
        """When segments cover ALL tokens, total attention should sum to ~1.0."""
        text = "Hello world"
        inputs = gpt2_tokenizer(text, return_tensors="pt")
        seq_len = inputs["input_ids"].shape[1]

        # Create a single segment covering all tokens
        seg_map = SegmentAttentionMap(
            segment_names=["all_tokens"],
            segment_token_sets=[list(range(seq_len))],
            segment_is_applicable=[True],
        )

        handle = register_capture_hooks(gpt2_model, seg_map)
        try:
            with torch.no_grad():
                gpt2_model(**inputs)

            results = get_capture_results(handle)
            assert results["num_steps"] == 1, "Expected 1 step from single forward"
            assert len(results["data"]) > 0, "Expected data for at least one layer"
            # The single segment should capture ~1.0 total attention
            # (sum across all key positions should equal 1 per query position)
            for layer_idx in results["data"]:
                attn_val = results["data"][layer_idx][0][0]
                assert 0.9 < attn_val < 1.1, (
                    f"Layer {layer_idx}: expected ~1.0, got {attn_val}"
                )
        finally:
            unregister_capture_hooks(handle)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
