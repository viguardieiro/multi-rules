"""
Quick sanity check to test basic functionality.

This script tests:
1. Loading a model
2. Creating boost configuration
3. Registering hooks
4. Generating text
5. Unregistering hooks
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.boost_config import TokenSubset, BoostConfig
from src.token_utils import create_token_subset_from_substring
from src.attention_hook import register_boost_hooks, unregister_boost_hooks, update_bias_mask


def main():
    print("=" * 80)
    print("Multi-Subset InstABoost - Basic Functionality Test")
    print("=" * 80)

    # Load model and tokenizer
    print("\n1. Loading model (GPT-2)...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Move to device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    print(f"   Model loaded on {device}")

    # Prepare input
    print("\n2. Preparing input...")
    text = "Instruction: Answer in French. Question: What is the capital of France?"
    print(f"   Text: {text}")

    # Tokenize
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    print(f"   Tokens: {input_ids.shape[1]}")

    # Create boost configuration
    print("\n3. Creating boost configuration...")
    try:
        instruction_subset = create_token_subset_from_substring(
            name="instruction",
            text=text,
            substring="Instruction: Answer in French.",
            tokenizer=tokenizer,
            bias=2.0
        )
        print(f"   Instruction subset: indices={instruction_subset.indices}, bias={instruction_subset.bias}")

        question_subset = create_token_subset_from_substring(
            name="question",
            text=text,
            substring="Question: What is the capital of France?",
            tokenizer=tokenizer,
            bias=1.0
        )
        print(f"   Question subset: indices={question_subset.indices}, bias={question_subset.bias}")

        config = BoostConfig(subsets=[instruction_subset, question_subset])
        print(f"   Config: {config}")
    except Exception as e:
        print(f"   ERROR creating config: {e}")
        return

    # Register hooks
    print("\n4. Registering hooks...")
    try:
        handle = register_boost_hooks(model, config, input_length=input_ids.shape[1])
        update_bias_mask(handle, seq_length=input_ids.shape[1], device=device)
        print(f"   Hooks registered successfully")
        print(f"   Bias mask shape: {handle.bias_mask.shape if handle.bias_mask is not None else 'None'}")
    except Exception as e:
        print(f"   ERROR registering hooks: {e}")
        import traceback
        traceback.print_exc()
        return

    # Generate without boosting first
    print("\n5. Generating (baseline, no boosting)...")
    try:
        # Unregister hooks temporarily
        unregister_boost_hooks(handle)

        with torch.no_grad():
            output_baseline = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 20,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        baseline_text = tokenizer.decode(output_baseline[0], skip_special_tokens=True)
        print(f"   Baseline: {baseline_text}")
    except Exception as e:
        print(f"   ERROR generating baseline: {e}")
        import traceback
        traceback.print_exc()

    # Register hooks again
    print("\n6. Re-registering hooks for boosted generation...")
    try:
        handle = register_boost_hooks(model, config, input_length=input_ids.shape[1])
        update_bias_mask(handle, seq_length=input_ids.shape[1], device=device)
    except Exception as e:
        print(f"   ERROR re-registering hooks: {e}")
        return

    # Generate with boosting
    print("\n7. Generating (with boosting)...")
    try:
        with torch.no_grad():
            output_boosted = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 20,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        boosted_text = tokenizer.decode(output_boosted[0], skip_special_tokens=True)
        print(f"   Boosted: {boosted_text}")
    except Exception as e:
        print(f"   ERROR generating with boosting: {e}")
        import traceback
        traceback.print_exc()

    # Unregister hooks
    print("\n8. Unregistering hooks...")
    try:
        unregister_boost_hooks(handle)
        print("   Hooks unregistered successfully")
    except Exception as e:
        print(f"   ERROR unregistering hooks: {e}")

    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
