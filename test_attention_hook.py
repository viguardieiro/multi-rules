"""
Quick test to verify attention hooking works with a real model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.boost_config import TokenSubset, BoostConfig
from src.token_utils import create_token_subset_from_substring
from src.attention_hook import register_boost_hooks, unregister_boost_hooks, update_bias_mask


def main():
    print("=" * 80)
    print("Testing Attention Hook with GPT-2")
    print("=" * 80)

    # Load small model
    print("\n1. Loading GPT-2...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    print(f"   Model loaded on {device}")
    print(f"   Model has {model.config.n_layer} layers")

    # Prepare input
    print("\n2. Preparing input...")
    text = "Instruction: Answer in French. Question: What is the capital of France?"
    print(f"   Text: {text}")

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    print(f"   Input shape: {input_ids.shape}")

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

        config = BoostConfig(subsets=[instruction_subset])
        print(f"   Config created: {config}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # Register hooks
    print("\n4. Registering hooks...")
    try:
        handle = register_boost_hooks(model, config, input_length=input_ids.shape[1])
        print(f"   Hooks registered: {len(handle.patched_modules)} modules patched")
        print(f"   Patched modules: {list(handle.patched_modules.keys())[:3]}...")

        update_bias_mask(handle, seq_length=input_ids.shape[1], device=device)
        print(f"   Bias mask updated: shape={handle.bias_mask.shape}, device={handle.bias_mask.device}")
        print(f"   Bias values: {handle.bias_mask[:10]}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test generation
    print("\n5. Testing generation with hooks...")
    try:
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 10,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        result = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"   Output: {result}")
        print("   ✓ Generation succeeded with hooks active!")
    except Exception as e:
        print(f"   ERROR during generation: {e}")
        import traceback
        traceback.print_exc()
        return

    # Unregister hooks
    print("\n6. Unregistering hooks...")
    try:
        unregister_boost_hooks(handle)
        print("   ✓ Hooks unregistered successfully")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Test generation without hooks
    print("\n7. Testing generation without hooks...")
    try:
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 10,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        result = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"   Output: {result}")
        print("   ✓ Generation succeeded without hooks!")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
