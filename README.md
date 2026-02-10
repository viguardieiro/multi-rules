# Multi-Subset InstABoost

An extended implementation of the InstABoost method for instruction following in Large Language Models, supporting multiple token subsets with different bias parameters.

## Overview

InstABoost (Instruction Attention Boosting) is an inference-time intervention method that improves instruction following by applying a constant additive bias to attention logits corresponding to instruction tokens. This implementation extends the original method to support boosting different parts of the input with different bias parameters.

### How It Works

The method modifies the pre-softmax attention scores in transformer models:
- **Original attention**: `A = Softmax(S)` where `S = QK^T / √d_k`
- **InstABoost**: `A' = Softmax(S + B·mask)` where B is added to selected token positions

In the original paper, a single bias B is applied to all instruction tokens. Our extension allows:
- Multiple token subsets to be specified
- Different bias values for each subset
- Flexible combination of biases when tokens belong to multiple subsets

### Key Features

1. **Multiple Token Subsets**: Specify different groups of tokens to boost (e.g., instructions, examples, constraints)
2. **Per-Subset Bias Parameters**: Each subset can have its own bias value B_i
3. **Generation-Time Boosting**: Bias is applied during both prefill AND autoregressive decoding
4. **Reasoning Model Support**: Works with reasoning models like `openai/gpt-oss-20b`
5. **Model Agnostic**: Works with any HuggingFace transformer model

## Project Structure

```
multi-rules/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── src/
│   ├── __init__.py
│   ├── attention_hook.py             # Core attention modification logic
│   ├── boost_config.py               # Configuration classes for boosting
│   └── token_utils.py                # Token manipulation and substring finding
├── notebooks/
│   ├── 01_basic_usage.ipynb          # Basic usage with GPT-2
│   └── 02_gpt_oss_20b.ipynb          # Experiments with reasoning model
└── tests/
    ├── __init__.py
    ├── test_boost_config.py          # Configuration validation tests
    └── test_token_utils.py           # Token utility tests
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.boost_config import BoostConfig
from src.token_utils import create_token_subset_from_substring
from src.attention_hook import register_boost_hooks, unregister_boost_hooks, update_bias_mask

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Prepare input
text = "Instruction: Answer in French. Question: What is the capital of France?"
input_ids = tokenizer(text, return_tensors="pt").input_ids

# Create token subset from substring (automatic index finding)
instruction_subset = create_token_subset_from_substring(
    name="instruction",
    text=text,
    substring="Instruction: Answer in French.",
    tokenizer=tokenizer,
    bias=2.0
)

# Create configuration and register hooks
config = BoostConfig(subsets=[instruction_subset])
handle = register_boost_hooks(model, config, input_length=input_ids.shape[1])
update_bias_mask(handle, seq_length=input_ids.shape[1])

# Generate with boosting
output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0]))

# Clean up
unregister_boost_hooks(handle)
```

### Usage with Reasoning Models (GPT-OSS-20B)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.boost_config import BoostConfig
from src.token_utils import create_token_subset_from_substring
from src.attention_hook import register_boost_hooks, unregister_boost_hooks, update_bias_mask

# Load reasoning model
model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)

# Use chat template for reasoning models
messages = [{"role": "user", "content": "Respond only in Spanish: What is 2 + 2?"}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to("cuda")

# Boost the instruction (use bias=2.0 for reasoning models)
subset = create_token_subset_from_substring(
    "instruction", formatted_prompt, "Respond only in Spanish", tokenizer, bias=2.0
)
config = BoostConfig(subsets=[subset])

handle = register_boost_hooks(model, config, input_length=input_ids.shape[1])
update_bias_mask(handle, seq_length=input_ids.shape[1], device=input_ids.device)

output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0][input_ids.shape[1]:]))

unregister_boost_hooks(handle)
```

## Optimal Bias Values

Based on experiments with `openai/gpt-oss-20b`:

| Bias Value | Effect |
|------------|--------|
| **0.5 - 2.0** | Optimal range. Model focuses better on instructions |
| **2.0** | Sweet spot for reasoning models |
| **3.0 - 5.0** | Risk of repetition in reasoning phase |
| **10.0+** | Severe degradation, nonsensical output |

**Recommendation**: Start with `bias=2.0` and adjust based on results.

## Key Findings with Reasoning Models

1. **Reasoning Focus**: Boosted model's internal reasoning (`analysis` channel) explicitly mentions the instruction earlier and more prominently
   - Baseline: *"The user asks:..."*
   - Boosted: *"We need to answer in exactly 3 words..."*

2. **Task Completion**: Boosted model more likely to reach final answer where baseline gets stuck in reasoning

3. **Generation-Time Application**: Bias is applied during both prefill AND each decoding step, ensuring consistent instruction attention throughout generation

## API Reference

### `BoostConfig`

Configuration for multi-subset boosting.

```python
from src.boost_config import TokenSubset, BoostConfig

# Define token subsets
subset1 = TokenSubset(name="instruction", indices=[0, 1, 2, 3], bias=2.0)
subset2 = TokenSubset(name="examples", indices=[10, 11, 12], bias=1.0)

# Create config
config = BoostConfig(
    subsets=[subset1, subset2],
    layers=None,      # None = all layers (or list of layer indices)
    heads=None,       # None = all heads (or list of head indices)
    combination="sum" # How to combine overlapping biases
)
```

### `create_token_subset_from_substring`

Automatically find token indices for a substring.

```python
from src.token_utils import create_token_subset_from_substring

subset = create_token_subset_from_substring(
    name="instruction",
    text="Full prompt text here",
    substring="substring to boost",
    tokenizer=tokenizer,
    bias=2.0
)
```

### Hook Management

```python
from src.attention_hook import register_boost_hooks, unregister_boost_hooks, update_bias_mask

# Register hooks
handle = register_boost_hooks(model, config, input_length=seq_len)

# Update bias mask (required before generation)
update_bias_mask(handle, seq_length=seq_len, device=device)

# ... generate ...

# Clean up
unregister_boost_hooks(handle)
```

## Technical Details

### How Bias is Applied During Generation

The implementation handles both prefill and autoregressive decoding:

1. **Prefill**: Attention shape `[batch, heads, seq_len, seq_len]` - bias applied directly
2. **Decoding**: Attention shape `[batch, heads, 1, seq_len+t]` - bias padded with zeros for new tokens

```
Original input: [tok0, tok1, tok2]  with bias [2.0, 2.0, 0.0]
After generating 2 tokens: [tok0, tok1, tok2, gen0, gen1]
Padded bias: [2.0, 2.0, 0.0, 0.0, 0.0]
```

This ensures instruction tokens continue receiving boosted attention throughout generation.

### Supported Model Architectures

The implementation auto-detects attention modules by pattern matching:
- GPT-2/GPT-Neo: `attn` modules
- LLaMA/Mistral: `self_attn` modules
- Generic: Any module with "attn" or "attention" in the name

Excluded patterns (projection layers): `q_proj`, `k_proj`, `v_proj`, `out_proj`, `o_proj`, `c_proj`

## Notebooks

1. **`01_basic_usage.ipynb`**: Introduction with GPT-2, demonstrates basic boosting workflow
2. **`02_gpt_oss_20b.ipynb`**: Experiments with OpenAI's reasoning model, includes bias value analysis

## Dependencies

- PyTorch >= 2.0
- Transformers >= 4.30.0
- NumPy >= 1.24.0
- Jupyter (for notebooks)

## Roadmap

- [x] Core implementation
  - [x] `boost_config.py` - Configuration dataclasses with validation
  - [x] `attention_hook.py` - Hook registration and bias application
  - [x] `token_utils.py` - Substring-to-indices mapping
- [x] Generation-time bias application (not just prefill)
- [x] Reasoning model support (`openai/gpt-oss-20b`)
- [x] Basic usage notebook
- [x] Reasoning model experiments notebook
- [ ] Visualization tools for attention patterns
- [ ] Benchmark suite for instruction following
- [ ] Support for dynamic bias (changing during generation)

## References

- Original InstABoost paper: "Instruction Following by Principled Attention Boosting of Large Language Models"
- Implementation inspired by TransformerLens hooks and PASTA method
