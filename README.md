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
├── src/                               # Core implementation
│   ├── attention_hook.py              # Hook registration and bias application
│   ├── boost_config.py                # Configuration dataclasses
│   ├── token_utils.py                 # Substring-to-token-indices mapping
│   └── rulearena/                     # RuleArena rule filtering pipeline
│       ├── rulebook_segments.py       # Rulebook segmentation (fine/coarse)
│       └── rule_applicability.py      # Per-problem rule selection
├── scripts/
│   ├── eval_rulearena.py              # RuleArena evaluation script
│   ├── sweep_boost_bias.py            # Bias value sweep experiments
│   ├── test_baseline_instructions.py  # Baseline instruction-following tests
│   ├── validate_rule_applicability.py # Validate rule filtering on all samples
│   └── rulearena/
│       └── save_rulebook_segments.py  # Pre-compute and save rulebook segments
├── datasets/
│   └── RuleArena/                     # Cloned benchmark repo (do not modify)
├── results/                           # Experiment outputs
├── notebooks/
│   ├── 01_basic_usage.ipynb           # Intro with GPT-2
│   ├── 02_gpt_oss_20b.ipynb          # Reasoning model experiments
│   ├── boosting_comparison.ipynb      # Boosting results comparison (20b)
│   ├── boosting_comparison_120b.ipynb # Boosting results comparison (120b)
│   └── rulebook_filtering_comparison_120b.ipynb  # Full vs filtered rulebook comparison
└── tests/
    ├── test_boost_config.py
    ├── test_token_utils.py
    ├── test_eval_rulearena.py
    ├── test_rulebook_segments.py      # Tests for rulebook segmentation
    └── test_rule_applicability.py     # Tests for rule filtering logic
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
3. **`boosting_comparison.ipynb`**: Compare boosted vs baseline results for gpt-oss-20b
4. **`boosting_comparison_120b.ipynb`**: Compare boosted vs baseline results for gpt-oss-120b
5. **`rulebook_filtering_comparison_120b.ipynb`**: Compare full rulebook vs applicable-rules-only filtering on 100 airline samples, includes per-sample inspector and LLM judge prompt generation

## Dependencies

- PyTorch >= 2.0
- Transformers >= 4.30.0
- NumPy >= 1.24.0
- Jupyter (for notebooks)

## RuleArena Evaluation

[RuleArena](https://github.com/RuleArena/RuleArena) is a benchmark for testing LLM rule-following across three domains: **airline** (baggage fee calculation), **NBA** (salary cap compliance), and **tax** (income tax computation). Each problem gives the model a long rulebook and asks it to apply the rules to a specific scenario.

### Running an evaluation

```bash
# Standard evaluation with full rulebook
python scripts/eval_rulearena.py \
    --model openai/gpt-oss-20b \
    --domain airline \
    --complexity 0 \
    --boost_strategy none        # or "uniform_rules" with --bias 2.0

# Evaluation with rule filtering (applicable rules only)
python scripts/eval_rulearena.py \
    --model openai/gpt-oss-120b \
    --domain airline \
    --complexity 0 \
    --rules_strategy applicable_only \
    --drop_fee_summaries \
    --use_example \
    --max_problems 20
```

Key flags:

| Flag | Description |
|------|-------------|
| `--max_problems N` | Limit number of samples to evaluate |
| `--start_idx N` | Start from sample index N (for resuming partial runs) |
| `--use_example` | Include a few-shot example in the prompt |
| `--textual` | Use text-only rule format (airline/tax) |
| `--rules_strategy` | `full` (default) or `applicable_only` — filter rulebook to relevant rules per problem |
| `--drop_fee_summaries` | With `applicable_only`: drop generic fee summary sentences that can conflict with cabin-specific fee tables |
| `--boost_strategy` | `none` (default) or `uniform_rules` — apply attention boosting |
| `--bias` | Bias value when using `uniform_rules` boost strategy |

### Results format

Each run saves to `results/rulearena/<model>/<domain>/comp_<N>/<run_id>/`:

- **`<idx>.json`** — one file per sample with predicted/ground-truth answers, model output, timing, token counts, and generation completion status. Written immediately after each sample completes.
- **`summary.json`** — aggregate metrics (accuracy, generation-finished ratio, avg output length, wall time)
- **`config.json`** — run configuration

### Rule Filtering Pipeline

The `--rules_strategy applicable_only` option activates a rule filtering pipeline that reduces the rulebook to only the rules relevant to each problem. This is implemented in `src/rulearena/`:

1. **Rulebook segmentation** (`rulebook_segments.py`): Splits the full rulebook into fine-grained segments (individual table rows, prose paragraphs) and coarse segments (full sections). Segments are always exact substrings of the original rulebook.

2. **Rule applicability** (`rule_applicability.py`): Given a problem (passenger info, items, route), determines which fine segments are needed. Structural elements (headers, table column rows) are automatically included when any row from that table is selected.

3. **Filtered rulebook assembly** (`build_filtered_rulebook()`): Reconstructs a coherent rulebook from selected segments, preserving the original text ordering and filling structural gaps.

Pre-compute segments for a domain:
```bash
python scripts/rulearena/save_rulebook_segments.py --domain airline
```

Validate rule filtering covers ground-truth answers for all samples:
```bash
python scripts/validate_rule_applicability.py --domain airline
```

### Setup

Clone the RuleArena benchmark into `datasets/`:

```bash
git clone https://github.com/RuleArena/RuleArena.git datasets/RuleArena
```

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
