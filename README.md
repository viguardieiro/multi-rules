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

## Extension: Multi-Subset Boosting

### Key Features

1. **Multiple Token Subsets**: Specify different groups of tokens to boost (e.g., instructions, examples, constraints)
2. **Per-Subset Bias Parameters**: Each subset can have its own bias value B_i
3. **Flexible Combination**: Handle overlapping subsets with configurable combination strategies (sum, max, average)
4. **Model Agnostic**: Works with any HuggingFace transformer model

### Mathematical Formulation

For a given attention head with pre-softmax scores S_ij:

```
S'_ij = S_ij + Σ(B_k · mask_k[j])
```

Where:
- `S_ij`: Original attention score from query i to key j
- `B_k`: Bias parameter for subset k
- `mask_k[j]`: Binary indicator (1 if token j is in subset k, 0 otherwise)
- `S'_ij`: Modified attention score

After applying biases, the standard attention computation continues:
```
A' = Softmax(S')
output = A' · V
```

## Project Structure

```
multi_rules/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── src/
│   ├── __init__.py
│   ├── attention_hook.py             # Core attention modification logic
│   ├── boost_config.py               # Configuration classes for boosting
│   ├── token_utils.py                # Token manipulation and substring finding
│   └── visualization.py              # Visualization and analysis tools
├── notebooks/
│   ├── 01_basic_usage.ipynb         # Basic usage examples
│   ├── 02_single_vs_multi_boost.ipynb  # Compare single vs multi-subset
│   ├── 03_bias_analysis.ipynb       # Analyze effects of different bias values
│   └── 04_gpt_oss_experiments.ipynb # Experiments with openai/gpt-oss-20b
└── tests/
    ├── test_attention_hook.py
    ├── test_token_utils.py
    └── test_boost_config.py
```

## Implementation Plan

### Phase 1: Core Implementation (src/)

#### 1.1 `boost_config.py`
Configuration data structures:
- `TokenSubset` dataclass: Defines a subset of tokens with indices and bias parameter
  - `name`: str (e.g., "instruction")
  - `indices`: List[int] (absolute token positions)
  - `bias`: float (B value to add)
- `BoostConfig` dataclass: Manages multiple token subsets
  - `subsets`: List[TokenSubset]
  - `layers`: Optional[List[int]] (None = all layers)
  - `heads`: Optional[List[int]] (None = all heads)
  - `combination`: str = "sum"
- Validation logic for configuration parameters

#### 1.2 `attention_hook.py`
Core hooking functionality:
- `apply_attention_bias(attn_scores, bias_mask)`: Modify attention scores with bias mask
- `create_bias_mask(token_subsets, seq_length)`: Generate bias mask from token subsets (with summing for overlaps)
- `register_boost_hooks(model, config)`: Register hooks on model's attention layers
  - Returns handle for cleanup
  - Handles different model architectures (GPT-2, GPT-Neo, LLaMA, etc.)
- `unregister_boost_hooks(model, handle)`: Remove hooks from model
- `update_boost_config(handle, new_config)`: Update configuration dynamically during generation

#### 1.3 `token_utils.py`
Token manipulation utilities:
- `find_substring_token_indices(text, substring, tokenizer)`: Find token indices for a substring
  - Validates substring is present in text
  - Returns List[int] of token indices
  - Raises ValueError if substring not found
  - Handles tokenization edge cases (e.g., BPE splits)
- `validate_token_indices(indices, max_length)`: Ensure indices are valid
- `create_token_subset_from_substring(name, text, substring, tokenizer, bias)`: Convenience function
  - Combines substring finding and TokenSubset creation

#### 1.4 `visualization.py`
Analysis and visualization:
- `visualize_attention()`: Visualize attention patterns with/without boosting
- `plot_bias_mask()`: Visualize the bias mask applied to tokens
- `compare_outputs()`: Compare generations with different boost configurations
- `attention_analysis()`: Analyze how attention changes with boosting

### Phase 2: Testing & Validation

#### 2.1 Unit Tests
- Test bias mask generation with various configurations
- Test attention modification correctness
- Test model wrapper functionality
- Edge cases: empty subsets, overlapping indices, out-of-range indices

#### 2.2 Integration Tests
- End-to-end generation with boosting
- Verify attention scores are modified correctly
- Test with different model architectures

### Phase 3: Experiments (notebooks/)

#### 3.1 Basic Usage (`01_basic_usage.ipynb`)
- Load a small model (e.g., GPT-2)
- Define simple instruction + query
- Apply boosting to instruction tokens
- Compare outputs with and without boosting

#### 3.2 Single vs Multi-Subset (`02_single_vs_multi_boost.ipynb`)
- Compare original InstABoost (single subset) vs multi-subset
- Scenarios:
  - Instruction + examples (different biases)
  - Instruction + constraints + context
  - Hierarchical instructions

#### 3.3 Bias Analysis (`03_bias_analysis.ipynb`)
- Sweep bias values for different subsets
- Measure:
  - Instruction following accuracy
  - Output quality (fluency, relevance)
  - Attention distribution changes
- Find optimal bias ranges

#### 3.4 GPT-OSS-20B Experiments (`04_gpt_oss_experiments.ipynb`)
- Load openai/gpt-oss-20b model
- Real-world instruction following tasks
- Compare against baselines:
  - No boosting
  - Original InstABoost
  - Multi-subset InstABoost
- Measure performance on standard benchmarks

### Phase 4: Documentation & Examples

- API documentation
- Usage examples for common scenarios
- Best practices guide
- Performance considerations

## Technical Considerations

### 1. Model Compatibility
- Identify attention layer patterns in HuggingFace models
- Handle different attention implementations:
  - GPT-2/GPT-Neo: `attn.c_attn` or similar
  - LLaMA: `self_attn.q_proj`, `self_attn.k_proj`
  - Generic: Auto-detect attention modules
- Support both encoder-decoder and decoder-only architectures
- Handle different attention score formats (some models use different shapes)

### 2. Substring-to-Index Mapping
- Tokenization challenges:
  - BPE tokenizers may split substrings unexpectedly
  - Need to handle whitespace and special tokens carefully
  - Verify substring tokens match exactly in the tokenized sequence
- Algorithm:
  1. Tokenize full text and substring separately
  2. Find substring token sequence in full token sequence
  3. Return start and end indices
  4. Validate match is exact (raise error if not found)
- Edge cases:
  - Substring appears multiple times (return first occurrence or all?)
  - Substring not found (raise clear error)
  - Empty substring (validation error)

### 3. Efficiency
- Minimize overhead of hook execution
- Efficient bias mask computation:
  - Pre-compute mask once, cache for reuse
  - Only recompute when configuration changes
  - Use sparse representations for memory efficiency
- Handle long sequences (e.g., 8k+ tokens)

### 4. Dynamic Configuration
- Support changing boost config during generation
- Callback mechanism: user provides function `f(step, tokens) -> BoostConfig`
- Efficiently update hooks without full re-registration
- Handle stateful boost strategies (e.g., gradually decrease bias)

### 5. Debugging & Analysis
- Logging of attention modifications
- Visualization tools for attention patterns
- Export attention scores before/after boosting
- Metrics for measuring boosting effectiveness

## Usage Example (Preview)

### Basic Usage with Manual Indices

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.attention_hook import register_boost_hooks, unregister_boost_hooks
from src.boost_config import TokenSubset, BoostConfig

# Load model normally
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b")
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

# Prepare input
text = "Instruction: Answer in French. Question: What is the capital of France?"
input_ids = tokenizer(text, return_tensors="pt").input_ids

# Define token subsets manually (if you know the indices)
subsets = [
    TokenSubset(name="instruction", indices=[0, 1, 2, 3, 4], bias=2.0),
    TokenSubset(name="question", indices=[5, 6, 7, 8, 9, 10], bias=1.0)
]
config = BoostConfig(subsets=subsets)

# Register hooks
handle = register_boost_hooks(model, config)

# Generate with boosting
output = model.generate(input_ids, max_length=50)
decoded = tokenizer.decode(output[0])
print(decoded)

# Clean up
unregister_boost_hooks(model, handle)
```

### Using Substring-to-Indices Helper

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.attention_hook import register_boost_hooks, unregister_boost_hooks
from src.token_utils import create_token_subset_from_substring
from src.boost_config import BoostConfig

# Load model
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b")
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

# Prepare input
text = "Instruction: Answer in French. Question: What is the capital of France?"

# Create token subsets from substrings (automatic index finding)
instruction_subset = create_token_subset_from_substring(
    name="instruction",
    text=text,
    substring="Instruction: Answer in French.",
    tokenizer=tokenizer,
    bias=2.0
)

question_subset = create_token_subset_from_substring(
    name="question",
    text=text,
    substring="Question: What is the capital of France?",
    tokenizer=tokenizer,
    bias=1.0
)

# Create configuration
config = BoostConfig(subsets=[instruction_subset, question_subset])

# Tokenize after creating subsets
input_ids = tokenizer(text, return_tensors="pt").input_ids

# Register and generate
handle = register_boost_hooks(model, config)
output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0]))

unregister_boost_hooks(model, handle)
```

### Dynamic Configuration During Generation

```python
# For changing boost config during generation, we'll provide a callback mechanism
from src.attention_hook import register_dynamic_boost_hooks

def boost_config_callback(step, tokens):
    """Called at each generation step to get boost configuration"""
    if step < 5:  # First 5 tokens: boost instruction heavily
        return BoostConfig(subsets=[TokenSubset("inst", [0, 1, 2, 3], bias=3.0)])
    else:  # After that: lighter boosting
        return BoostConfig(subsets=[TokenSubset("inst", [0, 1, 2, 3], bias=1.0)])

handle = register_dynamic_boost_hooks(model, boost_config_callback)
output = model.generate(input_ids, max_length=50)
unregister_boost_hooks(model, handle)
```

## Roadmap

- [x] Project setup and structure
- [ ] Phase 1: Core implementation
  - [ ] boost_config.py (configuration dataclasses)
  - [ ] attention_hook.py (hook registration and bias application)
  - [ ] token_utils.py (substring-to-indices mapping)
  - [ ] visualization.py (analysis and visualization tools)
- [ ] Phase 2: Testing
  - [ ] Unit tests for each module
  - [ ] Integration tests with real models
  - [ ] Test substring-to-indices edge cases
- [ ] Phase 3: Experiments
  - [ ] Basic usage notebook
  - [ ] Single vs multi-subset comparison
  - [ ] Bias sweep analysis
  - [ ] GPT-OSS-20B experiments
- [ ] Phase 4: Documentation
  - [ ] API documentation
  - [ ] Usage guides
  - [ ] Performance analysis

## Dependencies

- PyTorch >= 2.0
- Transformers >= 4.30.0
- numpy
- matplotlib (for visualization)
- jupyter (for notebooks)
- pytest (for testing)

## References

- Original InstABoost paper: "Instruction Following by Principled Attention Boosting of Large Language Models"
- Implementation inspired by TransformerLens hooks and PASTA method

## Design Decisions

Based on requirements:

1. **Bias Combination Strategy**: Sum all applicable biases when tokens belong to multiple subsets
2. **Token Indexing**: Absolute positions in the input sequence, with utility functions to identify indices from substrings
3. **Layer/Head Specificity**: Support applying to specific layers/heads, but default is all layers/all heads
4. **Dynamic vs Static**: Support changing boost configuration during generation (e.g., different biases per generation step)
