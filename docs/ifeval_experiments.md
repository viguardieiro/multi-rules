# IFEval Experiments Guide

This guide is for running IFEval experiments in this repo with:
- baseline
- static InstABoost
- dynamic InstABoost (boundary + selector)

## 1) Setup

From repo root:

```bash
python -m pip install -U -r requirements.txt
```

Make sure your selector model is available in Ollama:

```bash
ollama pull gpt-oss:20b
ollama ps
```

If you use another selector model, replace `gpt-oss:20b` in commands below.

## 2) Scripts You Can Use

- `scripts/run_ifeval_val_bias_sweep.py`
  - validation-only hyperparameter search for `boost_bias` (static + dynamic).
- `scripts/run_ifeval_all_methods.py`
  - runs baseline + static + dynamic in one command and writes a comparison table.
- `scripts/run_ifeval_baseline_static_compare.py`
  - runs baseline + static only.
- `scripts/run_ifeval_dynamic.py`
  - runs dynamic only.
- `scripts/compare_ifeval_methods.py`
  - compares already-generated result JSON files.

## 3) Hyperparameter Search (Validation)

Example:

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 HF_XET_HIGH_PERFORMANCE=0 python scripts/run_ifeval_val_bias_sweep.py \
  --model-name google/gemma-2-2b-it \
  --selector-model gpt-oss:20b \
  --limit 100 \
  --bias-values 0,2,4,6,8,10,12 \
  --max-new-tokens 256 \
  --trust-remote-code \
  --output-json results/ifeval/gemma-2-2b-it/val_sweep.json \
  --output-md results/ifeval/gemma-2-2b-it/val_sweep.md
```

Then freeze the selected bias and use it for final runs.

## 4) Run All Methods (One Command)

Example:

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 HF_XET_HIGH_PERFORMANCE=0 python scripts/run_ifeval_all_methods.py \
  --model-name google/gemma-2-2b-it \
  --selector-model gpt-oss:20b \
  --split test \
  --n-val 100 \
  --n-test 400 \
  --limit 0 \
  --boost-bias 8.0 \
  --max-new-tokens 256 \
  --trust-remote-code \
  --output-dir results/ifeval/gemma-2-2b-it/final_test
```

Outputs:
- `baseline.json`
- `static_instaboost.json`
- `dynamic_instaboost.json`
- `comparison.md`

Logging:
- By default these scripts also write a run log file:
  - directory-output scripts: `<output-dir>/run.log`
  - single-json-output scripts: `<output-json>.log`
- You can override with `--log-file <path>`.

## 5) Running Bigger Models (e.g., OSS-120B)

Yes, this is supported. Use the same scripts and pass the generation model in `--model-name`.

Example pattern:

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 HF_XET_HIGH_PERFORMANCE=0 python scripts/run_ifeval_all_methods.py \
  --model-name <your-oss-120b-hf-id-or-local-path> \
  --selector-model gpt-oss:20b \
  --split test \
  --n-val 100 \
  --n-test 400 \
  --limit 0 \
  --boost-bias <selected_from_val> \
  --max-new-tokens 256 \
  --trust-remote-code \
  --output-dir results/ifeval/oss-120b/final_test
```

Notes:
- Use the exact HF model id or local path available on your machine.
- On large models, prefer running on GPU and adjust `--dtype` (`bfloat16` or `float16` as supported).

## 6) About “Max Samples”

Controls:
- `--limit 0` means no additional cap.
- `--n-val` and `--n-test` define split sizes used by loaders.

If you want the largest possible test set after validation split:
- keep `--n-val` fixed (e.g., `100`)
- set `--n-test` very high (e.g., `100000`)
- use `--split test --limit 0`

This takes all available post-validation samples.

## 7) Quick Smoke Command

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 HF_XET_HIGH_PERFORMANCE=0 python scripts/run_ifeval_all_methods.py \
  --model-name google/gemma-2-2b-it \
  --selector-model gpt-oss:20b \
  --split val \
  --limit 10 \
  --max-new-tokens 32 \
  --trust-remote-code \
  --output-dir results/ifeval/gemma-2-2b-it/smoke10
```
