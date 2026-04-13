# IFEval Dynamic Attention Boosting Plan

## Scope
Build and evaluate three IFEval modes with `gpt-oss:20b`:
1. Baseline (no boost)
2. Static InstABoost (fixed instruction subset)
3. Dynamic InstABoost (boundary-triggered selector decides active instruction subset)

Constraints:
- Do not modify `datasets/ifeval_scripts/*`.
- Do not modify `run_ifeval_benchmark_instr_last.py`.
- Reuse ideas from `run_ifeval_benchmark_instr_last.py`, but implement new code in new modules/scripts.

## Non-negotiables (evaluation validity)
- Use the same decode settings across baseline/static/dynamic unless explicitly ablated:
  - same `max_new_tokens`, temperature, top-p/top-k, stop criteria.
- Tune all hyperparameters on validation split only; run test split once per finalized config.
- Keep split seed fixed and logged.
- Record exact model id and selector model id in each output artifact.
- Keep a versioned schema field in result JSON (for future compatibility).

## Goals
- Make a clean, reusable dynamic-boost framework that can later plug into RuleArena and other benchmarks.
- Keep benchmark-specific logic isolated.
- Add tests for every newly implemented component and require passing tests before moving phases.

## Architecture

### Reusable (benchmark-agnostic) components
Place under `src/dynamic_boost/`.

1. `boundaries.py`
- Token/text boundary detection.
- Debouncing logic.
- Configurable fallback checks.

2. `selector_protocol.py`
- Typed request/response schema for rule/instruction selectors.
- Validation helpers and safe defaults.

3. `selector_llm.py`
- External LLM selector client.
- Prompt assembly + strict JSON parsing + fallback handling.

4. `controller.py`
- Generation loop orchestration.
- Boundary-triggered selector calls.
- Dynamic token-subset updates for boost hooks.
- Telemetry: selector call count, fallback count, boundary hit count.
- Must support incremental decoding with cache (`past_key_values`) so mask updates can happen mid-generation.

5. `types.py`
- Dataclasses for `SelectorRequest`, `SelectorDecision`, `BoundaryConfig`, `DynamicRunTrace`.

### IFEval-specific adapter components
Place under `src/ifeval_dynamic/`.

1. `data_adapter.py`
- IFEval sample loading and normalization from the same sources/format used by current benchmark script.

2. `instruction_spans.py`
- Instruction text to token span mapping for instruction-last prompts.

3. `selector_context.py`
- Build compact selector context from current sample and partial generation.

4. `eval_adapter.py`
- Call official checker (`check_following`) and aggregate IFEval metrics.

### Runner scripts
Place under `scripts/`.

1. `run_ifeval_baseline_static_compare.py`
- Wrapper for baseline + static runs using the same shared decode/eval pipeline as dynamic mode.

2. `run_ifeval_dynamic.py`
- Main dynamic benchmark runner.
- Same split/seed discipline as existing script.
- Writes comparable outputs for baseline/static/dynamic analysis.
- Uses local selector backend by default (Ollama), with optional pluggable backend.

3. `compare_ifeval_methods.py`
- Aggregates outputs and prints table for baseline vs static vs dynamic.

### Selector playground notebook
- `notebooks/ifeval_selector_playground.ipynb`
- Purpose: iterate on external LLM selector prompt/setup, parse robustness, and step-by-step decisions.

## Boundary checker design

### Trigger behavior
- Primary trigger on completed sentence/paragraph boundaries in generated text.
- Supported boundaries: `.`, `?`, `!`, and paragraph break `\n\n`.

### Debouncing and fallback
- `min_tokens_between_checks` (required): suppress repeated checks too close together.
- `max_tokens_without_check` (required): force a selector call if no boundary appears for too long.

Example:
- For text ending in `"end.\n\n"`, only one selector check should fire due to debouncing.

Recommended initial defaults:
- `min_tokens_between_checks = 8`
- `max_tokens_without_check = 32`

Implementation note:
- Boundary detection should be stateful over a short rolling text buffer so split tokenization (e.g., `"\n"` then `"\n"`) is handled correctly.

## Selector design (IFEval)
- Output only JSON with:
  - `decision`: `"stay" | "switch" | "add"`
  - `active_instruction_ids`: list of instruction ids to boost next
  - `confidence`: float in `[0,1]`
  - `reason`: short string
- Keep selector stateless beyond provided context in v1.
- Add deterministic fallback selector if external model fails or returns invalid JSON.
- Add request timeout + retry budget (small, fixed) to cap latency.
- Log selector raw output when parse fails (sanitized) for debugging prompt drift.

## Critical risks and mitigations
1. Mid-generation intervention risk:
- `model.generate()` is not enough for precise boundary-time updates.
- Mitigation: implement explicit incremental decoding loop with cache and hook updates.

2. Fair comparison risk:
- Dynamic method can look better/worse due to changed sampling settings.
- Mitigation: lock decode config and verify identical parameters in saved metadata.

3. Validation leakage risk:
- Prompt tweaks or threshold changes based on test observations will contaminate results.
- Mitigation: finalize selector prompt + boundary params on validation only, then freeze for test.

4. Latency blow-up risk:
- Boundary-triggered external selector can dominate runtime.
- Mitigation: debounce (`min_tokens_between_checks`), forced check cap (`max_tokens_without_check`), timeout/retry limits, and fallback selector.

5. Adapter drift risk:
- New loader/evaluator may diverge from current benchmark behavior.
- Mitigation: add parity tests against reference functions/output conventions from existing script.

## Phased implementation plan (multi-session friendly)

## Phase 0: Setup and interfaces
Deliverables:
- Create module skeletons under `src/dynamic_boost/` and `src/ifeval_dynamic/`.
- Define dataclasses/protocols for boundary and selector APIs.
- Define canonical result schema and metadata contract used by all methods.

Tests required:
- `tests/test_dynamic_types.py`
- `tests/test_selector_protocol.py`
- `tests/test_result_schema.py`

Exit criteria:
- All new tests pass.
- Type/shape validation catches malformed selector outputs.

## Phase 1: Boundary checker
Deliverables:
- Implement boundary detection + debounce + min/max token gating.

Tests required:
- `tests/test_boundary_checker.py`
- Cases:
  - single boundary token triggers once
  - `".\n\n"` triggers once
  - multiple punctuation variants (`"!?..."`) do not create duplicate checks inside cooldown
  - no boundary triggers forced fallback at `max_tokens_without_check`
  - no checks before `min_tokens_between_checks`

Exit criteria:
- Boundary tests pass and cover edge cases.

## Phase 2: External selector + fallback
Deliverables:
- Implement LLM selector client and strict parser.
- Implement deterministic fallback selector.
- Add pluggable backend interface (start with local Ollama backend).

Tests required:
- `tests/test_selector_llm_parser.py`
- `tests/test_selector_fallback.py`
- `tests/test_selector_backend_ollama.py` (mocked HTTP)
- Cases:
  - valid JSON parses
  - extra text around JSON handled
  - invalid JSON falls back
  - out-of-range confidence clamped/rejected per policy
  - timeout/retry behavior respects configured limits

Exit criteria:
- Selector parser and fallback tests pass.

## Phase 3: Dynamic controller
Deliverables:
- Implement generation loop integration:
  - generate incrementally with `past_key_values`
  - call boundary checker
  - call selector when needed
  - update active instruction token subset in boost mask
- Emit run trace telemetry.

Tests required:
- `tests/test_dynamic_controller.py`
- Cases:
  - selector called only on boundary/forced checks
  - active subset changes follow selector decisions
  - fallback path works when selector fails
  - generation can continue after mid-run subset switch
  - cooldown prevents repeated checks for `".\n\n"`-style endings

Exit criteria:
- Controller tests pass with mocked model/selector.

## Phase 4: IFEval adapter integration
Deliverables:
- Build IFEval adapters (data, spans, eval).
- Add `scripts/run_ifeval_dynamic.py`.

Tests required:
- `tests/test_ifeval_instruction_spans.py`
- `tests/test_ifeval_eval_adapter.py`
- `tests/test_ifeval_parity_with_reference.py`
- Optional small smoke integration test with tiny fixture set.

Exit criteria:
- Adapter tests pass.
- Dynamic script runs on a small sample set without errors.

## Phase 5: Benchmark runs and comparison
Deliverables:
- Run three methods on same split:
  - baseline
  - static instaboost
  - dynamic instaboost
- Produce metrics and runtime comparison.

Validation checks:
- Strict accuracy and instruction-level accuracy are computed consistently.
- Runtime overhead and selector call statistics recorded.
- Verify method metadata confirms identical decode configs.
- Verify no test-time tuning changes were introduced after validation freeze.

Artifacts:
- `results/ifeval/<model>/baseline/*`
- `results/ifeval/<model>/static/*`
- `results/ifeval/<model>/dynamic/*`
- comparison report markdown/json.

## Testing strategy and commands

### Unit tests (required each phase)
- Run targeted tests while implementing:
  - `pytest -q tests/test_boundary_checker.py`
  - `pytest -q tests/test_selector_llm_parser.py`
  - etc.

### Full regression pass (before benchmark run)
- `pytest -q tests`

### Benchmark smoke run (required before full run)
- Run each method on a small fixed subset (e.g., 20-30 samples).
- Confirm output schema, telemetry, and evaluation pipeline all succeed.

### Notebook validation
- Validate JSON and code cell syntax for selector notebook:
  - parse notebook JSON
  - AST-parse each code cell

## Output schema (keep stable)
Each method run should save per-sample records with:
- sample id
- prompt/input text
- generation
- per-instruction eval results
- sample strict pass
- method metadata

Dynamic method should additionally save:
- boundary events
- selector decisions per event
- fallback usage
- total selector calls

## Session handoff protocol
For multi-Codex implementation:
- Keep this plan as source of truth.
- At end of each session, append:
  - completed phase steps
  - failing tests (if any)
  - next concrete task
- Do not start next phase until current phase tests are green.

Minimum handoff bundle per session:
- exact command list run
- exact tests run and pass/fail
- produced artifacts/paths
- unresolved blockers

## Definition of done
- All new tests pass.
- No modifications to forbidden files.
- Dynamic runner produces comparable outputs to baseline/static format.
- Comparison report clearly shows baseline vs static vs dynamic on same split.
- Selector playground notebook exists and is usable for prompt iteration with external LLM.
