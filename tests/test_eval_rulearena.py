"""Tests for scripts/eval_rulearena.py — RuleArena evaluation pipeline."""

import json
import sys
import pytest
from pathlib import Path

# Ensure project root is on sys.path so we can import the script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from scripts.eval_rulearena import (
    extract_answer_airline,
    extract_answer_tax,
    eval_accuracy_nba,
    build_prompt_airline,
    build_prompt_nba,
    build_query_prompt_nba,
    build_prompt_tax,
    build_boost_config,
    generate_run_id,
    build_results_dir,
    format_boost_metadata,
    save_run_config,
    save_sample_json,
    save_summary_json,
    extract_sample_input,
    load_problems,
    parse_gpt_oss_output,
    AIRLINE_PROMPT_TEMPLATE,
    AIRLINE_EXAMPLE,
    NBA_PROMPT_TEMPLATE,
)


# ===================================================================
# Answer Extraction — Airline
# ===================================================================

class TestAnswerExtractionAirline:
    def test_extract_simple_amount(self):
        response = "The total cost is $1,198."
        assert extract_answer_airline(response) == 1198

    def test_extract_no_comma(self):
        response = "The total cost is $500."
        assert extract_answer_airline(response) == 500

    def test_extract_no_match(self):
        response = "I cannot compute."
        assert extract_answer_airline(response) is None

    def test_extract_with_markdown_bold(self):
        response = "**The total cost is $750.**"
        assert extract_answer_airline(response) == 750

    def test_extract_trailing_text(self):
        response = "The total cost is $300. Thank you!"
        assert extract_answer_airline(response) == 300


# ===================================================================
# Answer Extraction — NBA (via eval_accuracy_nba)
# ===================================================================

class TestAnswerExtractionNBA:
    def test_correct_true_answer(self):
        problem = {
            "answer": True,
            "illegal_operation": "A",
            "problematic_team": "B",
        }
        response = "... Answer: True. Illegal Operation: A. Problematic Team: B. Done."
        is_correct, _, _ = eval_accuracy_nba(response, problem)
        assert is_correct is True

    def test_correct_false_answer(self):
        problem = {"answer": False, "illegal_operation": None, "problematic_team": None}
        response = "... Answer: False. Everything is fine."
        is_correct, _, _ = eval_accuracy_nba(response, problem)
        assert is_correct is True

    def test_wrong_answer(self):
        problem = {
            "answer": True,
            "illegal_operation": "A",
            "problematic_team": "A",
        }
        response = "Answer: False."
        is_correct, _, _ = eval_accuracy_nba(response, problem)
        assert is_correct is False


# ===================================================================
# Answer Extraction — Tax
# ===================================================================

class TestAnswerExtractionTax:
    def test_extract_owed(self):
        response = "The total tax owed is $5,432.10."
        assert extract_answer_tax(response) == pytest.approx(5432.10)

    def test_extract_overpaid(self):
        response = "The total tax overpaid is $1,200.50."
        assert extract_answer_tax(response) == pytest.approx(-1200.50)

    def test_extract_no_match(self):
        response = "I don't know the tax."
        assert extract_answer_tax(response) is None

    def test_extract_integer(self):
        response = "The total tax owed is $3000."
        assert extract_answer_tax(response) == pytest.approx(3000.0)

    def test_extract_with_bold(self):
        response = "**The total tax owed is $1,500.00.**"
        assert extract_answer_tax(response) == pytest.approx(1500.0)


# ===================================================================
# Prompt Construction — Airline
# ===================================================================

class TestPromptConstructionAirline:
    @pytest.fixture
    def sample_rules(self):
        return "Rule 1: Carry-on max 22x14x9\nRule 2: First bag free"

    @pytest.fixture
    def sample_problem(self):
        return {"prompt": "Alice is flying from NYC to LAX with 3 bags."}

    def test_basic_prompt_no_example(self, sample_problem, sample_rules):
        prompt, boostable, question = build_prompt_airline(sample_problem, sample_rules, use_example=False)
        assert "Rule 1:" in prompt
        assert "Alice is flying" in prompt
        assert "<example>" not in prompt

    def test_prompt_with_example(self, sample_problem, sample_rules):
        prompt, _, _ = build_prompt_airline(sample_problem, sample_rules, use_example=True)
        assert "<example>" in prompt

    def test_all_placeholders_replaced(self, sample_problem, sample_rules):
        prompt, _, _ = build_prompt_airline(sample_problem, sample_rules, use_example=True)
        assert "$reference_rules" not in prompt
        assert "$question_prompt" not in prompt
        assert "$example_prompt" not in prompt

    def test_boostable_text_is_reference_rules(self, sample_problem, sample_rules):
        _, boostable, _ = build_prompt_airline(sample_problem, sample_rules, use_example=False)
        assert boostable == sample_rules

    def test_question_text_is_problem_prompt(self, sample_problem, sample_rules):
        _, _, question = build_prompt_airline(sample_problem, sample_rules, use_example=False)
        assert question == sample_problem["prompt"]


# ===================================================================
# Prompt Construction — NBA
# ===================================================================

class TestPromptConstructionNBA:
    @pytest.fixture
    def sample_query(self):
        return {
            "team_situations": ["Team A has a salary of $100M."],
            "player_situations": ["Player A signed a 3-year deal."],
            "operations": ["A. Team A signs Player A."],
        }

    def test_build_query_prompt(self, sample_query):
        qp = build_query_prompt_nba(sample_query)
        assert "Team Situations:" in qp
        assert "Player Situations:" in qp
        assert "Operations:" in qp
        assert "Team A has a salary" in qp

    def test_basic_prompt_no_example(self, sample_query):
        rules = "CBA Rule: max salary is 35% of cap."
        prompt, boostable, question = build_prompt_nba(sample_query, rules, use_example=False)
        assert "CBA Rule:" in prompt
        assert "Team A signs Player A" in prompt
        assert "<example>" not in prompt

    def test_all_placeholders_replaced(self, sample_query):
        rules = "Some rules here."
        prompt, _, _ = build_prompt_nba(sample_query, rules, use_example=True)
        assert "$reference_rules" not in prompt
        assert "$question" not in prompt
        assert "$example" not in prompt

    def test_question_text_is_query_prompt(self, sample_query):
        rules = "Some rules."
        _, _, question = build_prompt_nba(sample_query, rules, use_example=False)
        assert "Team Situations:" in question
        assert "Player Situations:" in question
        assert "Operations:" in question


# ===================================================================
# Prompt Construction — Tax
# ===================================================================

class TestPromptConstructionTax:
    def test_basic_prompt_fills_forms(self):
        """Load a real comp_0 problem and verify forms are filled."""
        problems = load_problems("tax", 0)
        problem = problems[0]
        prompt, boostable, question = build_prompt_tax(problem, use_example=False)
        # TBD fields should be replaced with [__]
        assert "[__]" in prompt
        assert "$TBD" not in prompt
        # Should contain form structure
        assert "Form 1040" in prompt

    def test_does_not_mutate_input(self):
        """Input problem dict should be unchanged after prompt construction."""
        problems = load_problems("tax", 0)
        problem = problems[0]
        original_data_keys = set(problem["dict"]["data"].keys())
        build_prompt_tax(problem, use_example=False)
        assert set(problem["dict"]["data"].keys()) == original_data_keys

    def test_question_text_is_none(self):
        """Tax domain should return None for question_text."""
        problems = load_problems("tax", 0)
        problem = problems[0]
        _, _, question = build_prompt_tax(problem, use_example=False)
        assert question is None


# ===================================================================
# Boost Config
# ===================================================================

class TestBuildBoostConfig:
    @pytest.fixture
    def gpt2_tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("gpt2")

    def test_none_strategy_returns_none(self, gpt2_tokenizer):
        result = build_boost_config("none", "some text", gpt2_tokenizer, "rules", None, 2.0)
        assert result is None

    def test_zero_bias_returns_none(self, gpt2_tokenizer):
        result = build_boost_config("uniform_rules", "some text", gpt2_tokenizer, "rules", None, 0.0)
        assert result is None

    def test_uniform_rules_creates_config(self, gpt2_tokenizer):
        text = "Here are the rules: Follow them carefully. The end."
        rules = "Follow them carefully."
        config = build_boost_config("uniform_rules", text, gpt2_tokenizer, rules, None, 2.0)
        assert config is not None
        assert len(config.subsets) == 1
        assert config.subsets[0].name == "rules"
        assert config.subsets[0].bias == 2.0
        assert len(config.subsets[0].indices) > 0

    def test_unknown_strategy_raises(self, gpt2_tokenizer):
        with pytest.raises(ValueError, match="Unknown boost strategy"):
            build_boost_config("fancy_boost", "text", gpt2_tokenizer, "rules", None, 1.0)

    def test_uniform_question_creates_config(self, gpt2_tokenizer):
        text = "Rules here. Now the question: What is the cost? The end."
        question = "What is the cost?"
        config = build_boost_config("uniform_question", text, gpt2_tokenizer, None, question, 3.0)
        assert config is not None
        assert len(config.subsets) == 1
        assert config.subsets[0].name == "question"
        assert config.subsets[0].bias == 3.0
        assert len(config.subsets[0].indices) > 0

    def test_uniform_question_none_raises(self, gpt2_tokenizer):
        with pytest.raises(ValueError, match="uniform_question strategy requires question_text"):
            build_boost_config("uniform_question", "text", gpt2_tokenizer, "rules", None, 2.0)


# ===================================================================
# Run ID Generation
# ===================================================================

class TestRunIdGeneration:
    def test_none_strategy(self):
        assert generate_run_id("none", 2.0) == "none"

    def test_zero_bias(self):
        assert generate_run_id("uniform_rules", 0.0) == "none"

    def test_uniform_rules(self):
        assert generate_run_id("uniform_rules", 2.0) == "uniform_rules_bias2.0"


# ===================================================================
# Results Dir
# ===================================================================

class TestResultsDir:
    def test_basic_path(self):
        p = build_results_dir("results", "gpt2", "airline", 0, "none")
        assert p == Path("results/rulearena/gpt2/airline/comp_0/none")

    def test_model_with_slash(self):
        p = build_results_dir("results", "openai/gpt-oss-20b", "nba", 1, "uniform_bias2.0")
        assert p == Path("results/rulearena/openai_gpt-oss-20b/nba/comp_1/uniform_bias2.0")


# ===================================================================
# Boost Metadata Formatting
# ===================================================================

class TestBoostMetadata:
    def test_baseline(self):
        result = format_boost_metadata(None, 100)
        assert result == "No boosting (baseline)"

    def test_single_subset(self):
        from src.boost_config import TokenSubset, BoostConfig
        subset = TokenSubset(name="rules", indices=[10, 11, 12, 13], bias=2.5)
        config = BoostConfig(subsets=[subset])
        result = format_boost_metadata(config, 50)
        assert "rules" in result
        assert "2.5" in result
        assert "4 tokens" in result


# ===================================================================
# Config Serialization
# ===================================================================

class TestConfigSerialization:
    def test_save_and_load_roundtrip(self, tmp_path):
        config = {"model": "gpt2", "domain": "airline", "complexity": 0}
        save_run_config(tmp_path, config)
        loaded = json.loads((tmp_path / "config.json").read_text())
        assert loaded == config

    def test_creates_nested_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        save_run_config(nested, {"test": True})
        assert (nested / "config.json").exists()


# ===================================================================
# Per-Sample JSON
# ===================================================================

class TestSampleJson:
    def test_save_and_load(self, tmp_path):
        sample = {
            "sample_idx": 0,
            "sample_input": "Alice flies to LAX",
            "predicted_answer": 1198,
            "ground_truth_answer": 1198,
            "is_correct": True,
            "model_output": "The total cost is $1,198.",
            "generation_finished": True,
            "output_length_chars": 25,
            "input_length_tokens": 500,
            "output_length_tokens": 50,
            "sample_time_seconds": 3.14,
            "boost_metadata": "No boosting (baseline)",
        }
        save_sample_json(tmp_path, sample)
        loaded = json.loads((tmp_path / "0.json").read_text())
        assert loaded["sample_idx"] == 0
        assert loaded["is_correct"] is True
        assert loaded["predicted_answer"] == 1198

    def test_none_values(self, tmp_path):
        sample = {"sample_idx": 5, "predicted_answer": None}
        save_sample_json(tmp_path, sample)
        loaded = json.loads((tmp_path / "5.json").read_text())
        assert loaded["predicted_answer"] is None

    def test_creates_directory(self, tmp_path):
        nested = tmp_path / "x" / "y"
        save_sample_json(nested, {"sample_idx": 0})
        assert (nested / "0.json").exists()


# ===================================================================
# Summary JSON
# ===================================================================

class TestSummaryJson:
    def _make_samples(self, n_correct, n_total, n_finished=None):
        if n_finished is None:
            n_finished = n_total
        samples = []
        for i in range(n_total):
            samples.append({
                "is_correct": i < n_correct,
                "generation_finished": i < n_finished,
                "output_length_chars": 100 + i,
                "output_length_tokens": 50 + i,
            })
        return samples

    def test_basic_summary(self, tmp_path):
        samples = self._make_samples(7, 100, 95)
        config = {"model": "gpt2", "domain": "airline"}
        save_summary_json(tmp_path, config, samples, wall_time_seconds=120.5)

        loaded = json.loads((tmp_path / "summary.json").read_text())
        assert loaded["accuracy"] == pytest.approx(0.07)
        assert loaded["correct_count"] == 7
        assert loaded["total_count"] == 100
        assert loaded["generation_finished_count"] == 95
        assert loaded["generation_finished_ratio"] == pytest.approx(0.95)
        assert loaded["wall_time_seconds"] == 120.5
        assert loaded["config"] == config
        assert "timestamp" in loaded

    def test_empty_samples(self, tmp_path):
        save_summary_json(tmp_path, {}, [], wall_time_seconds=0.0)
        loaded = json.loads((tmp_path / "summary.json").read_text())
        assert loaded["accuracy"] == 0.0
        assert loaded["total_count"] == 0

    def test_creates_directory(self, tmp_path):
        nested = tmp_path / "a" / "b"
        save_summary_json(nested, {}, [], 0.0)
        assert (nested / "summary.json").exists()


# ===================================================================
# Extract Sample Input
# ===================================================================

class TestExtractSampleInput:
    def test_airline(self):
        problem = {"prompt": "Alice is flying from NYC to LAX."}
        result = extract_sample_input("airline", problem)
        assert result == "Alice is flying from NYC to LAX."

    def test_nba(self):
        problem = {
            "team_situations": ["Team A has salary $100M."],
            "player_situations": ["Player A signed a deal."],
            "operations": ["A. Team A signs Player A."],
        }
        result = extract_sample_input("nba", problem)
        assert "Team Situations:" in result
        assert "Operations:" in result

    def test_tax(self):
        problem = {
            "dict": {
                "name": "John",
                "filing_status": "Single",
                "age": 30,
                "spouse_age": None,
                "itemized": False,
                "self_employed": True,
                "num_qualifying_children": 0,
                "num_other_dependents": 0,
            }
        }
        result = extract_sample_input("tax", problem)
        assert "John" in result
        assert "Single" in result
        assert "Self-employed: True" in result

    def test_unknown_domain_raises(self):
        with pytest.raises(ValueError, match="Unknown domain"):
            extract_sample_input("chess", {})


# ===================================================================
# Problem Loading
# ===================================================================

class TestProblemLoading:
    def test_load_airline(self):
        problems = load_problems("airline", 0)
        assert len(problems) == 100
        assert "prompt" in problems[0]
        assert "info" in problems[0]

    def test_load_nba(self):
        problems = load_problems("nba", 0)
        assert len(problems) > 0
        assert "team_situations" in problems[0]
        assert "operations" in problems[0]

    def test_load_tax(self):
        problems = load_problems("tax", 0)
        assert len(problems) == 100
        assert "dict" in problems[0]
        assert "pydantic" in problems[0]


# ===================================================================
# GPT-OSS Output Parsing
# ===================================================================

class TestParseGptOssOutput:
    def test_both_channels(self):
        text = (
            "<|channel|>analysis<|message|>Let me think about this...<|end|>"
            "<|channel|>final<|message|>The answer is 42.<|end|>"
        )
        parsed = parse_gpt_oss_output(text)
        assert parsed['reasoning'] == "Let me think about this..."
        assert parsed['final'] == "The answer is 42."
        assert parsed['raw'] == text

    def test_final_only(self):
        text = "<|channel|>final<|message|>Just the answer.<|end|>"
        parsed = parse_gpt_oss_output(text)
        assert parsed['reasoning'] is None
        assert parsed['final'] == "Just the answer."

    def test_no_channels(self):
        text = "Plain text with no channel markers."
        parsed = parse_gpt_oss_output(text)
        assert parsed['reasoning'] is None
        assert parsed['final'] is None
        assert parsed['raw'] == text

    def test_commentary_channel(self):
        text = (
            "<|channel|>commentary<|message|>Thinking...<|end|>"
            "<|channel|>final<|message|>Done.<|end|>"
        )
        parsed = parse_gpt_oss_output(text)
        assert parsed['reasoning'] == "Thinking..."
        assert parsed['final'] == "Done."

    def test_multiline_content(self):
        text = (
            "<|channel|>analysis<|message|>Step 1: check rules\n"
            "Step 2: compute fees\nStep 3: sum up<|end|>"
            "<|channel|>final<|message|>The total cost is $1,198.<|end|>"
        )
        parsed = parse_gpt_oss_output(text)
        assert "Step 1" in parsed['reasoning']
        assert "Step 3" in parsed['reasoning']
        assert parsed['final'] == "The total cost is $1,198."

    def test_final_with_return_delimiter(self):
        text = "<|channel|>final<|message|>Answer here.<|return|>"
        parsed = parse_gpt_oss_output(text)
        assert parsed['final'] == "Answer here."

    def test_channel_output_used_for_answer_extraction(self):
        """When channels are present, final channel should be used for answer extraction."""
        raw_text = (
            "<|channel|>analysis<|message|>Computing bags...<|end|>"
            "<|channel|>final<|message|>The total cost is $750.<|end|>"
        )
        parsed = parse_gpt_oss_output(raw_text)
        # The final channel text should work with answer extraction
        assert parsed['final'] is not None
        answer = extract_answer_airline(parsed['final'])
        assert answer == 750


# ===================================================================
# Summary JSON — Reasoning/Answer Length Fields
# ===================================================================

class TestSummaryJsonReasoningFields:
    def test_summary_with_reasoning_fields(self, tmp_path):
        samples = [
            {
                "is_correct": True,
                "generation_finished": True,
                "output_length_chars": 500,
                "output_length_tokens": 100,
                "reasoning_length_chars": 300,
                "final_answer_length_chars": 200,
            },
            {
                "is_correct": False,
                "generation_finished": True,
                "output_length_chars": 400,
                "output_length_tokens": 80,
                "reasoning_length_chars": 250,
                "final_answer_length_chars": 150,
            },
        ]
        save_summary_json(tmp_path, {}, samples, wall_time_seconds=10.0)
        loaded = json.loads((tmp_path / "summary.json").read_text())
        assert loaded["avg_reasoning_length_chars"] == 275.0
        assert loaded["avg_final_answer_length_chars"] == 175.0

    def test_summary_without_reasoning_fields(self, tmp_path):
        samples = [
            {
                "is_correct": True,
                "generation_finished": True,
                "output_length_chars": 200,
                "output_length_tokens": 40,
                "reasoning_length_chars": None,
                "final_answer_length_chars": None,
            },
        ]
        save_summary_json(tmp_path, {}, samples, wall_time_seconds=5.0)
        loaded = json.loads((tmp_path / "summary.json").read_text())
        assert loaded["avg_reasoning_length_chars"] is None
        assert loaded["avg_final_answer_length_chars"] is None

    def test_summary_mixed_samples(self, tmp_path):
        """Some samples have reasoning fields, some don't."""
        samples = [
            {
                "is_correct": True,
                "generation_finished": True,
                "output_length_chars": 500,
                "output_length_tokens": 100,
                "reasoning_length_chars": 300,
                "final_answer_length_chars": 200,
            },
            {
                "is_correct": True,
                "generation_finished": True,
                "output_length_chars": 200,
                "output_length_tokens": 40,
                "reasoning_length_chars": None,
                "final_answer_length_chars": None,
            },
        ]
        save_summary_json(tmp_path, {}, samples, wall_time_seconds=8.0)
        loaded = json.loads((tmp_path / "summary.json").read_text())
        # Only the one sample with values should be averaged
        assert loaded["avg_reasoning_length_chars"] == 300.0
        assert loaded["avg_final_answer_length_chars"] == 200.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
