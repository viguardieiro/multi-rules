#!/usr/bin/env python3
"""
Test baseline instruction-following for GPT-OSS-20B.
Runs various instruction types and reports which ones the model follows.

Usage:
    # From notebook (model already loaded):
    %run ../scripts/test_baseline_instructions.py
    results = run_tests(model, tokenizer, device)

    # Or standalone (loads its own model):
    python test_baseline_instructions.py
"""

import sys
import re
import torch


# ============================================================================
# Output parsing (for reasoning models)
# ============================================================================

def parse_gpt_oss_output(generated_text: str) -> dict:
    """Parse GPT-OSS-20B output into reasoning and final answer."""
    result = {'reasoning': None, 'final': None, 'raw': generated_text}

    # Extract analysis/reasoning channel
    analysis_match = re.search(
        r'<\|channel\|>(?:analysis|commentary)<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>|$)',
        generated_text, re.DOTALL
    )
    if analysis_match:
        result['reasoning'] = analysis_match.group(1).strip()

    # Extract final answer channel
    final_match = re.search(
        r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|<\|channel\|>|$)',
        generated_text, re.DOTALL
    )
    if final_match:
        result['final'] = final_match.group(1).strip()

    return result


# ============================================================================
# Instruction checkers
# ============================================================================

def check_word_count(target):
    def checker(answer):
        if answer is None:
            return False, "No answer"
        words = answer.split()
        return len(words) == target, f"Got {len(words)} words: {words}"
    return checker

def check_single_word(answer):
    if answer is None:
        return False, "No answer"
    words = answer.strip().split()
    return len(words) == 1, f"Got {len(words)} words: '{answer}'"

def check_single_number(answer):
    if answer is None:
        return False, "No answer"
    cleaned = answer.strip().rstrip('.')
    is_number = cleaned.isdigit() or (cleaned.replace('.','',1).replace('-','',1).isdigit())
    return is_number, f"Answer: '{answer}'"

def check_all_caps(answer):
    if answer is None:
        return False, "No answer"
    alpha_chars = [c for c in answer if c.isalpha()]
    if not alpha_chars:
        return False, "No alphabetic characters"
    all_upper = all(c.isupper() for c in alpha_chars)
    return all_upper, f"Answer: '{answer}'"

def check_all_lowercase(answer):
    if answer is None:
        return False, "No answer"
    alpha_chars = [c for c in answer if c.isalpha()]
    if not alpha_chars:
        return False, "No alphabetic characters"
    all_lower = all(c.islower() for c in alpha_chars)
    return all_lower, f"Answer: '{answer}'"

def check_starts_with(prefix):
    def checker(answer):
        if answer is None:
            return False, "No answer"
        starts = answer.strip().lower().startswith(prefix.lower())
        return starts, f"Answer: '{answer[:50]}...'" if len(answer) > 50 else f"Answer: '{answer}'"
    return checker

def check_ends_with(suffix):
    def checker(answer):
        if answer is None:
            return False, "No answer"
        ends = answer.strip().endswith(suffix)
        return ends, f"Answer: '{answer}'"
    return checker

def check_no_letter(letter):
    def checker(answer):
        if answer is None:
            return False, "No answer"
        has_letter = letter.lower() in answer.lower()
        return not has_letter, f"Contains '{letter}': {has_letter}, answer: '{answer}'"
    return checker

def check_contains_word(word):
    def checker(answer):
        if answer is None:
            return False, "No answer"
        contains = word.lower() in answer.lower()
        return contains, f"Contains '{word}': {contains}, answer: '{answer}'"
    return checker

def check_bullet_count(target):
    def checker(answer):
        if answer is None:
            return False, "No answer"
        bullets = len(re.findall(r'^\s*[-*•]|\d+\.', answer, re.MULTILINE))
        return bullets == target, f"Got {bullets} bullets"
    return checker

def check_yes_or_no(answer):
    if answer is None:
        return False, "No answer"
    cleaned = answer.strip().lower().rstrip('.!,')
    is_yes_no = cleaned in ['yes', 'no']
    return is_yes_no, f"Answer: '{answer}'"

def check_language_spanish(answer):
    if answer is None:
        return False, "No answer"
    spanish_indicators = ['el', 'la', 'es', 'son', 'está', 'cuatro', 'dos', 'tres', 'uno',
                          'sí', 'que', 'de', 'en', 'por', 'para', 'con']
    answer_lower = answer.lower()
    is_numeric = answer.strip().isdigit()
    has_spanish = any(w in answer_lower.split() for w in spanish_indicators) or is_numeric
    english_words = ['the', 'is', 'are', 'this', 'that', 'answer', 'it']
    has_english = any(w in answer_lower.split() for w in english_words)
    return has_spanish and not has_english, f"Answer: '{answer}'"


# ============================================================================
# Test cases
# ============================================================================

TEST_CASES = [
    # Word count constraints
    ("3 words", "Answer in exactly 3 words: What is the capital of France?",
     "Answer in exactly 3 words", check_word_count(3)),

    ("1 word", "Answer in exactly one word: What color is the sky?",
     "Answer in exactly one word", check_single_word),

    ("5 words", "Answer in exactly 5 words: Why do birds fly?",
     "Answer in exactly 5 words", check_word_count(5)),

    # Format constraints
    ("number only", "Reply with only a number: What is 15 + 27?",
     "Reply with only a number", check_single_number),

    ("yes/no only", "Answer only yes or no: Is the Earth round?",
     "Answer only yes or no", check_yes_or_no),

    # Case constraints
    ("ALL CAPS", "RESPOND IN ALL CAPITAL LETTERS: What is your name?",
     "RESPOND IN ALL CAPITAL LETTERS", check_all_caps),

    ("lowercase", "respond entirely in lowercase letters: What is the largest planet?",
     "respond entirely in lowercase", check_all_lowercase),

    # Starting/ending constraints
    ("start with Yes", "Start your response with 'Yes': Can dogs swim?",
     "Start your response with 'Yes'", check_starts_with("yes")),

    ("end with period", "End your response with exactly one period: Name a fruit",
     "End your response with exactly one period", check_ends_with(".")),

    # Exclusion constraints
    ("no letter e", "Answer without using the letter 'e': What is 2+2?",
     "without using the letter 'e'", check_no_letter('e')),

    # Inclusion constraints
    ("include hello", "Include the word 'hello' in your response: What time is it?",
     "Include the word 'hello'", check_contains_word('hello')),

    # Language constraints
    ("Spanish only", "Responde solo en español: What is 2 + 2?",
     "Responde solo en español", check_language_spanish),

    # List format
    ("3 bullets", "Give exactly 3 bullet points: Name some colors",
     "Give exactly 3 bullet points", check_bullet_count(3)),
]


# ============================================================================
# Main test function
# ============================================================================

def run_tests(model, tokenizer, device, max_new_tokens=600, verbose=True):
    """
    Run all instruction-following tests.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        device: torch device
        max_new_tokens: Maximum tokens to generate
        verbose: Whether to print detailed output

    Returns:
        List of result dictionaries
    """
    results = []

    if verbose:
        print("=" * 90)
        print("BASELINE INSTRUCTION-FOLLOWING TEST (No Boosting)")
        print("=" * 90)

    for name, user_message, instruction, checker in TEST_CASES:
        if verbose:
            print(f"\n{'─' * 90}")
            print(f"TEST: {name}")
            print(f"Prompt: \"{user_message}\"")
            print(f"{'─' * 90}")

        # Prepare input
        messages = [{"role": "user", "content": user_message}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(formatted, return_tensors="pt").input_ids.to(device)
        input_len = ids.shape[1]

        # Generate
        with torch.no_grad():
            output = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        num_generated = output[0].shape[0] - input_len
        hit_limit = num_generated >= max_new_tokens

        generated_text = tokenizer.decode(output[0][input_len:], skip_special_tokens=False)
        parsed = parse_gpt_oss_output(generated_text)

        passed, explanation = checker(parsed['final'])

        results.append({
            'name': name,
            'prompt': user_message,
            'instruction': instruction,
            'num_tokens': num_generated,
            'hit_limit': hit_limit,
            'has_final': parsed['final'] is not None,
            'final_answer': parsed['final'],
            'passed': passed,
            'explanation': explanation,
        })

        if verbose:
            status = "HIT LIMIT" if hit_limit else f"{num_generated} tokens"
            print(f"Generation: [{status}]")

            if parsed['reasoning']:
                preview = parsed['reasoning'][:200] + "..." if len(parsed['reasoning']) > 200 else parsed['reasoning']
                print(f"Reasoning: {preview}")

            print(f"Final Answer: {parsed['final'] or '(none)'}")
            print(f"Result: {'PASS' if passed else 'FAIL'} - {explanation}")

    return results


def print_summary(results):
    """Print summary of test results."""
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    total = len(results)
    passed_count = sum(1 for r in results if r['passed'])

    print(f"\nOverall: {passed_count}/{total} passed ({100*passed_count/total:.0f}%)")
    print(f"Hit token limit: {sum(1 for r in results if r['hit_limit'])}/{total}")
    print(f"No final answer: {sum(1 for r in results if not r['has_final'])}/{total}")

    print(f"\n{'Test':<20} {'Tokens':<10} {'Final?':<8} {'Result':<8} {'Answer':<35}")
    print("─" * 90)

    for r in results:
        tokens = f"{r['num_tokens']}{'*' if r['hit_limit'] else ''}"
        has_final = "Y" if r['has_final'] else "N"
        result = "PASS" if r['passed'] else "FAIL"
        answer = r['final_answer'] or "(none)"
        answer = answer[:32] + "..." if len(answer) > 35 else answer
        print(f"{r['name']:<20} {tokens:<10} {has_final:<8} {result:<8} {answer:<35}")

    # Candidates for boosting
    candidates = [r for r in results if r['has_final'] and not r['passed']]
    if candidates:
        print("\n" + "=" * 90)
        print("GOOD CANDIDATES FOR BOOSTING (produced answer but didn't follow instruction)")
        print("=" * 90)
        for r in candidates:
            print(f"\n  {r['name']}")
            print(f"    Instruction: \"{r['instruction']}\"")
            print(f"    Got: \"{r['final_answer']}\"")
            print(f"    Issue: {r['explanation']}")


# ============================================================================
# Standalone execution
# ============================================================================

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 90)
    print("Loading model...")
    print("=" * 90)

    model_name = "openai/gpt-oss-20b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    results = run_tests(model, tokenizer, device)
    print_summary(results)
