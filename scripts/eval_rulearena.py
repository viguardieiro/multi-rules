#!/usr/bin/env python3
"""
Evaluate local HuggingFace models on the RuleArena benchmark with optional
InstABoost attention boosting.

Domains: airline, nba, tax
"""

import os
import re
import sys
import json
import copy
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RULEARENA_DIR = PROJECT_ROOT / "datasets" / "RuleArena"

# Ensure project root is on sys.path so `from src import ...` works
# even when this script is run as a subprocess (e.g. from sweep_boost_bias.py).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# System prompts & prompt templates (verbatim from each domain's auto_test.py)
# NOTE: These are duplicated here because the auto_test.py files in each
# domain have module-level argparse.parse_args() and API client setup,
# so importing from them would crash. The importable modules (prompt.py,
# compute_answer.py, structured_forms.py, micro_evaluation.py) are used
# where available.
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "airline": (
        "You are a helpful assistant at American Airlines. You are given the "
        "information of a passenger, his / her items, his / her special needs, "
        "and the policies of American Airlines. You should compute the total cost "
        "(including the flight ticket fee, checked bag fees, cost of special needs) "
        "according to the policies for the passenger."
    ),
    "nba": "You are a helpful NBA team consultant.",
    "tax": "You are a helpful US taxation consultant.",
}

# ---------------------------------------------------------------------------
# Prompt templates (verbatim from each domain's auto_test.py)
# ---------------------------------------------------------------------------

AIRLINE_PROMPT_TEMPLATE = """
The policies of American Airlines are as follows:

$reference_rules
$example_prompt
$question_prompt Compute the total cost for him step by step (don't omit any bag) and end your response with "The total cost is $xxx." (xxx is a number)
Your response:
"""

AIRLINE_EXAMPLE = """
Here is an example for you to follow:
<example>
John is a Main Plus Class passenger flying from Orlando to Lima with the following items:
1. A backpack: 21 x 11 x 7 inches, 9 lbs;
2. A luggage box: 51 x 24 x 27 inches, 65 lbs;
3. A backpack: 46 x 29 x 24 inches, 85 lbs;
4. A luggage box: 35 x 12 x 11 inches, 44 lbs;
5. A luggage box: 30 x 11 x 10 inches, 33 lbs;
John's flight ticket is $448. Compute the total cost for him step by step and end your response with "The total cost is $xxx." (xxx is a number)
Your response:
To calculate the total cost for John, we need to consider the flight ticket cost, checked bag fees, and any overweight or oversize fees according to American Airlines' policies.

Flight Ticket Cost:
- John's flight ticket: $448

Carry-on Allowance:
- John is allowed 1 carry-on bag and 1 personal item.
- Item 1 (Backpack):
  - Dimensions: 21 x 11 x 7 inches
  - Weight: 9 lbs
  - This fits within the carry-on size limit of 22 x 14 x 9 inches.
  - Fee: $0 (Allowed as carry-on bag)
- John does not have any items that fit the personal item dimensions (18 x 14 x 8 inches). So, no personal item is carried.

Checked Bags:
- Items to be checked: Items 2, 3, 4, and 5
- John is a Main Plus passenger, which includes 1 extra free checked bag in addition to the Main Cabin allowance, for a total of 2 free checked bags.
- Checked Bag Fees:
  - First Bag: $0 (free)
  - Second Bag: $0 (free)
  - Third Bag: $200
  - Fourth Bag: $200

Fees for Each Checked Bag:

1. Item 2 (Luggage box):
   - Dimensions: 51 x 24 x 27 inches
     - Total dimensions: 51 + 24 + 27 = 102 inches
     - Over the standard size limit of 62 inches.
   - Weight: 65 lbs
     - Over the standard weight limit of 50 lbs but under 70 lbs.
   - Checking Fee:
     - For the first checked bag, the checking fee is $0.
   - Oversize Fee:
     - For dimensions over 65 inches up to 115 inches between the U.S. and South America, the fee is $150.
   - Overweight Fee:
     - For weights over 53 lbs up to 70 lbs, the fee is $100.
   - The higher of oversize and overweight fee should apply.
   - Total Fee for Item 2: $0 (checking) + $150 (oversize) = $150
2. Item 3 (Backpack):
   - Dimensions: 46 x 29 x 24 inches
     - Total dimensions: 46 + 29 + 24 = 99 inches
     - Over the standard size limit of 62 inches.
   - Weight: 85 lbs
     - Over the standard weight limit of 50 lbs and over 70 lbs but under 100 lbs.
   - Checking Fee:
     - For the second checked bag, the checking fee is $0.
   - Oversize Fee:
     - For dimensions over 65 inches up to 115 inches between the U.S. and South America, the fee is $150.
   - Overweight Fee:
     - For weights over 70 lbs up to 100 lbs, the fee is $200.
   - The higher of oversize and overweight fee should apply.
   - Total Fee for Item 3: $0 (checking) + $200 (overweight) = $200
3. Item 4 (Luggage box):
   - Dimensions: 35 x 12 x 11 inches
     - Total dimensions: 35 + 12 + 11 = 58 inches
     - Within the standard size limit of 62 inches.
   - Weight: 44 lbs
     - Within the standard weight limit of 50 lbs.
   - Checking Fee:
     - For the third checked bag, the checking fee is $200.
   - Total Fee for Item 4: $200 (checking) + $0 (No overweight or oversize fees) = $200
4. Item 5 (Luggage box):
   - Dimensions: 30 x 11 x 10 inches
     - Total dimensions: 30 + 11 + 10 = 51 inches
     - Within the standard size limit of 62 inches.
   - Weight: 33 lbs
     - Within the standard weight limit of 50 lbs.
   - Checking Fee:
     - For the fourth checked bag, the checking fee is $200.
   - Total Fee for Item 5: $200 (checking) + $0 (No overweight or oversize fees) = $200
Summary of Baggage Fees:
  - Item 2: $200
  - Item 3: $150
  - Item 4: $200
  - Item 5: $200
Total Baggage Fees: $200 (Item 2) + $150 (Item 3) + $200 (Item 4) + $200 (Item 5) = $750
Total Cost:
- Flight Ticket: $448
- Total Baggage Fees: $750
- The total cost is $1,198.
</example>
"""

NBA_PROMPT_TEMPLATE = """
You are given rules in NBA Collective Bargaining Agreement and the information about some teams and players. Then you will be given a list of operations, each of which desribes how some teams conduct some transaction. You should determine whether each operation complies with the given rules.

Assume:
* the Salary Cap for the prior (2023-24) Salary Cap Year is $136,000,000;
* the Average Player Salary for the prior (2023-24) Salary Cap Year is $9,700,000;
* the Salary Cap for the current (2024-25) NBA Salary Cap Year is $140,588,000;
* the Luxury Tax is $170,814,000;
* the First Apron Level is $178,132,000;
* the Second Apron Level is $188,931,000;
* the Team Salary of each team listed under "Team Situations:" do not include the amount of contracts that expire at the end of 2023-2024 Salary Cap Year.

Reference Rules in NBA Collective Bargaining Agreement:

$reference_rules
$example
Decide whether any operation by any team violate the rules:

$question

Analyze the described operations and explicitly state the type of Salary Cap Exceptions if you think the exception should be involved. Conclude your response with:
* "Answer: False." if there is no violation to the rules;
* "Answer: True. Illegal Operation: X. Problematic Team: Y." if Team Y in Operation X violates the rules. Both X and Y should be a single capital letter as A/B/C/...
Your response:
"""

NBA_EXAMPLE = """
Here is an example for you to follow:
<example>
Decide whether operations of any team violate the rules:

Team Situations:
Team A has a team salary of $100,000,000.

Player Situations:
Player A was the 3rd first-round pick of Team A in 2009 NBA draft when he was 20 years old.
Player A signed a 3-year contract (annual salary $22,000,000, 5% increase per year) with Team B during 2021 Moratorium Period.
Player B was the 15th first-round pick of Team B in 2014 NBA draft when he was 19 years old.
Player B signed a 2-year contract (annual salary $10,000,000, 5% increase per year) with Team A during 2022 Moratorium Period.

Operations:
A. Team A signs a 4-year contract with Player A providing annual salary $21,000,000 in the first Salary Cap Year (2024-2025) and 5% increase per year.
B. Team A signs a 2-year contract with Player B providing annual salary $8,000,000 in the first Salary Cap Year (2024-2025) and 5% increase per year.

Analyze the described operations and explicitly state the type of Salary Cap Exceptions if you think the exception should be involved. Conclude your response with:
* "Answer: False." if there is no violation to the rules;
* "Answer: True. Illegal Operation: X. Problematic Team: Y." if Team Y in Operation X violates the rules. Both X and Y should be a single capital letter as A/B/C/...
Your response:
To determine whether any of the operations violate the rules, we must analyze each operation and see if any team involved in them is violating the NBA Collective Bargaining Agreement (CBA) rules.

Team Situations:
- Team A: Has a team salary of $100,000,000 (excluding expiring contracts at the end of the 2023-2024 Salary Cap Year).

Player Situations:
- Player A:
  - Drafted 3rd overall by Team A in 2009 at age 20.
  - Signed a 3-year contract during the 2021 Moratorium Period with starting salary of $22,000,000 and 5% annual increases.
- Player B:
  - Drafted 15th overall by Team B in 2014 at age 20.
  - Signed a 2-year contract during the 2022 Moratorium Period with starting salary of $10,000,000 and 5% annual increases.

Operations:

Operation A: Team A signs Player A to a 4-year contract starting at $21,000,000 with 5% annual increases.
- Team A's Situation:
  - Current Team Salary: $100,000,000.
  - Player B's 2-year contract for Team A just expired, so Player B is an Early-Qualifying Veteran Free Agent of Team A.
  - Player B's prior salary: $10,000,000 \u00d7 105% = $10,500,000.
  - Player B has not signed a new contract yet. According to Article VII, Section 4(d)(2), Player B will be included at one hundred thirty percent (130%) of his prior Salary: $10,500,000 \u00d7 130% = $13.65 million.
  - Therefore, Team A's actual Team Salary should be $100,000,000 + $13,650,000 = $113,650,000.
- Player A's Situation:
  - Current Age: 20 + (2024 - 2009) = 35 years old.
  - Years of Service: 2024 - 2009 = 15 years.
  - Completed his last contract in Team B, hence Team A has no bird rights of Player A.
- Maximum Salary Calculation (10 or more Years of Service):
  - 35% of Salary Cap ($140,588,000) = $49,205,800.
  - Proposed starting salary of $21,000,000 is within the maximum.
- Salary Cap Space Consumption:
  - The new contract covers 4 seasons. After the new contract ends, the player will be 35 + 4 = 39 years old.
  - Therefore, the salary of the fourth Salary Cap Year (2027-2028) should be attributed to the prior Salary Cap Years pro rata on the basis of the Salaries for such prior Salary Cap Years.
  - Salary in each Salary Cap Year:
    - 2024-2025: $21,000,000
    - 2025-2026: $21,000,000 + $21,000,000 \u00d7 5% = $22,050,000
    - 2026-2027: $22,050,000 + $21,000,000 \u00d7 5% = $23,100,000
    - 2027-2028: $23,100,000 + $21,000,000 \u00d7 5% = $24,150,000
  - Attribute the salary of 2027-2028 Salary Cap Year to the first three Salary Cap Years:
    - 2024-2025: $21,000,000 + $24,150,000 \u00d7 ($21,000,000 / ($21,000,000 + $22,050,000 + $23,100,000)) \u2248 $28.67 million.
    - 2025-2026: $22,050,000 + $24,150,000 \u00d7 ($22,050,000 / ($21,000,000 + $22,050,000 + $23,100,000)) = $30.10 million.
    - 2026-2027: $23,100,000 + $24,150,000 \u00d7 ($23,100,000 / ($21,000,000 + $22,050,000 + $23,100,000)) \u2248 $31.53 million.
  - After signing Player A, Team A's Team Salary for 2024-2025 Salary Cap Year becomes $113,650,000 + $28.67 million = $142.32 million.
  - Team A's Team Salary after signing exceeds the Salary Cap, it must use a Salary Cap Exception to sign Player A.
  - Team A's Team Salary before signing is below the Salary Cap, so the only exception it can use is the Mid-Level Salary Exceptio (MLE) for Room Teams.
  - The MLE for Room Teams allows a maximum salary of $140,588,000 \u00d7 5.678% \u2248 $7.98 million < $28.67 million.
  - Therefore, Team A cannot use the MLE for Room Teams to sign Player A, and Operation A violates the rules.

Operation B: Team A signs Player B to a 2-year contract starting at $8,000,000 with 5% annual increases.
- Team A's Situation After Operation A:
  - Team Salary: $142.32 million
- Player B's Situation:
  - Current Age: 20 + (2024 - 2014) = 30 years old.
  - Years of Service: 2024 - 2014 = 10 years.
  - Completed his last 2-year contract in Team A, hence Player B is an Early-Qualifying Veteran Free Agent of Team A.
- Maximum Salary Calculation (10 or more Years of Service):
  - 35% of Salary Cap ($140,588,000) = $49,205,800.
  - Proposed starting salary of $8,000,000 is within the maximum.
- Player B does not trigger the Over 38 Rule, so the Salary for 2024-2025 Salary Cap Year is $8,000,000.
- Team A has exceeded the Salary Cap, so it must use a Salary Cap Exception to sign Player B:
  - Team A has the early Bird rights for Player B, which allows a maximum salary as the greater of:
    - 175% of the Regular Salary for the final Salary Cap Year covered by his prior Contract: 175% \u00d7 $10,500,000 = $18.375 million.
    - 105% of the Average Player Salary for the prior Salary Cap Year: 105% \u00d7 $9,700,000 = $10.185 million.
  - The greater is $18.375 million > $8,000,000, and thus Team A can use its early Bird rights to re-sign Player B.

Conclusion:
  - Team A cannot sign Player A to a 4-year contract starting at $21,000,000 with 5% annual increases as it would exceed the Salary Cap.
  - Team A can re-sign Player B using its early Bird rights.

Answer: True. Illegal Operation: A. Problematic Team: A.
</example>
"""

TAX_PROMPT_TEMPLATE = """
You are given several forms used to report US income tax and the instructions or rules about how to fill the forms. Then you will be given the income and/or payment information about a tax payer According to the given information. You should calculate the income tax owed by this payer.
$example
IRS Forms for the tax payer:
$forms
Calculate the tax owed by the payer step-by-step according to the information provided by the forms. You should calculate all fields marked with [__]. DO NOT round numbers without explicit instructions. End your response with:
1. "The total tax owed is $xxx." (xxx is a number) if there is tax owed.
2. "The total tax overpaid is $xxx." (xxx is a number) if there is tax overpaid (and should be refunded).
Your response:
"""

# RuleArena default max new tokens (for API models, output only)
# DEFAULT_MAX_NEW_TOKENS = {"airline": 4096, "nba": 4096, "tax": 8192}

# Our default max new tokens (for local models, counting reasoning and output tokens)
DEFAULT_MAX_NEW_TOKENS = {"airline": 16000, "nba": 16000, "tax": 16000}

TBD_MARK = "[__]"


# ===================================================================
# GPT-OSS Output Parsing
# ===================================================================

def parse_gpt_oss_output(generated_text: str) -> dict:
    """Parse GPT-OSS-20B output into reasoning and final answer.

    GPT-OSS uses channel tokens to separate reasoning from final answer:
      <|channel|>analysis<|message|>...reasoning...<|end|>
      <|channel|>final<|message|>...answer...<|end|>

    Returns dict with keys: 'reasoning', 'final', 'raw'.
    """
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


# ===================================================================
# Problem Loading
# ===================================================================

def load_problems_airline(complexity: int) -> list:
    path = RULEARENA_DIR / "airline" / "synthesized_problems" / f"comp_{complexity}.jsonl"
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def load_problems_nba(complexity: int) -> list:
    path = RULEARENA_DIR / "nba" / "annotated_problems" / f"comp_{complexity}.json"
    with open(path, "r") as f:
        return json.load(f)


def load_problems_tax(complexity: int) -> list:
    path = RULEARENA_DIR / "tax" / "synthesized_problems" / f"comp_{complexity}.json"
    with open(path, "r") as f:
        return json.load(f)


def load_problems(domain: str, complexity: int) -> list:
    loaders = {
        "airline": load_problems_airline,
        "nba": load_problems_nba,
        "tax": load_problems_tax,
    }
    return loaders[domain](complexity)


# ===================================================================
# Reference Rules Loading
# ===================================================================

def load_reference_rules(domain: str, textual: bool = False) -> str | None:
    if domain == "airline":
        fname = "reference_rules_textual.txt" if textual else "reference_rules.txt"
        path = RULEARENA_DIR / "airline" / fname
        return path.read_text()
    elif domain == "nba":
        path = RULEARENA_DIR / "nba" / "reference_rules.txt"
        return path.read_text()
    elif domain == "tax":
        return None
    else:
        raise ValueError(f"Unknown domain: {domain}")


# ===================================================================
# Prompt Construction
# ===================================================================

def build_prompt_airline(problem: dict, reference_rules: str, use_example: bool) -> tuple[str, str, str]:
    prompt = AIRLINE_PROMPT_TEMPLATE
    prompt = prompt.replace("$reference_rules", reference_rules)
    prompt = prompt.replace("$question_prompt", problem["prompt"])
    if use_example:
        prompt = prompt.replace("$example_prompt", AIRLINE_EXAMPLE)
    else:
        prompt = prompt.replace("$example_prompt", "")
    return prompt, reference_rules, problem["prompt"]


def build_query_prompt_nba(query_dict: dict) -> str:
    team_info = "Team Situations:\n" + "\n".join(query_dict["team_situations"])
    player_info = "Player Situations:\n" + "\n".join(query_dict["player_situations"])
    operations = "Operations:\n" + "\n".join(query_dict["operations"])
    return team_info + "\n\n" + player_info + "\n\n" + operations


def build_prompt_nba(problem: dict, reference_rules: str, use_example: bool) -> tuple[str, str, str]:
    query_prompt = build_query_prompt_nba(problem)
    prompt = NBA_PROMPT_TEMPLATE
    prompt = prompt.replace("$reference_rules", reference_rules)
    prompt = prompt.replace("$question", query_prompt)
    if use_example:
        prompt = prompt.replace("$example", NBA_EXAMPLE)
    else:
        prompt = prompt.replace("$example", "")
    return prompt, reference_rules, query_prompt


def _import_tax_forms():
    """Import form templates from the tax domain's prompt.py."""
    tax_dir = str(RULEARENA_DIR / "tax")
    if tax_dir not in sys.path:
        sys.path.insert(0, tax_dir)
    from prompt import (
        basic_forms,
        basic_forms_textual,
        itemized_forms,
        self_employ_forms,
        edu_forms,
        schedule_8812,
    )
    return basic_forms, basic_forms_textual, itemized_forms, self_employ_forms, edu_forms, schedule_8812


def build_prompt_tax(problem: dict, use_example: bool, textual: bool = False) -> tuple[str, str, None]:
    (basic_forms, basic_forms_textual, itemized_forms,
     self_employ_forms, edu_forms, schedule_8812) = _import_tax_forms()

    # Deep-copy to avoid mutating the original problem
    tax_payer = copy.deepcopy(problem["dict"])

    # Assemble forms based on taxpayer flags
    if textual:
        forms_list = [basic_forms_textual]
    else:
        forms_list = [basic_forms]

    if tax_payer.get("itemized"):
        forms_list.append(itemized_forms)
    if tax_payer.get("self_employed"):
        forms_list.append(self_employ_forms)
    if tax_payer.get("has_student_loans_or_education_expenses"):
        forms_list.append(edu_forms)
    if tax_payer.get("child_and_dependent"):
        forms_list.append(schedule_8812)

    forms = "".join(forms_list)

    # Fill in data placeholders and collect TBD fields
    tbd_fields = []
    for k, v in tax_payer["data"].items():
        replacement = "$" + (f"{v:,}" if not isinstance(v, str) else v)
        forms = forms.replace("$" + k, replacement)
        if v == "$TBD":
            tbd_fields.append(k)

    # Remove TBD fields from data dict (matching original behavior)
    for field in tbd_fields:
        tax_payer["data"].pop(field)

    forms = forms.replace("$TBD", TBD_MARK)

    # Fill top-level taxpayer fields
    prompt = TAX_PROMPT_TEMPLATE.replace("$forms", forms)
    prompt = prompt.replace("$name", tax_payer["name"])
    prompt = prompt.replace("$age", str(tax_payer["age"]))
    prompt = prompt.replace("$spouse_age", str(tax_payer["spouse_age"]))
    prompt = prompt.replace("$blind", str(tax_payer["blind"]))
    prompt = prompt.replace("$spouse_blind", str(tax_payer["spouse_blind"]))
    prompt = prompt.replace("$filing_status", tax_payer["filing_status"])
    prompt = prompt.replace("$itemized", str(tax_payer["itemized"]))
    prompt = prompt.replace("$num_qualifying_children", str(tax_payer["num_qualifying_children"]))
    prompt = prompt.replace("$num_other_dependents", str(tax_payer["num_other_dependents"]))

    if use_example:
        # For now, no example support for local models (would need model-specific examples)
        prompt = prompt.replace("$example", "")
    else:
        prompt = prompt.replace("$example", "")

    boostable_text = forms
    return prompt, boostable_text, None


def build_prompt(domain: str, problem: dict, reference_rules: str | None,
                 use_example: bool, textual: bool = False) -> tuple[str, str, str | None]:
    if domain == "airline":
        return build_prompt_airline(problem, reference_rules, use_example)
    elif domain == "nba":
        return build_prompt_nba(problem, reference_rules, use_example)
    elif domain == "tax":
        return build_prompt_tax(problem, use_example, textual)
    else:
        raise ValueError(f"Unknown domain: {domain}")


def format_chat_prompt(system_prompt: str, user_prompt: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ===================================================================
# Answer Extraction
# ===================================================================

def extract_answer_airline(response: str) -> int | None:
    response = response.replace("**", "")
    start_idx = response.find("The total cost is")
    if start_idx == -1:
        return None
    conclusion = response[start_idx:]
    value_idx = conclusion.find("$")
    if value_idx == -1:
        return None
    value_str = conclusion[value_idx + 1:].replace(",", "").replace(".", "")
    # Take only the numeric prefix
    numeric = ""
    for ch in value_str:
        if ch.isdigit():
            numeric += ch
        else:
            break
    if not numeric:
        return None
    return int(numeric)


def extract_answer_tax(response: str) -> float | None:
    response = response.replace("**", "")
    pattern = r"The total tax (owed|overpaid) is \$((?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)\."
    match = re.search(pattern, response)
    if not match:
        return None
    status = match.group(1)
    value = float(match.group(2).replace(",", ""))
    if status == "overpaid":
        value = -value
    return value


# ===================================================================
# Ground Truth Computation
# ===================================================================

# Cache for airline fee tables (loaded once)
_airline_check_base_tables = None


def compute_ground_truth_airline(info_dict: dict) -> int:
    global _airline_check_base_tables
    airline_dir = str(RULEARENA_DIR / "airline")

    old_cwd = os.getcwd()
    try:
        # Must chdir BEFORE importing because compute_answer.py calls
        # load_checking_fee() at module level, which reads relative paths.
        os.chdir(airline_dir)
        if airline_dir not in sys.path:
            sys.path.insert(0, airline_dir)
        from compute_answer import compute_answer
        if _airline_check_base_tables is None:
            from compute_answer import check_base_tables as _loaded_tables
            _airline_check_base_tables = _loaded_tables
        fee, _ = compute_answer(**info_dict, check_base_tables=_airline_check_base_tables)
    finally:
        os.chdir(old_cwd)
    return fee


def compute_ground_truth_tax(problem: dict) -> float:
    tax_dir = str(RULEARENA_DIR / "tax")
    if tax_dir not in sys.path:
        sys.path.insert(0, tax_dir)

    from structured_forms import TaxPayer
    from micro_evaluation import compute_answer

    tax_payer_pydantic = TaxPayer(**problem["pydantic"])
    answer, _ = compute_answer(tax_payer_pydantic)
    return answer


# ===================================================================
# Accuracy Evaluation
# ===================================================================

def eval_accuracy_airline(response: str, info_dict: dict) -> tuple[bool, int | None, int]:
    pred = extract_answer_airline(response)
    truth = compute_ground_truth_airline(info_dict)
    if pred is None:
        return False, pred, truth
    return pred == truth, pred, truth


def eval_accuracy_nba(response: str, problem: dict) -> tuple[bool, str, str]:
    response_clean = response.replace("**", "")
    answer = problem["answer"]
    if not answer:
        target_response = "Answer: False"
    else:
        illegal_op = problem["illegal_operation"]
        prob_team = problem["problematic_team"]
        target_response = f"Answer: True. Illegal Operation: {illegal_op}. Problematic Team: {prob_team}"
    is_correct = target_response in response_clean
    # Return last 200 chars for logging
    response_tail = response_clean[-200:]
    return is_correct, response_tail, target_response


def eval_accuracy_tax(response: str, problem: dict) -> tuple[bool, float | None, float]:
    import numpy as np
    pred = extract_answer_tax(response)
    truth = compute_ground_truth_tax(problem)
    if pred is None:
        return False, pred, truth
    return bool(np.isclose(pred, truth)), pred, truth


def eval_accuracy(domain: str, response: str, problem: dict) -> tuple[bool, object, object]:
    if domain == "airline":
        return eval_accuracy_airline(response, problem["info"])
    elif domain == "nba":
        return eval_accuracy_nba(response, problem)
    elif domain == "tax":
        return eval_accuracy_tax(response, problem)
    else:
        raise ValueError(f"Unknown domain: {domain}")


# ===================================================================
# Boost Config
# ===================================================================

def build_boost_config(strategy: str, formatted_prompt: str, tokenizer,
                       reference_rules_text: str | None,
                       question_text: str | None, bias: float):
    if strategy == "none" or bias == 0.0:
        return None

    from src import BoostConfig, create_token_subset_from_substring

    if strategy == "uniform_rules":
        if reference_rules_text is None:
            raise ValueError("Cannot boost without reference rules text")
        subset = create_token_subset_from_substring(
            name="rules",
            text=formatted_prompt,
            substring=reference_rules_text,
            tokenizer=tokenizer,
            bias=bias,
        )
        return BoostConfig(subsets=[subset])
    elif strategy == "uniform_question":
        if question_text is None:
            raise ValueError(
                "uniform_question strategy requires question_text "
                "(not supported for tax domain)"
            )
        subset = create_token_subset_from_substring(
            name="question",
            text=formatted_prompt,
            substring=question_text,
            tokenizer=tokenizer,
            bias=bias,
        )
        return BoostConfig(subsets=[subset])
    else:
        raise ValueError(f"Unknown boost strategy: {strategy}")


# ===================================================================
# Results Infrastructure
# ===================================================================

def generate_run_id(strategy: str, bias: float) -> str:
    if strategy == "none" or bias == 0.0:
        return "none"
    return f"{strategy}_bias{bias}"


def build_results_dir(log_dir: str, model_name: str, domain: str,
                      complexity: int, run_id: str) -> Path:
    safe_model = model_name.replace("/", "_")
    return Path(log_dir) / "rulearena" / safe_model / domain / f"comp_{complexity}" / run_id


def save_run_config(results_dir: Path, config_dict: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)


def format_boost_metadata(boost_config, input_length: int) -> str:
    if boost_config is None:
        return "No boosting (baseline)"

    lines = [f"Input length: {input_length} tokens"]
    for subset in boost_config.subsets:
        lines.append(
            f"Subset '{subset.name}': bias={subset.bias}, "
            f"{len(subset.indices)} tokens "
            f"(indices {min(subset.indices)}-{max(subset.indices)})"
        )
    return "\n".join(lines)


def extract_sample_input(domain: str, problem: dict) -> str:
    """Extract the question-specific text from a problem, excluding the rulebook."""
    if domain == "airline":
        return problem["prompt"]
    elif domain == "nba":
        return build_query_prompt_nba(problem)
    elif domain == "tax":
        d = problem.get("dict", {})
        parts = [f"Taxpayer: {d.get('name', 'N/A')}"]
        parts.append(f"Filing status: {d.get('filing_status', 'N/A')}")
        parts.append(f"Age: {d.get('age', 'N/A')}")
        if d.get("spouse_age"):
            parts.append(f"Spouse age: {d['spouse_age']}")
        parts.append(f"Itemized: {d.get('itemized', False)}")
        parts.append(f"Self-employed: {d.get('self_employed', False)}")
        if d.get("num_qualifying_children"):
            parts.append(f"Qualifying children: {d['num_qualifying_children']}")
        if d.get("num_other_dependents"):
            parts.append(f"Other dependents: {d['num_other_dependents']}")
        return " | ".join(parts)
    else:
        raise ValueError(f"Unknown domain: {domain}")


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy/torch types in JSON serialization."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_sample_json(results_dir: Path, sample: dict) -> None:
    """Save a single sample result as results_dir/<sample_idx>.json."""
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{sample['sample_idx']}.json", "w") as f:
        json.dump(sample, f, indent=2, cls=_NumpyEncoder)


def save_summary_json(results_dir: Path, config_dict: dict,
                      samples: list[dict], wall_time_seconds: float) -> None:
    """Compute aggregate metrics and save summary.json."""
    results_dir.mkdir(parents=True, exist_ok=True)
    total = len(samples)
    correct = sum(1 for s in samples if s["is_correct"])
    finished = sum(1 for s in samples if s["generation_finished"])
    avg_output_chars = (
        sum(s["output_length_chars"] for s in samples) / total if total else 0.0
    )
    avg_output_tokens = (
        sum(s["output_length_tokens"] for s in samples) / total if total else 0.0
    )

    # Compute average reasoning and final answer lengths (only for samples that have them)
    reasoning_samples = [s for s in samples if s.get("reasoning_length_chars") is not None]
    final_samples = [s for s in samples if s.get("final_answer_length_chars") is not None]
    avg_reasoning_chars = (
        sum(s["reasoning_length_chars"] for s in reasoning_samples) / len(reasoning_samples)
        if reasoning_samples else None
    )
    avg_final_answer_chars = (
        sum(s["final_answer_length_chars"] for s in final_samples) / len(final_samples)
        if final_samples else None
    )

    summary = {
        "config": config_dict,
        "accuracy": correct / total if total else 0.0,
        "correct_count": correct,
        "total_count": total,
        "generation_finished_count": finished,
        "generation_finished_ratio": finished / total if total else 0.0,
        "avg_output_length_chars": round(avg_output_chars, 1),
        "avg_output_length_tokens": round(avg_output_tokens, 1),
        "avg_reasoning_length_chars": round(avg_reasoning_chars, 1) if avg_reasoning_chars is not None else None,
        "avg_final_answer_length_chars": round(avg_final_answer_chars, 1) if avg_final_answer_chars is not None else None,
        "wall_time_seconds": round(wall_time_seconds, 1),
        "timestamp": datetime.now().isoformat(),
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=_NumpyEncoder)



# ===================================================================
# CLI + Main
# ===================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate HF models on RuleArena")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--domain", type=str, required=True, choices=["airline", "nba", "tax"])
    parser.add_argument("--complexity", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--boost_strategy", type=str, default="none",
                        choices=["none", "uniform_rules", "uniform_question"])
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--rules_strategy", type=str, default="full",
                        choices=["full", "applicable_only"],
                        help="Which rules to include in the prompt: "
                             "'full' = entire rulebook (default), "
                             "'applicable_only' = only rules applicable to each sample")
    parser.add_argument("--drop_fee_summaries", action="store_true",
                        help="When using --rules_strategy applicable_only, drop the "
                             "generic checked-bag fee summary sentences that can "
                             "conflict with cabin-specific fee tables")
    parser.add_argument("--use_example", action="store_true")
    parser.add_argument("--textual", action="store_true")
    parser.add_argument("--log_dir", type=str, default="results")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Index of first problem to evaluate (default: 0)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Determine max_new_tokens
    max_new_tokens = args.max_new_tokens or DEFAULT_MAX_NEW_TOKENS[args.domain]

    # Generate run_id and results dir
    run_id = generate_run_id(args.boost_strategy, args.bias)
    if args.rules_strategy == "applicable_only":
        run_id = "applicable_rules_only"
        if args.drop_fee_summaries:
            run_id += "_no_fee_summaries"
    results_dir = build_results_dir(args.log_dir, args.model, args.domain,
                                    args.complexity, run_id)

    # Save config
    config_dict = {
        "model": args.model,
        "domain": args.domain,
        "complexity": args.complexity,
        "boost_strategy": args.boost_strategy,
        "bias": args.bias,
        "rules_strategy": args.rules_strategy,
        "drop_fee_summaries": args.drop_fee_summaries,
        "use_example": args.use_example,
        "textual": args.textual,
        "max_new_tokens": max_new_tokens,
        "start_idx": args.start_idx,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
    }
    save_run_config(results_dir, config_dict)

    # Load model + tokenizer
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Load problems and reference rules
    problems = load_problems(args.domain, args.complexity)
    reference_rules = load_reference_rules(args.domain, args.textual)

    # Pre-compute rulebook segments for applicable_only strategy
    _ra_fine_segments = None
    _ra_coarse_segments = None
    if args.rules_strategy == "applicable_only":
        if args.domain != "airline":
            raise ValueError("--rules_strategy applicable_only is only supported for airline domain")
        from src.rulearena.rulebook_segments import get_fine_segments, get_coarse_segments
        from src.rulearena.rule_applicability import build_filtered_rulebook
        _ra_fine_segments = get_fine_segments(reference_rules)
        _ra_coarse_segments = get_coarse_segments(reference_rules)
        print(f"Loaded {len(_ra_fine_segments)} fine / {len(_ra_coarse_segments)} coarse segments "
              f"for applicable_only filtering")

    problems = problems[args.start_idx:]
    if args.max_problems is not None:
        problems = problems[:args.max_problems]

    print(f"Loaded {len(problems)} problems for {args.domain} comp_{args.complexity}"
          f" (start_idx={args.start_idx})")

    system_prompt = SYSTEM_PROMPTS[args.domain]
    correct_count = 0
    total_count = len(problems)
    sample_results = []

    wall_start = time.time()

    for idx, problem in enumerate(problems, start=args.start_idx):
        print(f"\n--- Problem {idx + 1}/{total_count + args.start_idx} ---")
        sample_start = time.time()

        # Build per-sample reference rules (filtered or full)
        if args.rules_strategy == "applicable_only":
            sample_reference_rules = build_filtered_rulebook(
                problem["info"], reference_rules, _ra_fine_segments, _ra_coarse_segments,
                drop_fee_summaries=args.drop_fee_summaries,
            )
        else:
            sample_reference_rules = reference_rules

        # Build prompt
        user_prompt, rules_text, question_text = build_prompt(
            args.domain, problem, sample_reference_rules, args.use_example, args.textual
        )

        # Format with chat template
        formatted_prompt = format_chat_prompt(system_prompt, user_prompt, tokenizer)

        # Build boost config
        boost_config = build_boost_config(
            args.boost_strategy, formatted_prompt, tokenizer,
            rules_text, question_text, args.bias,
        )

        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1]

        # Register hooks if boosting
        handle = None
        if boost_config is not None:
            from src import register_boost_hooks, update_bias_mask
            handle = register_boost_hooks(model, boost_config, input_length=input_length)
            update_bias_mask(handle, seq_length=input_length, device=model.device)

        boost_metadata = format_boost_metadata(boost_config, input_length)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Unregister hooks
        if handle is not None:
            from src import unregister_boost_hooks
            unregister_boost_hooks(handle)

        # Decode response only (skip input tokens)
        generated_tokens = output_ids[0][input_length:]
        output_length_tokens = len(generated_tokens)
        generation_finished = (
            output_length_tokens < max_new_tokens
            or generated_tokens[-1].item() == tokenizer.eos_token_id
        )

        # Decode with channel markers preserved, then parse for reasoning models
        response_raw = tokenizer.decode(generated_tokens, skip_special_tokens=False)
        parsed = parse_gpt_oss_output(response_raw)

        if parsed['final'] is not None:
            # Model used channels — use final channel for answer extraction
            response = parsed['final']
            reasoning_length_chars = len(parsed['reasoning']) if parsed['reasoning'] else 0
            final_answer_length_chars = len(parsed['final'])
        else:
            # Non-reasoning model — fall back to clean decode
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            reasoning_length_chars = None
            final_answer_length_chars = None

        # Evaluate
        is_correct, predicted, truth = eval_accuracy(args.domain, response, problem)
        if is_correct:
            correct_count += 1

        sample_time = time.time() - sample_start
        elapsed = time.time() - wall_start
        done = idx - args.start_idx + 1
        avg_time = elapsed / done
        eta = avg_time * (total_count - done)
        eta_min, eta_sec = divmod(int(eta), 60)
        acc_so_far = correct_count / done
        print(f"  Correct: {is_correct} | Predicted: {predicted} | Truth: {truth}"
              f" | {sample_time:.1f}s"
              f" | Running: {correct_count}/{done} ({acc_so_far:.1%})"
              f" | ETA: {eta_min}m{eta_sec:02d}s")

        # Collect sample result and save incrementally
        sample = {
            "sample_idx": idx,
            "sample_input": extract_sample_input(args.domain, problem),
            "predicted_answer": predicted,
            "ground_truth_answer": truth,
            "is_correct": is_correct,
            "model_output": response_raw,
            "generation_finished": generation_finished,
            "output_length_chars": len(response_raw),
            "input_length_tokens": int(input_length),
            "output_length_tokens": int(output_length_tokens),
            "reasoning_length_chars": reasoning_length_chars,
            "final_answer_length_chars": final_answer_length_chars,
            "sample_time_seconds": round(sample_time, 2),
            "boost_metadata": boost_metadata,
        }
        sample_results.append(sample)
        save_sample_json(results_dir, sample)

    wall_time = time.time() - wall_start

    # Save aggregate summary
    save_summary_json(results_dir, config_dict, sample_results, wall_time)

    accuracy = correct_count / total_count if total_count > 0 else 0.0
    print(f"\n=== Final: {correct_count}/{total_count} = {accuracy:.4f} ===")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
