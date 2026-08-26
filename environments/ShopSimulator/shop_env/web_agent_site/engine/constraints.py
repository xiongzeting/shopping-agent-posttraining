"""Pure, reproducible goal helpers for Environment v2."""

from __future__ import annotations

import hashlib
import math
import random
import re

from shopping_grpo.price_semantics import (
    explicit_budget_upper,
)


CONSTRAINT_CONTRACT_VERSION = "shopping-task-constraints-v1"

def _clean_annotation_values(value):
    """Normalize task-authored annotations without inferring new constraints."""
    if not isinstance(value, list):
        return [], False
    cleaned = []
    for item in value:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned, True


def compile_task_constraint_contract(instruction_record):
    """Compile the task's existing annotations into a fail-closed contract.

    Environment v2 uses only fields already attached to the current natural
    language instruction. It deliberately does not infer brand/model or read
    extra attributes from the target product.
    """
    if not isinstance(instruction_record, dict):
        instruction_record = {}
    attributes, attributes_valid = _clean_annotation_values(
        instruction_record.get("attributes")
    )
    options, options_valid = _clean_annotation_values(
        instruction_record.get("instruction_options")
    )
    instruction_valid = bool(
        re.sub(r"\s+", "", str(instruction_record.get("instruction") or ""))
    )
    complete = instruction_valid and attributes_valid and options_valid
    return {
        "hard_constraints": {
            "complete": complete,
            "contract_version": CONSTRAINT_CONTRACT_VERSION,
            "annotation_source": "instruction.attributes",
            "core_functions": attributes,
            "brand": [],
            "model": [],
            "key_specs": [],
            # Options continue to be checked by Reward v2's key_options gate.
            "annotated_option_count": len(options),
        },
        # The current task data does not label hard versus soft preferences.
        # Do not guess that distinction from keywords.
        "weighted_preferences": [],
    }


def explicit_budget_from_instruction(instruction):
    """Extract a clearly stated upper budget; return None when ambiguous."""
    return explicit_budget_upper(instruction)


def _price_range_above(price):
    if price <= 100:
        step = 3
    elif price <= 1000:
        step = 10
    elif price <= 5000:
        step = 50
    elif price <= 10000:
        step = 100
    else:
        step = 4
    base = math.ceil(price / 10) * 10
    return [base + index * 10 for index in range(step)]


def deterministic_price_upper(asin, instruction, price):
    explicit = explicit_budget_from_instruction(instruction)
    if explicit is not None:
        return explicit
    price_range = _price_range_above(float(price))
    if len(price_range) < 2:
        return 10000000
    digest = hashlib.sha256(
        f"{asin}\0{instruction}".encode("utf-8")
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    _, upper = sorted(rng.sample(price_range, 2))
    return upper
