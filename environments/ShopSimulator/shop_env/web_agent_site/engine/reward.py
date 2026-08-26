"""Query-auditable, consumer-oriented terminal Reward v4."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, replace

from web_agent_site.engine.comparators import (
    COMPARATOR_VERSION,
    FAIL,
    PASS,
    UNVERIFIABLE,
    compare_category,
    compare_core_functions,
    compare_model,
    comparison,
    load_brand_aliases,
    normalize_text,
    semantic_text_match,
)
from web_agent_site.engine.reward_features import (
    CONSTRAINT_SEMANTICS_VERSION,
    QUERY_CONSTRAINT_VERSION,
    REWARD_FEATURE_VERSION,
    normalize_option_text,
)
from web_agent_site.engine.variant_price import (
    VARIANT_PRICE_VERSION,
    candidate_options_for_evaluation,
    compare_required_options,
    resolve_variant_price,
)

REWARD_VERSION = "shopsimulator-reward-v4"
DIMENSION_WEIGHTS = {
    "brand": 0.25,
    "model": 0.25,
    "core_functions": 0.25,
    "key_options": 0.25,
}
DEFAULT_REWARDS = {
    "gold_purchase": 1.0,
    "valid_alternative_purchase": 0.80,
    "partial_purchase_base": 0.50,
    "partial_purchase_scale": 0.30,
    "assistant_final": -0.80,
    "early_abstain": -0.40,
    "max_steps": 0.0,
    "repeat_loop": -0.60,
    "wrong_purchase": -1.0,
    "reward_unverifiable": 0.0,
}
KNOWN_ACCEPTABLE_MATCH_THRESHOLD = 0.70
KNOWN_ACCEPTABLE_COVERAGE_THRESHOLD = 0.75
STEP_PENALTY_VERSION = "shopping-step-penalty-v1"
STEP_PENALTY_SCHEDULE = (
    (16, 20, 0.01),
    (21, 25, 0.02),
    (26, 30, 0.03),
    (31, 35, 0.04),
    (36, 40, 0.05),
    (41, 45, 0.06),
)

_MEASUREMENT_UNITS = {
    "毫米": ("length", 0.001),
    "mm": ("length", 0.001),
    "厘米": ("length", 0.01),
    "cm": ("length", 0.01),
    "米": ("length", 1.0),
    "m": ("length", 1.0),
    "克": ("weight", 0.001),
    "g": ("weight", 0.001),
    "千克": ("weight", 1.0),
    "公斤": ("weight", 1.0),
    "kg": ("weight", 1.0),
    "毫升": ("volume", 0.001),
    "ml": ("volume", 0.001),
    "升": ("volume", 1.0),
    "l": ("volume", 1.0),
    "英寸": ("inch", 1.0),
    "寸": ("cun", 1.0),
    "支": ("count", 1.0),
    "个": ("count", 1.0),
    "片": ("count", 1.0),
    "只": ("count", 1.0),
    "套": ("count", 1.0),
    "包": ("count", 1.0),
}
_MEASUREMENT_UNIT_PATTERN = "|".join(
    sorted((re.escape(unit) for unit in _MEASUREMENT_UNITS), key=len, reverse=True)
)
_MEASUREMENT_PATTERN = re.compile(
    rf"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>{_MEASUREMENT_UNIT_PATTERN})(?![a-z])",
    flags=re.IGNORECASE,
)


def _measurements(value: object) -> list[dict]:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    results = []
    for match in _MEASUREMENT_PATTERN.finditer(text):
        unit = match.group("unit").casefold()
        dimension, factor = _MEASUREMENT_UNITS[unit]
        results.append(
            {
                "dimension": dimension,
                "value": float(match.group("value")) * factor,
                "raw": match.group(0),
            }
        )
    return results


def _public_measurement_constraints(instruction: object) -> list[dict]:
    """Extract only explicit relational measurements from the public query.

    Exact hidden option values remain the strict target.  This fallback exists
    only for public expressions such as ``15厘米以下`` where multiple variants
    are genuinely acceptable to the user.
    """
    text = unicodedata.normalize("NFKC", str(instruction or "")).casefold()
    number = rf"(\d+(?:\.\d+)?)\s*({_MEASUREMENT_UNIT_PATTERN})"
    patterns = (
        ("lte", re.compile(rf"(?:不超过|不能超过|至多|最大(?:为|到)?|小于等于)\s*{number}")),
        ("lte", re.compile(rf"{number}\s*(?:以内|以下|及以下)")),
        ("gte", re.compile(rf"(?:不少于|不低于|至少|最小(?:为|到)?|大于等于)\s*{number}")),
        ("gte", re.compile(rf"{number}\s*(?:以上|及以上|起)")),
        ("around", re.compile(rf"{number}\s*(?:左右|上下)")),
    )
    constraints = []
    occupied = []
    range_pattern = re.compile(
        rf"(\d+(?:\.\d+)?)\s*({_MEASUREMENT_UNIT_PATTERN})?\s*"
        rf"(?:-|~|～|至|到)\s*(\d+(?:\.\d+)?)\s*({_MEASUREMENT_UNIT_PATTERN})"
    )
    for match in range_pattern.finditer(text):
        low_unit = (match.group(2) or match.group(4)).casefold()
        high_unit = match.group(4).casefold()
        low_dimension, low_factor = _MEASUREMENT_UNITS[low_unit]
        high_dimension, high_factor = _MEASUREMENT_UNITS[high_unit]
        if low_dimension != high_dimension:
            continue
        low = float(match.group(1)) * low_factor
        high = float(match.group(3)) * high_factor
        if high < low:
            continue
        occupied.append((match.start(), match.end()))
        constraints.append(
            {
                "operator": "between",
                "dimension": low_dimension,
                "min": low,
                "max": high,
                "text": match.group(0),
            }
        )
    for operator, pattern in patterns:
        for match in pattern.finditer(text):
            if any(match.start() < end and start < match.end() for start, end in occupied):
                continue
            unit = match.group(2).casefold()
            dimension, factor = _MEASUREMENT_UNITS[unit]
            value = float(match.group(1)) * factor
            record = {
                "operator": operator,
                "dimension": dimension,
                "value": value,
                "text": match.group(0),
            }
            if operator == "around":
                record.update({"min": value * 0.9, "max": value * 1.1})
            constraints.append(record)
    return constraints


def _measurement_satisfies(measurement: dict, constraint: dict) -> bool:
    if measurement["dimension"] != constraint["dimension"]:
        return False
    value = measurement["value"]
    operator = constraint["operator"]
    if operator == "lte":
        return value <= constraint["value"]
    if operator == "gte":
        return value >= constraint["value"]
    if operator in {"between", "around"}:
        return constraint["min"] <= value <= constraint["max"]
    return False


def _public_option_semantic_match(
    product: dict,
    canonical_axis: str,
    requirement: object,
    selected_options: object,
    instruction: object,
    exact_result: dict,
) -> dict | None:
    if exact_result.get("status") == PASS or not isinstance(selected_options, dict):
        return None
    required_value = requirement.get("value") if isinstance(requirement, dict) else requirement
    public_evidence = (
        requirement.get("query_evidence")
        if isinstance(requirement, dict)
        else None
    )
    required_fragments = [
        normalize_option_text(fragment)
        for fragment in (
            public_evidence
            if isinstance(public_evidence, list)
            else re.split(
                r"[^0-9a-zA-Z\u4e00-\u9fff]+",
                str(required_value or ""),
            )
        )
        if len(normalize_option_text(fragment)) >= 2
        or re.fullmatch(r"[a-z0-9]", normalize_option_text(fragment))
    ]
    selected_values = [
        str(value) for value in selected_options.values() if str(value or "").strip()
    ]
    product_values = []
    for field in (
        "title",
        "Title",
        "name",
        "brand",
        "shop_name",
        "attribute",
        "attributes",
        "small_description",
        "BulletPoints",
        "full_description",
        "Description",
    ):
        value = product.get(field)
        if isinstance(value, dict):
            product_values.extend(str(item) for pair in value.items() for item in pair)
        elif isinstance(value, (list, tuple, set)):
            product_values.extend(str(item) for item in value)
        elif value:
            product_values.append(str(value))
    visible_values = [*selected_values, *product_values]

    def fragment_matches(fragment: str, value: str) -> bool:
        normalized_value = normalize_option_text(value)
        if re.fullmatch(r"[a-z0-9]", fragment):
            return re.search(
                rf"(?<![a-z0-9]){re.escape(fragment)}(?![a-z0-9])",
                normalized_value,
            ) is not None
        return semantic_text_match(fragment, value)

    def fragments_match(values: list[str]) -> bool:
        return bool(required_fragments) and all(
            any(fragment_matches(fragment, value) for value in values)
            for fragment in required_fragments
        )

    def selected_has_explicit_conflict() -> bool:
        required_text = " ".join(required_fragments)
        selected_text = " ".join(selected_values)
        quantity_pattern = re.compile(
            r"([一二三四五六七八九十百\d]+)\s*"
            r"(阶|罐|瓶|桶|支|包|盒|套|个|片|g|kg|ml|l|cm|mm|英寸|寸)",
            flags=re.IGNORECASE,
        )
        required_quantities = quantity_pattern.findall(required_text)
        selected_quantities = quantity_pattern.findall(selected_text)
        for required_number, required_unit in required_quantities:
            peers = {
                number
                for number, unit in selected_quantities
                if unit.casefold() == required_unit.casefold()
            }
            if peers and required_number not in peers:
                return True
        colors = {
            color
            for color in (
                "黑色", "白色", "红色", "橙色", "黄色", "绿色",
                "蓝色", "紫色", "粉色", "灰色", "银色", "金色",
            )
            if color in required_text
        }
        selected_colors = {
            color
            for color in (
                "黑色", "白色", "红色", "橙色", "黄色", "绿色",
                "蓝色", "紫色", "粉色", "灰色", "银色", "金色",
            )
            if color in selected_text
        }
        return bool(colors and selected_colors and colors.isdisjoint(selected_colors))

    selected_match = fragments_match(selected_values)
    visible_match = fragments_match(visible_values)
    if selected_match or (visible_match and not selected_has_explicit_conflict()):
        return comparison(
            PASS,
            comparator="public_query_option_visible_evidence_v3",
            required=required_value,
            actual=selected_options,
            source_field="instruction|selected_options|product_visible_fields",
            evidence={
                "axis": canonical_axis,
                "required_fragments": required_fragments,
                "visible_values": visible_values,
                "optional_fragments": (
                    requirement.get("optional_query_evidence") or []
                    if isinstance(requirement, dict)
                    else []
                ),
                "exact_target_comparison": exact_result,
            },
        )
    selected_measurements = _measurements(" ".join(selected_values))
    required_measurements = _measurements(required_value)
    if not selected_measurements or not required_measurements:
        return None
    public_constraints = _public_measurement_constraints(instruction)
    applicable = [
        constraint
        for constraint in public_constraints
        if any(
            _measurement_satisfies(measurement, constraint)
            for measurement in required_measurements
        )
    ]
    if not applicable:
        return None
    satisfied = all(
        any(
            _measurement_satisfies(measurement, constraint)
            for measurement in selected_measurements
        )
        for constraint in applicable
    )
    if not satisfied:
        return None
    return comparison(
        PASS,
        comparator="instruction_numeric_option_constraint_v1",
        required={
            "strict_target_option": required_value,
            "public_constraints": applicable,
        },
        actual=selected_options,
        source_field="instruction|selected_options",
        evidence={
            "axis": canonical_axis,
            "exact_target_comparison": exact_result,
            "selected_measurements": selected_measurements,
        },
    )


@dataclass(frozen=True)
class RewardResult:
    reward: float
    reward_type: str
    reward_valid: bool
    termination_reason: str
    target_asin_match: bool
    hard_gates: dict
    weighted_score: float
    evidence: dict

    def to_dict(self):
        payload = asdict(self)
        preference_scoring = self.evidence.get("preference_scoring") or {}
        dimensions = preference_scoring.get("dimensions") or {}
        constraint_results = self.evidence.get("query_constraint_results") or []
        constraint_audit_results = (
            self.evidence.get("query_constraint_audit_results")
            or constraint_results
        )
        constraint_status_counts = {
            status: sum(
                result.get("status") == status
                for result in constraint_results
            )
            for status in (PASS, FAIL, UNVERIFIABLE)
        }
        payload.update(
            {
                "reward_version": REWARD_VERSION,
                "query_constraint_version": QUERY_CONSTRAINT_VERSION,
                "constraint_semantics_version": CONSTRAINT_SEMANTICS_VERSION,
                "termination_subreason": self.evidence.get(
                    "termination_subreason"
                ),
                "terminal_utility": self.reward,
                "purchase_success": self.reward_type
                in {"gold_purchase", "valid_alternative_purchase"},
                "sampling_invalid": not self.reward_valid,
                "evidence_coverage": float(
                    preference_scoring.get("evidence_coverage", 0.0)
                ),
                "dimension_scores": {
                    name: float(result.get("score", 0.0))
                    for name, result in dimensions.items()
                },
                "constraint_results": constraint_results,
                "constraint_summary": {
                    "total": len(constraint_results),
                    "status_counts": constraint_status_counts,
                },
                "constraint_audit_results": constraint_audit_results,
                "constraint_audit_summary": {
                    "total": len(constraint_audit_results),
                    "enforcement_counts": {
                        enforcement: sum(
                            str(result.get("enforcement") or "scored")
                            == enforcement
                            for result in constraint_audit_results
                        )
                        for enforcement in ("scored", "audit_only")
                    },
                    "status_counts": {
                        status: sum(
                            result.get("status") == status
                            for result in constraint_audit_results
                        )
                        for status in (PASS, FAIL, UNVERIFIABLE)
                    },
                },
                "base_terminal_utility": float(
                    self.evidence.get("base_terminal_utility", self.reward)
                ),
                "step_count": int(self.evidence.get("step_count", 0)),
                "step_penalty": float(self.evidence.get("step_penalty", 0.0)),
                "step_penalty_version": self.evidence.get(
                    "step_penalty_version", STEP_PENALTY_VERSION
                ),
            }
        )
        return payload


def calculate_step_penalty(step_count: int) -> float:
    """Return the cumulative terminal penalty for executed tool steps."""
    steps = max(0, int(step_count))
    penalty = sum(
        max(0, min(steps, end) - start + 1) * rate
        for start, end, rate in STEP_PENALTY_SCHEDULE
    )
    return round(-penalty, 10)


def _apply_step_penalty(result: RewardResult, step_count: int) -> RewardResult:
    steps = max(0, int(step_count))
    # Infrastructure/reward-invalid samples are discarded and must keep a
    # neutral learning signal.
    penalty = calculate_step_penalty(steps) if result.reward_valid else 0.0
    evidence = {
        **result.evidence,
        "base_terminal_utility": float(result.reward),
        "step_count": steps,
        "step_penalty": penalty,
        "step_penalty_version": STEP_PENALTY_VERSION,
    }
    return replace(
        result,
        reward=round(float(result.reward) + penalty, 10),
        evidence=evidence,
    )


def _selected_unit_count(selected_options: object, unit: object) -> int | None:
    if not isinstance(selected_options, dict):
        return None
    text = " ".join(str(value) for value in selected_options.values() if value)
    normalized_unit = str(unit or "").strip()
    unit_aliases = {
        "平方米": ("平方米", "平米", "㎡"),
    }.get(normalized_unit, (normalized_unit,))
    aliases = "|".join(re.escape(alias) for alias in unit_aliases if alias)
    if aliases:
        direct = re.search(rf"(?<!\d)(\d{{1,5}})\s*(?:{aliases})(?![\w])", text)
        if direct and int(direct.group(1)) > 0:
            return int(direct.group(1))
    multiplied = re.findall(r"(?:\*|x|X|×)\s*(\d{1,5})(?!\d)", text)
    if multiplied:
        count = int(multiplied[-1])
        return count if count > 0 else None
    return None


def _price_resolution_for_basis(
    price_resolution: object,
    constraint: object,
    selected_options: object,
) -> object:
    if not isinstance(price_resolution, dict) or not isinstance(constraint, dict):
        return price_resolution
    basis = constraint.get("basis")
    if not isinstance(basis, dict) or basis.get("kind") != "per_unit":
        return price_resolution
    if price_resolution.get("status") != PASS:
        return price_resolution
    count = _selected_unit_count(selected_options, basis.get("unit"))
    if not count:
        return {
            **price_resolution,
            "evidence": {
                **(price_resolution.get("evidence") or {}),
                "price_basis": basis,
                "unit_count": None,
                "unit_price_assumed_from_variant": True,
            },
        }
    try:
        total_price = float(price_resolution["price"])
    except (KeyError, TypeError, ValueError):
        return price_resolution
    return {
        **price_resolution,
        "price": total_price / count,
        "method": "selected_variant_unit_price_v1",
        "evidence": {
            **(price_resolution.get("evidence") or {}),
            "price_basis": basis,
            "unit_count": count,
            "variant_total_price": total_price,
        },
    }


def _price_gate(
    price_resolution: object,
    goal: dict,
    selected_options: object = None,
) -> dict:
    constraint = goal.get("price_constraint")
    if not isinstance(constraint, dict):
        upper = goal.get("price_upper")
        constraint = (
            {"operator": "lte", "value": upper, "approximate": False}
            if upper is not None
            else None
        )
    if constraint is None:
        return comparison(
            PASS,
            comparator="budget_not_declared_v1",
            required=None,
            actual=(
                price_resolution.get("price")
                if isinstance(price_resolution, dict)
                else None
            ),
            source_field="instruction",
            evidence=price_resolution,
        )
    price_resolution = _price_resolution_for_basis(
        price_resolution,
        constraint,
        selected_options,
    )
    if (
        not isinstance(price_resolution, dict)
        or price_resolution.get("status") != PASS
    ):
        return comparison(
            UNVERIFIABLE,
            comparator="variant_price_budget_v1",
            required=constraint,
            actual=(
                price_resolution.get("price")
                if isinstance(price_resolution, dict)
                else None
            ),
            source_field="variant_price",
            evidence=price_resolution,
        )
    try:
        actual = float(price_resolution["price"])
        operator = str(constraint.get("operator") or "")
        if operator in {"lt", "lte", "gt", "gte"}:
            limit = float(constraint["value"])
            lower = upper = None
        elif operator in {"between", "approximately"}:
            lower = float(constraint["min"])
            upper = float(constraint["max"])
            limit = None
        else:
            raise ValueError(f"unsupported price operator: {operator}")
    except (KeyError, TypeError, ValueError):
        status = UNVERIFIABLE
        actual = price_resolution.get("price")
        required = constraint
    else:
        if (
            not math.isfinite(actual)
            or actual < 0
        ):
            status = UNVERIFIABLE
        elif operator in {"lt", "lte", "gt", "gte"} and (
            limit is None or not math.isfinite(limit) or limit <= 0
        ):
            status = UNVERIFIABLE
        elif operator in {"between", "approximately"} and (
            lower is None
            or upper is None
            or not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower < 0
            or upper < lower
        ):
            status = UNVERIFIABLE
        else:
            passed = {
                "lt": lambda: actual < limit,
                "lte": lambda: actual <= limit,
                "gt": lambda: actual > limit,
                "gte": lambda: actual >= limit,
                "between": lambda: lower <= actual <= upper,
                "approximately": lambda: lower <= actual <= upper,
            }[operator]()
            status = PASS if passed else FAIL
        required = constraint
    return comparison(
        status,
        comparator="variant_price_constraint_v2",
        required=required,
        actual=actual,
        source_field="variant_price",
        evidence=price_resolution,
    )


def _brand_gate(expected: str, product: dict) -> dict:
    aliases = load_brand_aliases()
    canonical = aliases.get(normalize_text(expected), normalize_text(expected))
    expected_aliases = {
        alias
        for alias, candidate_canonical in aliases.items()
        if candidate_canonical == canonical and len(alias) >= 2
    }
    expected_aliases.add(normalize_text(expected))
    structured_text = normalize_text(
        " ".join(
            str(value)
            for value in (product.get("brand"), product.get("shop_name"))
            if value
        )
    )
    title_text = normalize_text(product.get("title") or product.get("name"))
    raw_attributes = product.get("attribute") or product.get("attributes") or []
    if not isinstance(raw_attributes, list):
        raw_attributes = [raw_attributes]
    attribute_values = [normalize_text(value) for value in raw_attributes if value]
    compatibility_prefixes = ("适用", "适配", "兼容", "用于", "可用", "替代")

    def safe_title_match(alias: str) -> bool:
        start = title_text.find(alias)
        while start >= 0:
            prefix = title_text[max(0, start - 4) : start]
            if not any(prefix.endswith(marker) for marker in compatibility_prefixes):
                return True
            start = title_text.find(alias, start + 1)
        return False

    matched_sources = []
    for alias in sorted(expected_aliases, key=len, reverse=True):
        if alias and alias in structured_text:
            matched_sources.append({"source": "brand|shop_name", "alias": alias})
            break
    for alias in sorted(expected_aliases, key=len, reverse=True):
        if alias and safe_title_match(alias):
            matched_sources.append({"source": "title|name", "alias": alias})
            break
    for alias in sorted(expected_aliases, key=len, reverse=True):
        if alias and any(value == alias for value in attribute_values):
            matched_sources.append({"source": "attribute", "alias": alias})
            break
    matched = bool(matched_sources)
    return comparison(
        PASS if matched else FAIL,
        comparator="explicit_brand_alias_evidence_v2",
        required=expected,
        actual={
            "brand_or_shop": structured_text,
            "title_or_name": title_text,
            "attributes": attribute_values,
        },
        source_field="brand|shop_name|title|attribute",
        evidence={"alias_match": matched, "matched_sources": matched_sources},
    )


def _score_requirements(requirements, evaluator) -> dict:
    results = [evaluator(requirement) for requirement in requirements]
    total = len(results)
    passed = sum(result["status"] == PASS for result in results)
    verifiable = sum(
        result["status"] != UNVERIFIABLE for result in results
    )
    return {
        "active": total > 0,
        "score": passed / total if total else 0.0,
        "coverage": verifiable / total if total else 0.0,
        "required_count": total,
        "passed_count": passed,
        "verifiable_count": verifiable,
        "results": results,
    }


def _option_dimension(
    product: dict,
    goal: dict,
    selected_options: object,
) -> dict:
    required = goal.get("required_options_by_key") or {}
    unresolved = goal.get("unresolved_option_requirements") or []
    contract_options = [
        constraint
        for constraint in (
            (goal.get("query_constraint_contract") or {}).get("constraints") or []
        )
        if isinstance(constraint, dict)
        and constraint.get("constraint_type") == "option"
    ]
    strict_axes = {
        str(constraint.get("axis"))
        for constraint in contract_options
        if constraint.get("role") == "strict_query_variant"
        and constraint.get("axis") is not None
    }
    strict_unresolved_values = {
        normalize_option_text(constraint.get("expected"))
        for constraint in contract_options
        if constraint.get("role") == "strict_query_variant"
        and constraint.get("axis") is None
    }
    strict_required = dict(required)
    strict_unresolved = list(unresolved)
    if contract_options:
        strict_required = {
            axis: requirement
            for axis, requirement in required.items()
            if str(axis) in strict_axes
        }
        strict_unresolved = [
            item
            for item in unresolved
            if normalize_option_text(item.get("value")) in strict_unresolved_values
        ]
    exact_results = [
        compare_required_options(
            product,
            {axis: requirement},
            selected_options,
        )
        for axis, requirement in required.items()
    ]
    strict_exact_results = [
        compare_required_options(
            product,
            {axis: requirement},
            selected_options,
        )
        for axis, requirement in strict_required.items()
    ]
    results = []
    for (axis, requirement), exact_result in zip(
        strict_required.items(), strict_exact_results, strict=True
    ):
        semantic_result = _public_option_semantic_match(
            product,
            axis,
            requirement,
            selected_options,
            goal.get("instruction_text"),
            exact_result,
        )
        results.append(semantic_result or exact_result)
    selected_values = {
        normalize_option_text(value)
        for value in (
            selected_options.values()
            if isinstance(selected_options, dict)
            else ()
        )
    }
    unresolved_exact_results = [
        comparison(
            (
                PASS
                if normalize_option_text(item.get("value"))
                in selected_values
                else FAIL
            ),
            comparator="exact_option_value_fallback_v1",
            required=item.get("value"),
            actual=list(
                selected_options.values()
                if isinstance(selected_options, dict)
                else ()
            ),
            source_field="instruction_options",
            evidence=item,
        )
        for item in strict_unresolved
    ]
    results.extend(unresolved_exact_results)
    all_exact_results = [*exact_results]
    all_strict_results = [*strict_exact_results, *unresolved_exact_results]
    total = len(results)
    passed = sum(result["status"] == PASS for result in results)
    verifiable = sum(
        result["status"] != UNVERIFIABLE for result in results
    )
    return {
        "active": total > 0,
        "score": passed / total if total else 0.0,
        "coverage": verifiable / total if total else 0.0,
        "required_count": total,
        "passed_count": passed,
        "verifiable_count": verifiable,
        "exact_target_satisfied": all(
            result["status"] == PASS for result in all_exact_results
        ),
        "strict_exact_target_satisfied": all(
            result["status"] == PASS for result in all_strict_results
        ),
        "strict_query_satisfied": all(
            result["status"] == PASS for result in all_strict_results
        ),
        "results": results,
        "exact_target_results": all_exact_results,
    }


def _preference_dimensions(
    product: dict,
    goal: dict,
    selected_options: object,
) -> dict:
    contract_constraints = (
        (goal.get("query_constraint_contract") or {}).get("constraints") or []
    )
    strict_core_values = {
        normalize_text(constraint.get("expected"))
        for constraint in contract_constraints
        if isinstance(constraint, dict)
        and constraint.get("constraint_type") == "core_function"
        and constraint.get("role") == "matching_dimension"
    }
    core_functions = goal.get("expected_core_functions") or []
    if contract_constraints:
        core_functions = [
            value
            for value in core_functions
            if normalize_text(value) in strict_core_values
        ]
    dimensions = {
        "brand": _score_requirements(
            goal.get("expected_brand") or [],
            lambda value: _brand_gate(value, product),
        ),
        "model": _score_requirements(
            goal.get("expected_model") or [],
            lambda value: compare_model(value, product),
        ),
        "core_functions": _score_requirements(
            core_functions,
            lambda value: compare_core_functions([value], product),
        ),
        "key_options": _option_dimension(
            product,
            goal,
            selected_options,
        ),
    }
    active_weight = sum(
        DIMENSION_WEIGHTS[name]
        for name, result in dimensions.items()
        if result["active"]
    )
    if active_weight:
        match_score = sum(
            DIMENSION_WEIGHTS[name] * result["score"]
            for name, result in dimensions.items()
            if result["active"]
        ) / active_weight
        evidence_coverage = sum(
            DIMENSION_WEIGHTS[name] * result["coverage"]
            for name, result in dimensions.items()
            if result["active"]
        ) / active_weight
    else:
        match_score = 1.0
        evidence_coverage = 1.0
    all_satisfied = (
        math.isclose(match_score, 1.0, abs_tol=1.0e-8)
        and math.isclose(evidence_coverage, 1.0, abs_tol=1.0e-8)
    )
    return {
        "weights": DIMENSION_WEIGHTS,
        "active_weight": active_weight,
        "match_score": match_score,
        "evidence_coverage": evidence_coverage,
        "all_satisfied": all_satisfied,
        "exact_option_target_satisfied": dimensions["key_options"].get(
            "exact_target_satisfied", True
        ),
        "strict_query_option_satisfied": dimensions["key_options"].get(
            "strict_query_satisfied", True
        ),
        "strict_exact_option_target_satisfied": dimensions["key_options"].get(
            "strict_exact_target_satisfied", True
        ),
        "dimensions": dimensions,
    }


def _constraint_strength(constraint: dict) -> str:
    strength = str(constraint.get("strength") or "")
    if strength in {"hard", "soft", "ignore", "needs_review"}:
        return strength
    if constraint.get("role") in {
        "gold_variant_reference",
        "gold_annotation_reference",
        "query_preference_reference",
    }:
        return "ignore"
    return "hard"


def _option_constraint_comparison(
    constraint: dict,
    *,
    product: dict,
    goal: dict,
    selected_options: object,
) -> dict:
    axis = constraint.get("axis")
    expected = constraint.get("expected")
    if axis is None:
        selected_values = {
            normalize_option_text(value)
            for value in (
                selected_options.values()
                if isinstance(selected_options, dict)
                else ()
            )
        }
        return comparison(
            PASS if normalize_option_text(expected) in selected_values else FAIL,
            comparator="exact_option_value_fallback_v2",
            required=expected,
            actual=list(
                selected_options.values()
                if isinstance(selected_options, dict)
                else ()
            ),
            source_field="instruction_options",
            evidence={"axis": None},
        )
    requirement = dict(
        (goal.get("required_options_by_key") or {}).get(str(axis)) or {}
    )
    requirement["value"] = expected
    exact_result = compare_required_options(
        product,
        {str(axis): requirement},
        selected_options,
    )
    semantic_result = _public_option_semantic_match(
        product,
        str(axis),
        requirement,
        selected_options,
        goal.get("instruction_text"),
        exact_result,
    )
    return semantic_result or exact_result


def _query_constraint_results(
    *,
    hard_gates: dict,
    price_comparison: dict,
    preferences: dict,
    goal: dict,
    product: dict,
    selected_options: object,
) -> list[dict]:
    """Join frozen Query/Gold provenance with deterministic comparisons."""

    contract = goal.get("query_constraint_contract") or {}
    constraints = contract.get("constraints") or []
    rows = []
    for raw_constraint in constraints:
        if not isinstance(raw_constraint, dict):
            continue
        constraint = dict(raw_constraint)
        constraint_type = constraint.get("constraint_type")
        strength = _constraint_strength(constraint)
        if strength == "ignore":
            result = comparison(
                PASS,
                comparator="query_constraint_not_scored_v2",
                required=constraint.get("expected"),
                actual=None,
                source_field=str(constraint.get("source") or "unknown"),
                evidence={"scored": False, "strength": strength},
            )
        elif strength == "needs_review":
            result = comparison(
                UNVERIFIABLE,
                comparator="query_constraint_needs_review_v1",
                required=constraint.get("expected"),
                actual=None,
                source_field=str(constraint.get("source") or "unknown"),
                evidence={"scored": False, "strength": strength},
            )
        elif constraint_type == "category":
            result = hard_gates.get("category")
        elif constraint_type in {"budget_upper", "price_lower", "price_range"}:
            result = price_comparison
        elif constraint_type == "brand":
            result = _brand_gate(str(constraint.get("expected") or ""), product)
        elif constraint_type == "model":
            result = compare_model(constraint.get("expected"), product)
        elif constraint_type == "core_function":
            result = compare_core_functions(
                [constraint.get("expected")],
                product,
            )
        elif constraint_type == "option":
            result = _option_constraint_comparison(
                constraint,
                product=product,
                goal=goal,
                selected_options=selected_options,
            )
        else:
            result = None
        result = result if isinstance(result, dict) else comparison(
            UNVERIFIABLE,
            comparator="missing_constraint_comparison_v1",
            required=constraint.get("expected"),
            actual=None,
            source_field=str(constraint.get("source") or "unknown"),
            evidence={},
        )
        rows.append(
            {
                **constraint,
                "status": result.get("status", UNVERIFIABLE),
                "comparator": result.get("comparator"),
                "actual": result.get("actual"),
                "source_field": result.get("source_field"),
                "evidence": result.get("evidence") or {},
            }
        )
    return rows


def _purchase_constraints(results: object, strength: str) -> list[dict]:
    """Return scored constraints with the requested frozen semantic strength."""
    selected = []
    for raw_result in results if isinstance(results, list) else []:
        if not isinstance(raw_result, dict):
            continue
        # Contracts created before ``enforcement`` existed retain their
        # original scoring behavior.  New audit-only semantic clauses remain
        # visible in evidence but cannot invalidate or downgrade Reward.
        if str(raw_result.get("enforcement") or "scored") != "scored":
            continue
        if _constraint_strength(raw_result) == strength:
            selected.append(raw_result)
    return selected


def _evaluate(
    product: dict,
    goal: dict,
    *,
    selected_options: object,
    price_resolution: object,
) -> tuple[dict, dict, dict]:
    hard_gates = {
        "category": compare_category(goal.get("category"), product),
    }
    price_comparison = _price_gate(price_resolution, goal, selected_options)
    preferences = _preference_dimensions(
        product,
        goal,
        selected_options,
    )
    return hard_gates, preferences, price_comparison


def _explicit_price_resolution(price: object) -> dict:
    try:
        value = float(price)
    except (TypeError, ValueError):
        value = None
    return {
        "status": (
            PASS
            if value is not None
            and math.isfinite(value)
            and value >= 0
            else UNVERIFIABLE
        ),
        "price": value,
        "version": VARIANT_PRICE_VERSION,
        "method": "explicit_test_price",
        "evidence": {},
    }


def instruction_contract_integrity(goal: object) -> dict:
    """Verify that Reward features were compiled from this exact task instruction."""
    task_goal = goal if isinstance(goal, dict) else {}
    instruction_text = str(task_goal.get("instruction_text") or "")
    expected = hashlib.sha256(instruction_text.encode("utf-8")).hexdigest()
    feature_hash = str(task_goal.get("instruction_sha256") or "")
    contract = task_goal.get("query_constraint_contract") or {}
    contract_hash = str(
        contract.get("instruction_sha256")
        if isinstance(contract, dict)
        else ""
    )
    valid = bool(instruction_text and feature_hash and contract_hash) and (
        feature_hash == expected == contract_hash
    )
    return {
        "valid": valid,
        "instruction_sha256": expected,
        "feature_instruction_sha256": feature_hash,
        "contract_instruction_sha256": contract_hash,
    }


def evaluate_purchase(
    product: dict,
    goal: dict,
    *,
    selected_options: object,
    price_resolution: dict | None = None,
    price: object = None,
    rewards: dict[str, float] | None = None,
    step_count: int = 0,
) -> RewardResult:
    """Reward purchases only after every frozen hard constraint passes.

    An exact target ASIN is classified as ``gold_purchase``. A different ASIN
    is ``valid_alternative_purchase`` only when every verifiable soft
    preference passes; otherwise it is ``partial_alternative_purchase`` and
    receives continuous soft-preference credit. Ambiguous hard semantics
    invalidate the sample instead of being silently weakened.
    """
    integrity = instruction_contract_integrity(goal)
    values = {**DEFAULT_REWARDS, **(rewards or {})}
    if price_resolution is None or (
        price is not None
        and (
            not isinstance(price_resolution, dict)
            or price_resolution.get("status") != PASS
        )
    ):
        price_resolution = (
            _explicit_price_resolution(price)
            if price is not None
            else resolve_variant_price(product, selected_options)
        )
    hard_gates, preferences, price_comparison = _evaluate(
        product,
        goal,
        selected_options=selected_options,
        price_resolution=price_resolution,
    )
    all_constraint_results = _query_constraint_results(
        hard_gates=hard_gates,
        price_comparison=price_comparison,
        preferences=preferences,
        goal=goal,
        product=product,
        selected_options=selected_options,
    )
    hard_constraint_results = _purchase_constraints(
        all_constraint_results,
        "hard",
    )
    soft_constraint_results = _purchase_constraints(
        all_constraint_results,
        "soft",
    )
    review_constraint_results = _purchase_constraints(
        all_constraint_results,
        "needs_review",
    )
    constraint_results = [
        *hard_constraint_results,
        *soft_constraint_results,
        *review_constraint_results,
    ]
    audit_only_constraint_results = [
        result
        for result in all_constraint_results
        if str(result.get("enforcement") or "scored") == "audit_only"
    ]
    asin_match = str(product.get("asin")) == str(goal.get("asin"))
    semantic_hard_gates = {}
    for result in hard_constraint_results:
        key = (
            "category"
            if result.get("constraint_type") == "category"
            else str(result.get("constraint_id") or result.get("constraint_type"))
        )
        semantic_hard_gates[key] = result
    hard_unverifiable = any(
        result.get("status") == UNVERIFIABLE
        for result in hard_constraint_results
    )
    hard_failed = any(
        result.get("status") == FAIL for result in hard_constraint_results
    )
    review_required = bool(review_constraint_results)
    hard_passed = sum(
        result.get("status") == PASS for result in hard_constraint_results
    )
    hard_total = len(hard_constraint_results)
    hard_all_satisfied = hard_total > 0 and hard_passed == hard_total
    soft_verifiable_results = [
        result
        for result in soft_constraint_results
        if result.get("status") in {PASS, FAIL}
    ]
    soft_passed = sum(
        result.get("status") == PASS for result in soft_verifiable_results
    )
    soft_failed = sum(
        result.get("status") == FAIL for result in soft_verifiable_results
    )
    soft_total = len(soft_constraint_results)
    soft_verifiable_total = len(soft_verifiable_results)
    soft_unverifiable = soft_total - soft_verifiable_total
    soft_score = (
        soft_passed / soft_verifiable_total
        if soft_verifiable_total
        else 1.0
    )
    if hard_total:
        strict_match_score = hard_passed / hard_total
    else:
        # Compatibility for hand-built tests and legacy one-off goals that do
        # not carry the frozen query contract. Canonical tasks always have a
        # non-empty contract compiled from the visible instruction.
        fallback_statuses = [
            *(gate.get("status") for gate in hard_gates.values()),
            price_comparison.get("status"),
        ]
        fallback_passed = sum(status == PASS for status in fallback_statuses)
        strict_match_score = (
            fallback_passed / len(fallback_statuses)
            * float(preferences["match_score"])
        )
        hard_all_satisfied = (
            fallback_passed == len(fallback_statuses)
            and bool(preferences["all_satisfied"])
        )
    # A confirmed hard-constraint violation is already sufficient to classify
    # the purchase as wrong.  Additional unverifiable hard constraints must not
    # mask that known failure as reward_unverifiable.
    if hard_failed:
        reward_type = "wrong_purchase"
        reward_valid = True
        reward = values[reward_type]
    elif review_required or hard_unverifiable:
        reward_type = "reward_unverifiable"
        reward_valid = False
        reward = values[reward_type]
    elif hard_all_satisfied:
        reward_valid = True
        if asin_match:
            reward_type = "gold_purchase"
            reward = values[reward_type]
        elif soft_failed:
            reward_type = "partial_alternative_purchase"
            reward = (
                values["partial_purchase_base"]
                + values["partial_purchase_scale"] * soft_score
            )
        else:
            reward_type = "valid_alternative_purchase"
            reward = values[reward_type]
    else:
        reward_type = "wrong_purchase"
        reward_valid = True
        reward = values[reward_type]
    return _apply_step_penalty(
        RewardResult(
            reward=float(reward),
            reward_type=reward_type,
            reward_valid=reward_valid,
            termination_reason=reward_type,
            target_asin_match=asin_match,
            hard_gates=semantic_hard_gates or hard_gates,
            weighted_score=float(strict_match_score),
            evidence={
                "reward_feature_version": goal.get("reward_feature_version"),
                "expected_reward_feature_version": REWARD_FEATURE_VERSION,
                "instruction_contract_integrity": integrity,
                "comparator_version": COMPARATOR_VERSION,
                "variant_price_version": VARIANT_PRICE_VERSION,
                "price_resolution": price_resolution,
                "price_comparison": price_comparison,
                "preference_scoring": preferences,
                "strict_purchase_contract": {
                    "target_asin_required_for_gold": True,
                    "fully_satisfying_alternative_allowed": True,
                    "target_asin_match": asin_match,
                    "all_hard_constraints_satisfied": hard_all_satisfied,
                    "hard_passed": hard_passed,
                    "hard_total": hard_total,
                    "soft_passed": soft_passed,
                    "soft_failed": soft_failed,
                    "soft_total": soft_total,
                    "soft_verifiable_total": soft_verifiable_total,
                    "soft_unverifiable": soft_unverifiable,
                    "soft_score": soft_score,
                    "needs_review": len(review_constraint_results),
                    "audit_only": len(audit_only_constraint_results),
                    "audit_only_needs_review": sum(
                        _constraint_strength(result) == "needs_review"
                        for result in audit_only_constraint_results
                    ),
                    "constraint_semantics_version": (
                        CONSTRAINT_SEMANTICS_VERSION
                    ),
                },
                "query_constraint_results": constraint_results,
                "query_constraint_audit_results": all_constraint_results,
                "exact_target_variant_match": bool(
                    asin_match and preferences["exact_option_target_satisfied"]
                ),
            },
        ),
        step_count,
    )


def evaluate_candidate_eligibility(product: dict, goal: dict) -> dict:
    """Determine whether an opened candidate is sufficiently acceptable to block abstention."""
    integrity = instruction_contract_integrity(goal)
    selected, option_resolution = candidate_options_for_evaluation(
        product,
        goal.get("required_options_by_key"),
    )
    price_resolution = resolve_variant_price(product, selected)
    hard_gates, preferences, price_comparison = _evaluate(
        product,
        goal,
        selected_options=selected,
        price_resolution=price_resolution,
    )
    hard_pass = all(
        gate["status"] == PASS for gate in hard_gates.values()
    )
    known_acceptable = (
        hard_pass
        and preferences["match_score"]
        >= KNOWN_ACCEPTABLE_MATCH_THRESHOLD
        and preferences["evidence_coverage"]
        >= KNOWN_ACCEPTABLE_COVERAGE_THRESHOLD
    )
    return {
        "status": PASS if known_acceptable else FAIL,
        "known_acceptable": known_acceptable,
        # Session-state field consumed by Environment v2.4 termination logic.
        "known_valid": known_acceptable,
        "selected_options": selected,
        "option_resolution": option_resolution,
        "price_resolution": price_resolution,
        "hard_gates": hard_gates,
        "price_comparison": price_comparison,
        "match_score": preferences["match_score"],
        "evidence_coverage": preferences["evidence_coverage"],
        "instruction_contract_integrity": integrity,
    }


def evaluate_abstain(
    *,
    effective_result_sets: int,
    opened_candidates: int,
    known_acceptable_candidates: int | None = None,
    known_valid_candidates: int | None = None,
    rewards: dict[str, float] | None = None,
    step_count: int = 0,
) -> RewardResult:
    values = {**DEFAULT_REWARDS, **(rewards or {})}
    known_count = int(
        known_acceptable_candidates
        if known_acceptable_candidates is not None
        else known_valid_candidates or 0
    )
    reward_type = "early_abstain"
    return _apply_step_penalty(
        RewardResult(
            reward=float(values[reward_type]),
            reward_type=reward_type,
            reward_valid=True,
            termination_reason=reward_type,
            target_asin_match=False,
            hard_gates={},
            weighted_score=0.0,
            evidence={
                "effective_result_sets": int(effective_result_sets),
                "opened_candidates": int(opened_candidates),
                "known_acceptable_candidates": known_count,
            },
        ),
        step_count,
    )


def fixed_termination(
    reason: str,
    *,
    subreason: str | None = None,
    rewards: dict[str, float] | None = None,
    step_count: int = 0,
) -> RewardResult:
    values = {**DEFAULT_REWARDS, **(rewards or {})}
    if reason not in {"assistant_final", "repeat_loop", "max_steps"}:
        raise ValueError(f"unsupported fixed termination reason: {reason}")
    allowed_subreasons = {
        "assistant_final": {"assistant_final"},
        "repeat_loop": {"exact_action_repeat", "no_progress_loop"},
        "max_steps": {"max_steps"},
    }
    if subreason is None:
        subreason = {
            "assistant_final": "assistant_final",
            "repeat_loop": "no_progress_loop",
            "max_steps": "max_steps",
        }[reason]
    if subreason not in allowed_subreasons[reason]:
        raise ValueError(
            f"unsupported termination subreason {subreason!r} for {reason!r}"
        )
    return _apply_step_penalty(
        RewardResult(
            reward=float(values[reason]),
            reward_type=reason,
            reward_valid=True,
            termination_reason=reason,
            target_asin_match=False,
            hard_gates={},
            weighted_score=0.0,
            evidence={"termination_subreason": subreason},
        ),
        step_count,
    )
