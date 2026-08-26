"""Derive query-auditable Reward v4 features for one shopping task."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from shopping_grpo.price_semantics import reward_price_constraint
from web_agent_site.engine.comparators import (
    load_brand_aliases,
    load_semantic_aliases,
    normalize_text,
)

REWARD_FEATURE_VERSION = "shopping-reward-features-v2"
QUERY_CONSTRAINT_VERSION = "shopping-query-constraints-v1"
CONSTRAINT_SEMANTICS_VERSION = "shopping-constraint-semantics-v2"
OPTION_AXIS_VERSION = "option-axis-v1"
_ANNOTATION_REPAIR_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "task_annotation_repairs.json"
)
_AXIS_ALIASES = {
    "color": {"颜色", "颜色分类"},
    "size": {"尺码", "鞋码"},
    "dimensions": {"尺寸", "大小"},
    "net_content": {"净含量", "总净含量"},
    "flavor": {"口味", "食品口味"},
    "specification": {"规格", "规格描述", "规格类型"},
    "bundle": {"套餐", "套餐类型", "组合套餐"},
    "capacity": {"容量", "规格容量"},
}
_MODEL_TOKEN = re.compile(
    r"(?<![a-z0-9])(?=[a-z0-9._+-]{2,24}(?![a-z0-9]))"
    r"(?=[a-z0-9._+-]*\d)[a-z0-9._+-]+",
    flags=re.IGNORECASE,
)
_SHOP_SUFFIXES = (
    "官方旗舰店",
    "旗舰店",
    "专卖店",
    "专营店",
    "企业店",
    "店",
)
_QUERY_CLAUSE_SPLIT = re.compile(r"[\n\r，。；！？,;!?]+")
_OPTIONAL_ANYWHERE = re.compile(
    r"无所谓|不要求|没要求|无要求|不限|不限制|不限定|"
    r"任意|随意|随便|不挑|可有可无"
)
_OPTIONAL_BEFORE_EVIDENCE = re.compile(
    r"也可以|也行|可以是|可选|最好(?:是|用|选|要|能|有)?|"
    r"优先(?:是|用|选|要)?|尽量(?:是|用|选|要)?"
)
_OPTIONAL_AFTER_EVIDENCE = re.compile(r"也可以|也行|都行")
_HARD_REQUIRE = re.compile(
    r"必须|务必|一定(?:要|得|需)|非.{0,8}不可|只(?:要|能|可)|仅限"
)
_HARD_BOUND = re.compile(
    r"不超过|不要超过|别超过|不能超过|不得超过|不可超过|不高于|"
    r"至少|不低于|不少于|以内|以下|以上|之间|控制在"
)
_HARD_FORBID = re.compile(
    r"绝对(?:不要|不能|不许)|坚决不要|严禁|禁止|"
    r"不得|不可|不能有|必须(?:没有|不含|不带|不能)|"
    r"一定(?:不要|不能|不需要)"
)
_PLAIN_FORBID = re.compile(r"不要(?!太)|(?<!特)别(?:要|让)(?!太)")
_AMBIGUOUS_NEGATION = re.compile(r"不需要|无需|不用")
_SOFT_PREFERENCE = re.compile(
    r"最好|优先|尽量|希望|偏好|倾向|比较喜欢|更喜欢|"
    r"比较合适|越.{0,8}越好|可以的话|预期|预计|预估|左右|上下|大概|大约|"
    r"约莫|约摸|差不多|大致|将近|接近|附近|不要太|别太"
    r"|约(?=[零一二两三四五六七八九十百千万\d])"
)
_SOFT_QUALIFIED_BOUNDARY = re.compile(
    r"(?:最好|优先|尽量|希望|倾向|预期|预计|预估|大概|大约|约莫|约摸|差不多|大致)"
    r".{0,16}(?:不超过|不要超过|别超过|不能超过|不高于|至少|不低于|"
    r"不少于|以内|以下|以上|之间|控制在)|"
    r"(?:不超过|不要超过|别超过|不能超过|不高于|至少|不低于|"
    r"不少于|以内|以下|以上|之间|控制在).{0,10}"
    r"(?:比较合适|最好|大概|大约|左右|上下)"
)
_INDIFFERENCE = re.compile(
    r"无所谓|不要求|没要求|无要求|不限|不限制|不限定|"
    r"任意|随意|随便|不挑|可有可无|都行|均可|没有需求|没需求"
)
_CONCESSION = re.compile(r"也可以|也行|都可以|没有.{0,8}也可以")
_SEMANTIC_CLAUSE_MARKER = re.compile(
    r"绝对|必须|务必|一定|不得|不可|不能|不要|不需要|无需|不用|"
    r"无所谓|不要求|不限|不限制|可有可无|最好|优先|尽量|希望|"
    r"偏好|倾向|预期|预计|预估|左右|上下|大概|大约|约莫|约摸|差不多|大致|"
    r"将近|接近|附近|不超过|不高于|至少|不低于|以内|以下|"
    r"以上|之间|控制在|别太|别超过|别让|越.{0,8}越好|"
    r"约(?=[零一二两三四五六七八九十百千万\d])"
)
_SEMANTIC_SCOPE_MARKER = re.compile(
    r"绝对(?:不要|不能|不许)|坚决不要|严禁|禁止|"
    r"一定(?:不要|不能|不需要|要|得|需)|必须|务必|"
    r"不要超过|别超过|不能超过|不得超过|不可超过|不超过|不高于|"
    r"至少|不低于|不少于|控制在|不得|不可|不能有|不要(?!太)|"
    r"不需要|无需|不用|"
    r"无所谓|不要求|没要求|无要求|不限|不限制|不限定|"
    r"任意|随意|随便|不挑|可有可无|也可以|也行|都可以|都行|均可|"
    r"最好|优先|尽量|希望|偏好|倾向|预期|预计|预估|"
    r"大概|大约|约莫|约摸|差不多|大致|将近|接近|附近|"
    r"不要太|别太|比较合适|"
    r"约(?=[零一二两三四五六七八九十百千万\d])"
)
_SEMANTIC_SCOPE_LINKING_TEXT = re.compile(
    r"^(?:的|地|得|要|需|需要|能|能够|可以|可|是|为|在|有|带|"
    r"用|使用|选|选择|采用|控制|保持|落在|做成|设为|"
    r"并|且|并且|而且|同时|还|也|又|或|或者|以及|但|但是|不过|"
    r"价格|价位|预算|金额|总价|单价|定金|规格|尺寸|大小|"
    r"容量|重量|长度|宽度|高度|厚度|功率|电压|数量|颜色|材质|"
    r"款式|型号|品牌|功能|属性|范围|区间|整体|目前|我的)*$"
)
_AUDIT_TOOL_FREE_REQUIREMENT = re.compile(
    r"(?:不用|无需|不必).{0,6}工具.{0,12}(?:也能|就能|即可|可以|可)"
)
_AUDIT_NO_DAMAGE_REQUIREMENT = re.compile(
    r"(?:无需|不用|不必|不得|不可).{0,8}(?:破坏|破线|剪线|改线)"
)
_AUDIT_EXPLICITLY_NOT_REQUIRED = re.compile(
    r"不需要(?:有|带有|具备)|不用参与|不用减脂"
)
_AUDIT_PLAIN_REJECTION = re.compile(
    r"不用(?:花里胡哨|复杂|多余|额外)"
)


def normalize_option_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("/", "|")
    return re.sub(r"\s+", "", text)


def canonicalize_option_axis(value: object) -> str:
    normalized = normalize_option_text(value)
    for canonical, aliases in _AXIS_ALIASES.items():
        if normalized in {
            normalize_option_text(alias) for alias in aliases
        }:
            return canonical
    return normalized


def _clean_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


@lru_cache(maxsize=1)
def _task_annotation_repairs() -> dict[str, dict]:
    if not _ANNOTATION_REPAIR_PATH.is_file():
        return {}
    payload = json.loads(_ANNOTATION_REPAIR_PATH.read_text(encoding="utf-8"))
    if payload.get("version") != "task-annotation-repairs-v1":
        raise ValueError("task annotation repair table has the wrong version")
    repairs = payload.get("repairs") or {}
    return repairs if isinstance(repairs, dict) else {}


def apply_task_annotation_repair(instruction: dict) -> dict:
    instruction_text = str(instruction.get("instruction") or "")
    instruction_hash = hashlib.sha256(instruction_text.encode("utf-8")).hexdigest()
    repair = _task_annotation_repairs().get(instruction_hash)
    if not isinstance(repair, dict):
        return instruction
    repaired = dict(instruction)
    for field in ("attributes", "instruction_options"):
        if field in repair:
            repaired[field] = list(repair.get(field) or [])
    if "query_verified_instruction_options" in repair:
        repaired["_query_verified_instruction_options"] = list(
            repair.get("query_verified_instruction_options") or []
        )
    return repaired


def _target_option_axes(target_product: dict) -> dict[str, list[str]]:
    axes = {}
    for raw_axis, entries in (
        target_product.get("customization_options") or {}
    ).items():
        values = []
        for entry in entries or []:
            if (
                isinstance(entry, dict)
                and normalize_option_text(entry.get("value"))
            ):
                values.append(str(entry["value"]))
        axes[str(raw_axis)] = values
    return axes


def _resolve_required_options(
    option_values: list[str],
    target_product: dict,
) -> tuple[dict, list[dict]]:
    axes = _target_option_axes(target_product)
    resolved = {}
    unresolved = []
    for required_value in option_values:
        normalized_required = normalize_option_text(required_value)
        matches = [
            raw_axis
            for raw_axis, values in axes.items()
            if normalized_required
            in {normalize_option_text(value) for value in values}
        ]
        if len(matches) != 1:
            unresolved.append(
                {
                    "value": required_value,
                    "reason": (
                        "axis_not_found"
                        if not matches
                        else "axis_ambiguous"
                    ),
                    "axes": matches,
                }
            )
            continue
        raw_axis = matches[0]
        canonical_axis = canonicalize_option_axis(raw_axis)
        if canonical_axis in resolved:
            unresolved.append(
                {
                    "value": required_value,
                    "reason": "canonical_axis_collision",
                    "axes": [raw_axis],
                }
            )
            continue
        resolved[canonical_axis] = {
            "value": required_value,
            "source_axis": raw_axis,
            "source": "instruction.instruction_options",
        }
    return resolved, unresolved


def _explicit_brand(instruction: str, target_product: dict) -> list[str]:
    instruction_text = normalize_text(instruction)
    target_text = normalize_text(
        " ".join(
            str(value)
            for value in (
                target_product.get("title"),
                target_product.get("shop_name"),
            )
            if value
        )
    )
    aliases = load_brand_aliases()
    matches = {
        canonical
        for alias, canonical in aliases.items()
        if len(alias) >= 2
        and alias in instruction_text
        and alias in target_text
    }
    shop_name = normalize_text(target_product.get("shop_name"))
    for suffix in _SHOP_SUFFIXES:
        normalized_suffix = normalize_text(suffix)
        if shop_name.endswith(normalized_suffix):
            shop_name = shop_name[: -len(normalized_suffix)]
            break
    title = normalize_text(target_product.get("title"))
    for length in range(min(len(shop_name), 12), 1, -1):
        prefix = shop_name[:length]
        if prefix in instruction_text and prefix in title:
            matches.add(prefix)
            break
    return sorted(matches)


def _explicit_models(instruction: str, target_product: dict) -> list[str]:
    instruction_tokens = {
        token.casefold() for token in _MODEL_TOKEN.findall(instruction)
    }
    target_text = " ".join(
        str(value)
        for value in (
            target_product.get("title"),
            target_product.get("full_description"),
        )
        if value
    )
    target_tokens = {
        token.casefold() for token in _MODEL_TOKEN.findall(target_text)
    }
    return sorted(instruction_tokens.intersection(target_tokens))


def _raw_query_evidence(instruction: str, value: object) -> list[str]:
    """Return deterministic public-query spans supporting a constraint value."""

    raw_value = str(value or "").strip()
    if not raw_value:
        return []
    normalized_instruction = normalize_text(instruction)
    normalized_value = normalize_text(raw_value)
    if normalized_value and normalized_value in normalized_instruction:
        return [raw_value]
    fragments = [
        fragment
        for fragment in re.split(r"[^0-9a-zA-Z\u4e00-\u9fff]+", raw_value)
        if len(normalize_text(fragment)) >= 2
        and normalize_text(fragment) in normalized_instruction
    ]
    aliases = load_semantic_aliases()
    expected_canonical = aliases.get(normalized_value)
    if expected_canonical:
        fragments.extend(
            alias
            for alias, canonical in aliases.items()
            if canonical == expected_canonical
            and len(alias) >= 2
            and alias in normalized_instruction
        )
    mixed_ascii_token = bool(
        re.search(r"[a-z]", normalized_value)
        and re.search(r"\d", normalized_value)
    )
    measurement_token = bool(
        re.fullmatch(
            r"\d+(?:\.\d+)?(?:mm|cm|m|ml|l|g|kg)",
            normalized_value,
        )
    )
    if mixed_ascii_token and not measurement_token:
        return list(dict.fromkeys(fragment for fragment in fragments if fragment))
    blocks = SequenceMatcher(
        None,
        normalized_value,
        normalized_instruction,
        autojunk=False,
    ).get_matching_blocks()
    meaningful_blocks = [
        normalized_value[block.a : block.a + block.size]
        for block in blocks
        if block.size >= 2
    ]
    covered = sum(len(block) for block in meaningful_blocks)
    if normalized_value and covered / len(normalized_value) >= 0.60:
        fragments.extend(meaningful_blocks)
    return list(dict.fromkeys(fragment for fragment in fragments if fragment))


def _semantic_scope_payload(text: str) -> str:
    """Return non-marker text used to decide whether a new scope begins."""

    normalized = normalize_text(text)
    return _SEMANTIC_SCOPE_MARKER.sub("", normalized)


def _split_semantic_clause(clause: str) -> list[str]:
    """Split one punctuation clause only at independent semantic scopes.

    Adjacent qualifiers such as ``尽量别超过`` and
    ``预期价格不超过`` remain together.  A later marker starts a new span
    once the preceding scope already contains an actual requirement value,
    which prevents ``绝对不要5升以上最好要黑色`` from making both values
    hard.
    """

    text = str(clause or "").strip()
    if not text:
        return []
    markers = list(_SEMANTIC_SCOPE_MARKER.finditer(text))
    if len(markers) < 2:
        return [text]
    split_points = []
    segment_start = 0
    for marker in markers[1:]:
        payload = _semantic_scope_payload(text[segment_start : marker.start()])
        if _SEMANTIC_SCOPE_LINKING_TEXT.fullmatch(payload):
            continue
        split_points.append(marker.start())
        segment_start = marker.start()
    if not split_points:
        return [text]
    segments = []
    start = 0
    for end in [*split_points, len(text)]:
        segment = text[start:end].strip(" \t、和及并且但也")
        if segment:
            segments.append(segment)
        start = end
    return segments or [text]


def _query_semantic_segments(instruction: str) -> list[str]:
    segments = []
    for raw_clause in _QUERY_CLAUSE_SPLIT.split(str(instruction or "")):
        segments.extend(_split_semantic_clause(raw_clause))
    return list(dict.fromkeys(segment for segment in segments if segment))


def _evidence_is_optional_in_clause(clause: str, evidence: object) -> bool:
    """Recognize preference/concession wording without weakening capabilities.

    ``可以电磁炉用`` and ``都可以食用`` describe required capabilities, so a
    bare ``可以`` is never enough to make evidence optional.  We only downgrade
    explicit indifference, soft-preference markers, or ``也可以`` concessions
    that govern this evidence in the same clause.
    """

    normalized_clause = normalize_text(clause)
    normalized_evidence = normalize_text(evidence)
    if not normalized_clause or not normalized_evidence:
        return False
    evidence_index = normalized_clause.find(normalized_evidence)
    if evidence_index < 0:
        return False
    if _OPTIONAL_ANYWHERE.search(normalized_clause):
        return True
    if _AUDIT_EXPLICITLY_NOT_REQUIRED.search(normalized_clause):
        return True
    prefix = normalized_clause[:evidence_index]
    suffix = normalized_clause[evidence_index + len(normalized_evidence) :]
    if _OPTIONAL_BEFORE_EVIDENCE.search(prefix[-16:]):
        return True
    if _OPTIONAL_AFTER_EVIDENCE.search(suffix[:16]):
        return True
    return False


def _evidence_is_background_context(clause: str, evidence: object) -> bool:
    """Exclude narrative setup that does not describe a product requirement."""

    normalized_clause = normalize_text(clause)
    normalized_evidence = normalize_text(evidence)
    if not normalized_clause or not normalized_evidence:
        return False
    evidence_index = normalized_clause.find(normalized_evidence)
    if evidence_index < 0:
        return False
    prefix = normalized_clause[:evidence_index]
    return bool(
        re.search(r"(?:准备|打算|计划|正要|将要|快要)去.{0,8}$", prefix)
    )


def _partition_query_evidence(
    instruction: str,
    value: object,
) -> tuple[list[str], list[str]]:
    """Split public evidence into required and explicitly optional spans."""

    clauses = _query_semantic_segments(instruction)
    required = []
    optional = []
    for evidence in _raw_query_evidence(instruction, value):
        matching_clauses = [
            clause
            for clause in clauses
            if normalize_text(evidence) in normalize_text(clause)
        ]
        if matching_clauses and all(
            _evidence_is_background_context(clause, evidence)
            for clause in matching_clauses
        ):
            continue
        if matching_clauses and all(
            _evidence_is_optional_in_clause(clause, evidence)
            or _evidence_is_background_context(clause, evidence)
            for clause in matching_clauses
        ):
            optional.append(evidence)
        else:
            required.append(evidence)
    return (
        list(dict.fromkeys(required)),
        list(dict.fromkeys(optional)),
    )


def _query_evidence(instruction: str, value: object) -> list[str]:
    """Return only evidence that is allowed to affect purchase success."""

    required, _ = _partition_query_evidence(instruction, value)
    return required


def _optional_query_evidence(instruction: str, value: object) -> list[str]:
    """Return visible preference/concession evidence retained only for audit."""

    _, optional = _partition_query_evidence(instruction, value)
    return optional


def _matching_query_clauses(
    instruction: str,
    evidence_values: object,
) -> list[str]:
    evidence = [
        normalize_text(value)
        for value in (
            evidence_values
            if isinstance(evidence_values, list)
            else [evidence_values]
        )
        if normalize_text(value)
    ]
    if not evidence:
        return []
    matches = []
    for clause in _query_semantic_segments(instruction):
        normalized_clause = normalize_text(clause)
        if clause and any(value in normalized_clause for value in evidence):
            matches.append(clause)
    return list(dict.fromkeys(matches))


def _constraint_semantics(
    *,
    instruction_text: str,
    role: str,
    constraint_type: str,
    expected: object = None,
    query_evidence: list[str] | None = None,
    optional_query_evidence: list[str] | None = None,
    price_constraint: dict | None = None,
) -> dict:
    """Classify only high-confidence public language into hard/soft semantics.

    Ambiguous negative wording is deliberately sent to ``needs_review``.  It
    must not silently become either a hard prohibition or an ignorable
    preference, because phrases such as ``不需要售后`` and ``不需要内搭短裤``
    have different shopping intent despite sharing the same surface marker.
    """

    required_evidence = list(query_evidence or [])
    optional_evidence = list(optional_query_evidence or [])
    evidence = [*required_evidence, *optional_evidence]
    if constraint_type in {"budget_upper", "price_lower", "price_range"}:
        source_text = str((price_constraint or {}).get("source_text") or "")
        if source_text:
            evidence = [source_text]
    clauses = _matching_query_clauses(instruction_text, evidence)
    quote = "；".join(clauses)
    normalized_quote = normalize_text(quote)
    every_clause_is_soft = bool(clauses) and all(
        _SOFT_PREFERENCE.search(normalize_text(clause))
        for clause in clauses
    )

    def result(strength: str, polarity: str, reason: str) -> dict:
        return {
            "strength": strength,
            "polarity": polarity,
            "enforcement": (
                "audit_only"
                if constraint_type == "query_clause"
                or strength in {"ignore", "needs_review"}
                else "scored"
            ),
            "query_quote": quote,
            "semantics_reason": reason,
            "semantics_version": CONSTRAINT_SEMANTICS_VERSION,
        }

    if role in {"gold_variant_reference", "gold_annotation_reference"}:
        return result("ignore", "indifferent", "gold_only_reference")
    if constraint_type == "category":
        return result("hard", "require", "category_contract")
    if not quote and role == "query_preference_reference":
        return result("ignore", "indifferent", "ungrounded_reference")
    if _HARD_FORBID.search(normalized_quote):
        return result("hard", "forbid", "explicit_hard_forbid_marker")
    if _HARD_REQUIRE.search(normalized_quote):
        return result("hard", "require", "explicit_hard_require_marker")
    if every_clause_is_soft and re.search(
        r"不要太|别太", normalized_quote
    ):
        return result("soft", "prefer", "soft_negative_preference")
    if _SOFT_QUALIFIED_BOUNDARY.search(normalized_quote):
        return result("soft", "prefer", "soft_qualified_boundary")
    if _PLAIN_FORBID.search(normalized_quote):
        return result("hard", "forbid", "explicit_forbid_marker")
    if _INDIFFERENCE.search(normalized_quote) or _CONCESSION.search(normalized_quote):
        return result("ignore", "indifferent", "explicit_indifference")
    if _AMBIGUOUS_NEGATION.search(normalized_quote):
        if constraint_type == "query_clause":
            if _AUDIT_TOOL_FREE_REQUIREMENT.search(normalized_quote):
                return result(
                    "hard",
                    "require",
                    "tool_free_capability_requirement",
                )
            if _AUDIT_NO_DAMAGE_REQUIREMENT.search(normalized_quote):
                return result(
                    "hard",
                    "forbid",
                    "no_damage_installation_requirement",
                )
            if _AUDIT_EXPLICITLY_NOT_REQUIRED.search(normalized_quote):
                return result(
                    "ignore",
                    "indifferent",
                    "explicitly_not_required",
                )
            if _AUDIT_PLAIN_REJECTION.search(normalized_quote):
                return result(
                    "hard",
                    "forbid",
                    "plain_negative_product_preference",
                )
        normalized_expected = normalize_text(expected)
        if constraint_type != "query_clause" and re.search(
            r"无|免|不|不可|未", normalized_expected
        ):
            return result(
                "hard",
                "forbid",
                "negative_query_supported_by_target_evidence",
            )
        return result("needs_review", "unknown", "ambiguous_negation")
    if _HARD_BOUND.search(normalized_quote):
        return result("hard", "require", "explicit_hard_boundary")
    if every_clause_is_soft:
        return result("soft", "prefer", "explicit_soft_preference")
    if constraint_type in {"budget_upper", "price_lower", "price_range"}:
        if bool((price_constraint or {}).get("approximate")):
            return result("soft", "prefer", "approximate_price")
        return result("hard", "require", "explicit_price_boundary")
    if optional_evidence and not required_evidence:
        return result("soft", "prefer", "optional_query_language")
    if constraint_type == "query_clause":
        return result("needs_review", "unknown", "unclassified_semantic_clause")
    if required_evidence:
        return result("hard", "require", "explicit_query_requirement")
    return result("ignore", "indifferent", "no_public_query_grounding")


def _query_constraint_contract(
    *,
    instruction_text: str,
    product: dict,
    expected_brand: list[str],
    expected_model: list[str],
    core_functions: list[str],
    optional_brand: list[str],
    optional_model: list[str],
    optional_core_functions: list[str],
    required_options: dict,
    unresolved_options: list[dict],
) -> dict:
    """Freeze the source and public-query support of every Reward input.

    The Gold product may define the strict target category and variant, but it
    must not silently turn all of its private fields into user requirements.
    Query support is therefore recorded explicitly for later Reward/Rubric
    audits without exposing private task facts to the Actor.
    """

    category = str(product.get("category") or "")
    category_leaf = next(
        (
            part.strip()
            for part in reversed(re.split(r"[›>/]", category))
            if part.strip()
        ),
        category,
    )
    constraints = [
        {
            "constraint_type": "category",
            "role": "hard_gate",
            "expected": category,
            "source": "task.category_contract",
            "query_evidence": _query_evidence(
                instruction_text,
                category_leaf,
            ),
        }
    ]
    price_constraint = reward_price_constraint(instruction_text)
    if price_constraint is not None:
        operator = price_constraint.get("operator")
        constraint_type = {
            "lt": "budget_upper",
            "lte": "budget_upper",
            "gt": "price_lower",
            "gte": "price_lower",
            "between": "price_range",
            "approximately": "price_range",
        }.get(operator, "price_range")
        constraints.append(
            {
                "constraint_type": constraint_type,
                "role": "required_constraint",
                "expected": price_constraint,
                "source": (
                    "query.explicit_budget"
                    if constraint_type == "budget_upper"
                    else "query.explicit_price"
                ),
                "query_evidence": [price_constraint.get("source_text") or instruction_text],
            }
        )
    for constraint_type, values, source in (
        ("brand", expected_brand, "query.explicit_brand"),
        ("model", expected_model, "query.explicit_model"),
        ("core_function", core_functions, "instruction.annotation"),
    ):
        for value in values:
            constraints.append(
                {
                    "constraint_type": constraint_type,
                    "role": "matching_dimension",
                    "expected": value,
                    "source": source,
                    "query_evidence": _query_evidence(
                        instruction_text,
                        value,
                    ),
                }
            )
    for constraint_type, values, source in (
        ("brand", optional_brand, "query.optional_brand"),
        ("model", optional_model, "query.optional_model"),
        (
            "core_function",
            optional_core_functions,
            "instruction.optional_annotation",
        ),
    ):
        for value in values:
            constraints.append(
                {
                    "constraint_type": constraint_type,
                    "role": "query_preference_reference",
                    "expected": value,
                    "source": source,
                    "query_evidence": [],
                    "optional_query_evidence": _optional_query_evidence(
                        instruction_text,
                        value,
                    ),
                }
            )
    for canonical_axis, requirement in required_options.items():
        value = (
            requirement.get("value")
            if isinstance(requirement, dict)
            else requirement
        )
        query_evidence, optional_query_evidence = _partition_query_evidence(
            instruction_text,
            value,
        )
        constraints.append(
            {
                "constraint_type": "option",
                "role": (
                    "strict_query_variant"
                    if query_evidence
                    else "query_preference_reference"
                    if optional_query_evidence
                    else "gold_variant_reference"
                ),
                "axis": canonical_axis,
                "expected": value,
                "source": "instruction.instruction_options",
                "query_evidence": query_evidence,
                "optional_query_evidence": optional_query_evidence,
            }
        )
    for requirement in unresolved_options:
        value = requirement.get("value")
        query_evidence, optional_query_evidence = _partition_query_evidence(
            instruction_text,
            value,
        )
        constraints.append(
            {
                "constraint_type": "option",
                "role": (
                    "strict_query_variant"
                    if query_evidence
                    else "query_preference_reference"
                    if optional_query_evidence
                    else "gold_variant_reference"
                ),
                "axis": None,
                "expected": value,
                "source": "instruction.instruction_options",
                "query_evidence": query_evidence,
                "optional_query_evidence": optional_query_evidence,
            }
        )
    semantic_constraints = []
    for constraint in constraints:
        semantic_constraints.append(
            {
                **constraint,
                **_constraint_semantics(
                    instruction_text=instruction_text,
                    role=str(constraint.get("role") or ""),
                    constraint_type=str(
                        constraint.get("constraint_type") or ""
                    ),
                    expected=constraint.get("expected"),
                    query_evidence=constraint.get("query_evidence") or [],
                    optional_query_evidence=constraint.get(
                        "optional_query_evidence"
                    )
                    or [],
                    price_constraint=(
                        constraint.get("expected")
                        if constraint.get("constraint_type")
                        in {"budget_upper", "price_lower", "price_range"}
                        and isinstance(constraint.get("expected"), dict)
                        else None
                    ),
                ),
            }
        )

    # Preserve every visibly qualified Query clause even when the current
    # deterministic comparators cannot safely score it.  These audit-only
    # rows keep requirements such as authenticity, negation, approximate
    # dimensions, and contextual boundaries available to Rubric/Judge
    # evaluation without turning missing product evidence into false Reward
    # failures.
    for clause in _query_semantic_segments(instruction_text):
        normalized_clause = normalize_text(clause)
        if not clause or not _SEMANTIC_CLAUSE_MARKER.search(normalized_clause):
            continue
        if any(
            normalized_clause
            and normalized_clause
            in normalize_text(constraint.get("query_quote"))
            for constraint in semantic_constraints
        ):
            continue
        audit_constraint = {
            "constraint_type": "query_clause",
            "role": "query_semantic_audit",
            "expected": clause,
            "source": "query.semantic_clause",
            "query_evidence": [clause],
        }
        semantic_constraints.append(
            {
                **audit_constraint,
                **_constraint_semantics(
                    instruction_text=instruction_text,
                    role="query_semantic_audit",
                    constraint_type="query_clause",
                    expected=clause,
                    query_evidence=[clause],
                ),
            }
        )

    return {
        "schema_version": QUERY_CONSTRAINT_VERSION,
        "semantics_version": CONSTRAINT_SEMANTICS_VERSION,
        "instruction_sha256": hashlib.sha256(
            instruction_text.encode("utf-8")
        ).hexdigest(),
        "constraints": [
            {
                "constraint_id": f"q{index:04d}",
                **constraint,
            }
            for index, constraint in enumerate(semantic_constraints, start=1)
        ],
    }


def compile_reward_features(
    instruction_record: object,
    target_product: object,
) -> dict:
    """Build scoring inputs while preserving their Query/Gold provenance."""
    instruction = (
        instruction_record if isinstance(instruction_record, dict) else {}
    )
    instruction = apply_task_annotation_repair(instruction)
    product = target_product if isinstance(target_product, dict) else {}
    instruction_text = str(instruction.get("instruction") or "")
    option_values = _clean_list(instruction.get("instruction_options"))
    required_options, unresolved_options = _resolve_required_options(
        option_values,
        product,
    )
    for requirement in required_options.values():
        value = requirement.get("value")
        required_evidence, optional_evidence = _partition_query_evidence(
            instruction_text,
            value,
        )
        requirement["query_evidence"] = required_evidence
        requirement["optional_query_evidence"] = optional_evidence
    for requirement in unresolved_options:
        value = requirement.get("value")
        required_evidence, optional_evidence = _partition_query_evidence(
            instruction_text,
            value,
        )
        requirement["query_evidence"] = required_evidence
        requirement["optional_query_evidence"] = optional_evidence
    annotated_brand = _explicit_brand(instruction_text, product)
    expected_brand = [
        value for value in annotated_brand if _query_evidence(instruction_text, value)
    ]
    optional_brand = [
        value
        for value in annotated_brand
        if not _query_evidence(instruction_text, value)
        and _optional_query_evidence(instruction_text, value)
    ]
    annotated_model = _explicit_models(instruction_text, product)
    expected_model = [
        value for value in annotated_model if _query_evidence(instruction_text, value)
    ]
    optional_model = [
        value
        for value in annotated_model
        if not _query_evidence(instruction_text, value)
        and _optional_query_evidence(instruction_text, value)
    ]
    annotated_core_functions = _clean_list(instruction.get("attributes"))
    core_functions = [
        value
        for value in annotated_core_functions
        if _query_evidence(instruction_text, value)
    ]
    optional_core_functions = [
        value
        for value in annotated_core_functions
        if not _query_evidence(instruction_text, value)
        and _optional_query_evidence(instruction_text, value)
    ]
    price_constraint = reward_price_constraint(instruction_text)
    instruction_sha256 = hashlib.sha256(
        instruction_text.encode("utf-8")
    ).hexdigest()
    return {
        "reward_feature_version": REWARD_FEATURE_VERSION,
        "instruction_sha256": instruction_sha256,
        "category": product.get("category"),
        "expected_brand": expected_brand,
        "expected_model": expected_model,
        "expected_core_functions": core_functions,
        "optional_brand_preferences": optional_brand,
        "optional_model_preferences": optional_model,
        "optional_core_function_preferences": optional_core_functions,
        "required_options_by_key": required_options,
        "unresolved_option_requirements": unresolved_options,
        "option_axis_version": OPTION_AXIS_VERSION,
        "price_constraint": price_constraint,
        "price_upper": (
            float(price_constraint["value"])
            if isinstance(price_constraint, dict)
            and price_constraint.get("operator") in {"lt", "lte"}
            else float(price_constraint["max"])
            if isinstance(price_constraint, dict)
            and price_constraint.get("operator") in {"between", "approximately"}
            else None
        ),
        "query_constraint_contract": _query_constraint_contract(
            instruction_text=instruction_text,
            product=product,
            expected_brand=expected_brand,
            expected_model=expected_model,
            core_functions=core_functions,
            optional_brand=optional_brand,
            optional_model=optional_model,
            optional_core_functions=optional_core_functions,
            required_options=required_options,
            unresolved_options=unresolved_options,
        ),
        "feature_sources": {
            "category": "task.target_product.category",
            "brand": "instruction_explicit_alias",
            "model": "instruction_target_token_intersection",
            "core_functions": "instruction.query_supported_attributes",
            "options": "instruction.instruction_options",
        },
    }
