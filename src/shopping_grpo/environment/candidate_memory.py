"""Trajectory-local memory of previously observed public product evidence.

The memory never reads a task goal, Reward output, hidden candidate judgment, or
catalog data outside the structured observation already shown to the model.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from shopping_grpo.environment.product_id import is_product_id


CANDIDATE_MEMORY_VERSION_V1 = "shopping-candidate-memory-v1"
CANDIDATE_MEMORY_VERSION = "shopping-candidate-memory-v2"
CANDIDATE_MEMORY_START_V1 = "[CANDIDATE_MEMORY_V1]"
CANDIDATE_MEMORY_END_V1 = "[/CANDIDATE_MEMORY_V1]"
CANDIDATE_MEMORY_START = "[CANDIDATE_MEMORY_V2]"
CANDIDATE_MEMORY_END = "[/CANDIDATE_MEMORY_V2]"
CANDIDATE_CONVERGENCE_NOTICE_PREFIX = "候选记忆提示:"
DEFAULT_MAX_ENTRIES = 6
MAX_ENTRIES_LIMIT = 6
MAX_RENDERED_ENTRY_CHARS = 156
FOOTER_MARKER = "\n\n搜索功能是否可用:"
FOOTER_LINE_MARKER = "\n搜索功能是否可用:"


def new_candidate_memory(
    max_entries: int = DEFAULT_MAX_ENTRIES,
    *,
    stable_candidate_ids: bool = True,
) -> dict:
    """Create a JSON-serializable memory owned by exactly one trajectory."""
    max_entries = int(max_entries)
    if not 1 <= max_entries <= MAX_ENTRIES_LIMIT:
        raise ValueError("candidate memory max_entries must be between 1 and 6")
    return {
        "version": (
            CANDIDATE_MEMORY_VERSION
            if stable_candidate_ids
            else CANDIDATE_MEMORY_VERSION_V1
        ),
        "stable_candidate_ids": bool(stable_candidate_ids),
        "max_entries": max_entries,
        "entries": [],
        "last_search": {},
        "updates": 0,
        "search_updates": 0,
        "evictions": 0,
    }


def update_candidate_memory(
    memory: dict,
    observation_state: Mapping,
    *,
    step_count: int,
) -> bool:
    """Track public search locations and product-detail evidence."""
    _validate_memory(memory)
    if not isinstance(observation_state, Mapping):
        return False
    if observation_state.get("observation_version") != "shopping-observation-v2":
        return False
    page_type = observation_state.get("page_type")
    if page_type == "search_results":
        _remember_search_context(memory, observation_state)
        return False
    if page_type != "product_detail":
        return False
    product = observation_state.get("product")
    if not isinstance(product, Mapping):
        return False
    asin = _clean_text(product.get("asin"), 16)
    if not is_product_id(asin):
        return False

    entries = memory["entries"]
    existing_index = next(
        (index for index, entry in enumerate(entries) if entry.get("asin") == asin),
        None,
    )
    stable_candidate_ids = bool(memory.get("stable_candidate_ids"))
    existing = entries[existing_index] if existing_index is not None else None
    if existing_index is not None and not stable_candidate_ids:
        existing = entries.pop(existing_index)
        existing_index = None
    if (
        stable_candidate_ids
        and existing_index is None
        and len(entries) >= int(memory["max_entries"])
    ):
        # Evaluation memory is a fixed, append-only shortlist.  Once full, later
        # products remain visible on their current detail page but do not evict or
        # overwrite the stable candidates already exposed to the model.
        return False
    step_count = max(0, int(step_count))
    price = observation_state.get("selected_price")
    if price is None:
        price = product.get("price")
    source = _source_location(memory, asin)
    if not source and existing is not None:
        source = {
            key: existing.get(key)
            for key in (
                "source_query",
                "source_page",
                "source_rank",
                "source_position",
            )
            if existing.get(key) not in (None, "")
        }
    entry = {}
    if stable_candidate_ids:
        entry["candidate_id"] = (
            str(existing.get("candidate_id")).strip().upper()
            if existing is not None
            and re.fullmatch(
                r"C[1-6]",
                str(existing.get("candidate_id") or "").strip().upper(),
            )
            else _allocate_candidate_id(memory)
        )
    entry.update(
        {
            "asin": asin,
            "title": _clean_text(product.get("title"), 64),
            "brand": _clean_text(product.get("brand"), 32),
            "category": _clean_text(product.get("category"), 48),
            "price": _clean_text(price, 32),
            "selected_options": _selected_options(
                observation_state.get("selected_options")
            ),
            "evidence": _public_evidence(product),
            "first_seen_step": (
                int(existing.get("first_seen_step", step_count))
                if existing is not None
                else step_count
            ),
            "last_seen_step": step_count,
            "observations": (
                int(existing.get("observations", 0)) + 1
                if existing is not None
                else 1
            ),
            **source,
        }
    )
    if not stable_candidate_ids:
        entries.append(entry)
        while len(entries) > int(memory["max_entries"]):
            entries.pop(0)
            memory["evictions"] += 1
    elif existing_index is None:
        entries.append(entry)
    else:
        entries[existing_index] = entry
    memory["updates"] += 1
    return True


def render_candidate_memory(
    memory: Mapping,
    *,
    current_asin: str | None = None,
) -> str:
    """Render the selected evaluation or legacy training memory contract."""
    if not isinstance(memory, Mapping):
        return ""
    entries = memory.get("entries")
    if not isinstance(entries, list):
        return ""
    current_asin = str(current_asin or "").strip()
    stable_candidate_ids = bool(memory.get("stable_candidate_ids"))
    current_entry = next(
        (
            entry
            for entry in entries
            if isinstance(entry, Mapping) and entry.get("asin") == current_asin
        ),
        None,
    )
    visible_entries = [
        entry
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("asin") != current_asin
    ]
    if stable_candidate_ids:
        visible_entries = sorted(
            visible_entries,
            key=lambda entry: _candidate_id_number(entry.get("candidate_id")),
        )
    else:
        visible_entries = list(reversed(visible_entries))
    if not visible_entries and current_entry is None:
        return ""

    if stable_candidate_ids:
        max_entries = int(memory.get("max_entries", DEFAULT_MAX_ENTRIES))
        candidate_range = f"C1-C{max_entries}"
        lines = [
            CANDIDATE_MEMORY_START,
        ]
        if len(entries) >= max_entries:
            lines.append(
                f"{CANDIDATE_CONVERGENCE_NOTICE_PREFIX} 已保存 {len(entries)} 个候选，"
                f"达到 {candidate_range} 的记忆容量上限。后续商品仍可正常搜索和核验，"
                "但不会写入或替换本候选记忆；需要保留其信息时请依据当前详情与对话历史判断。"
            )
        lines.extend([
            f"已核验候选：{candidate_range} 是候选在当前轨迹内的稳定编号；不代表优劣、满足情况或推荐。",
            "格式：候选ID｜ASIN｜位置(@检索词/P页/R排名)｜价格｜品牌｜品类｜已选规格｜标题｜公开证据",
            "用途：用于跨页面比较、避免重复核验，并辅助后续搜索与当前商品决策。",
        ])
        if current_entry is not None:
            lines.append(
                "当前详情候选："
                f"{_clean_text(current_entry.get('candidate_id'), 3)}｜"
                f"ASIN {_clean_text(current_entry.get('asin'), 16)}；"
                "完整信息见当前商品详情正文，本块不重复占用 Token。"
            )
    else:
        lines = [
            CANDIDATE_MEMORY_START_V1,
            "已查看候选：A-F 按最近访问排序；不代表优劣、满足情况或推荐。",
            "格式：代号｜ASIN｜位置(@检索词/P页/R排名)｜价格｜品牌｜品类｜已选规格｜标题｜公开证据",
            "回访：ASIN 已在当前页则直接 open_product；否则按位置重新搜索/翻页。",
        ]
    for index, entry in enumerate(visible_entries, start=1):
        label = (
            _clean_text(entry.get("candidate_id"), 3)
            if stable_candidate_ids
            else chr(ord("A") + index - 1)
        )
        price = _clean_text(entry.get("price"), 16) or "-"
        brand = _clean_text(entry.get("brand"), 10) or "-"
        category = _clean_text(entry.get("category"), 10) or "-"
        source = _render_source(entry)
        selected = entry.get("selected_options")
        option_text = "-"
        if isinstance(selected, Mapping) and selected:
            option_text = ",".join(
                f"{_clean_text(axis, 24)}={_clean_text(label, 48)}"
                for axis, label in sorted(selected.items())
                if _clean_text(axis, 24) and _clean_text(label, 48)
            )
            option_text = _truncate(option_text, 18) if option_text else "-"
        title = _clean_text(entry.get("title"), 18) or "-"
        evidence = entry.get("evidence")
        evidence_text = "-"
        if isinstance(evidence, list) and evidence:
            evidence_text = _truncate(
                ",".join(_clean_text(item, 16) for item in evidence),
                24,
            )
        parts = [
            label,
            _clean_text(entry.get("asin"), 16),
            source,
            price,
            brand,
            category,
            option_text,
            title,
            evidence_text,
        ]
        lines.append(_truncate("｜".join(parts), MAX_RENDERED_ENTRY_CHARS))
    lines.append(
        CANDIDATE_MEMORY_END if stable_candidate_ids else CANDIDATE_MEMORY_END_V1
    )
    return "\n".join(lines)


def _allocate_candidate_id(memory: Mapping) -> str:
    used = {
        str(entry.get("candidate_id") or "").strip().upper()
        for entry in memory.get("entries", [])
        if isinstance(entry, Mapping)
    }
    for index in range(1, int(memory.get("max_entries", DEFAULT_MAX_ENTRIES)) + 1):
        candidate_id = f"C{index}"
        if candidate_id not in used:
            return candidate_id
    oldest = min(
        (
            entry
            for entry in memory.get("entries", [])
            if isinstance(entry, Mapping)
        ),
        key=lambda entry: (
            int(entry.get("last_seen_step", -1)),
            _candidate_id_number(entry.get("candidate_id")),
        ),
    )
    return str(oldest.get("candidate_id"))


def _candidate_id_number(value: object) -> int:
    match = re.fullmatch(r"C([1-6])", str(value or "").strip().upper())
    return int(match.group(1)) if match else 99


def _remember_search_context(memory: dict, observation_state: Mapping) -> None:
    products = observation_state.get("products")
    if not isinstance(products, list):
        return
    rows = []
    for position, product in enumerate(products, start=1):
        if not isinstance(product, Mapping):
            continue
        asin = _clean_text(product.get("asin"), 16)
        if not is_product_id(asin):
            continue
        try:
            rank = int(product.get("rank", position))
        except (TypeError, ValueError):
            rank = position
        rows.append(
            {
                "asin": asin,
                "rank": max(1, rank),
                "position": position,
            }
        )
    if not rows:
        return
    try:
        page = max(1, int(observation_state.get("page", 1)))
    except (TypeError, ValueError):
        page = 1
    memory["last_search"] = {
        "query": _clean_text(observation_state.get("query"), 64),
        "page": page,
        "products": rows,
    }
    memory["search_updates"] += 1


def _source_location(memory: Mapping, asin: str) -> dict:
    search = memory.get("last_search")
    if not isinstance(search, Mapping):
        return {}
    products = search.get("products")
    if not isinstance(products, list):
        return {}
    matched = next(
        (
            row
            for row in products
            if isinstance(row, Mapping) and row.get("asin") == asin
        ),
        None,
    )
    if matched is None:
        return {}
    return {
        "source_query": _clean_text(search.get("query"), 64),
        "source_page": max(1, int(search.get("page", 1))),
        "source_rank": max(1, int(matched.get("rank", 1))),
        "source_position": max(1, int(matched.get("position", 1))),
    }


def _render_source(entry: Mapping) -> str:
    query = _clean_text(entry.get("source_query"), 24)
    page = entry.get("source_page")
    rank = entry.get("source_rank")
    if not query or page in (None, "") or rank in (None, ""):
        return "@-"
    return f"@{query}/P{int(page)}/R{int(rank)}"


def attach_candidate_memory(observation: str, memory_block: str) -> str:
    """Insert a memory block before the action footer, keeping the footer last."""
    base, _ = detach_candidate_memory(observation)
    memory_block = str(memory_block or "").strip()
    if not memory_block:
        return base
    index = base.rfind(FOOTER_MARKER)
    if index < 0:
        index = base.rfind(FOOTER_LINE_MARKER)
    if index < 0:
        return base.rstrip() + "\n\n" + memory_block
    return base[:index].rstrip() + "\n\n" + memory_block + base[index:]


def detach_candidate_memory(observation: str) -> tuple[str, str]:
    """Return the observation without memory and the exact embedded memory block."""
    observation = str(observation)
    markers = (
        (CANDIDATE_MEMORY_START, CANDIDATE_MEMORY_END),
        (CANDIDATE_MEMORY_START_V1, CANDIDATE_MEMORY_END_V1),
    )
    matched = next(
        (
            (observation.find(start_marker), start_marker, end_marker)
            for start_marker, end_marker in markers
            if observation.find(start_marker) >= 0
        ),
        None,
    )
    if matched is None:
        return observation, ""
    start, _, end_marker = matched
    end = observation.find(end_marker, start)
    if end < 0:
        return observation, ""
    end += len(end_marker)
    block = observation[start:end]
    before = observation[:start].rstrip()
    after = observation[end:].lstrip("\r\n")
    separator = "\n\n" if before and after else ""
    return before + separator + after, block


def _selected_options(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    selected = {}
    for raw_axis, raw_record in value.items():
        axis = _clean_text(raw_axis, 24)
        if isinstance(raw_record, Mapping):
            label = _clean_text(raw_record.get("label"), 48)
        else:
            label = _clean_text(raw_record, 48)
        if axis and label:
            selected[axis] = label
    return selected


def _public_evidence(product: Mapping) -> list[str]:
    values = []
    seen = set()
    for field in ("key_attributes", "features", "attributes"):
        raw_values = product.get(field)
        if not isinstance(raw_values, (list, tuple)):
            continue
        for raw_value in raw_values:
            value = _clean_text(raw_value, 40)
            identity = value.casefold()
            if not value or identity in seen:
                continue
            seen.add(identity)
            values.append(value)
            if len(values) >= 10:
                return values
    return values


def _clean_text(value: object, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    for unsafe in (
        CANDIDATE_MEMORY_START,
        CANDIDATE_MEMORY_END,
        CANDIDATE_MEMORY_START_V1,
        CANDIDATE_MEMORY_END_V1,
        "搜索功能是否可用:",
        "可点击的按钮:",
    ):
        safe = unsafe.replace(":", "：").replace("[", "(").replace("]", ")")
        text = text.replace(unsafe, safe)
    return _truncate(text, limit)


def _truncate(text: str, limit: int) -> str:
    text = str(text)
    limit = max(1, int(limit))
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"


def _validate_memory(memory: object) -> None:
    if not isinstance(memory, dict):
        raise TypeError("candidate memory must be a mutable object")
    if memory.get("version") not in {
        CANDIDATE_MEMORY_VERSION,
        CANDIDATE_MEMORY_VERSION_V1,
    }:
        raise ValueError("unsupported candidate memory version")
    if not isinstance(memory.get("entries"), list):
        raise ValueError("candidate memory entries must be a list")
    if not isinstance(memory.get("last_search"), Mapping):
        raise ValueError("candidate memory last_search must be an object")
    if not 1 <= int(memory.get("max_entries", 0)) <= MAX_ENTRIES_LIMIT:
        raise ValueError("candidate memory max_entries must be between 1 and 6")
