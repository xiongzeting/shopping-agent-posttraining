"""Structured, answer-free ShopSimulator state for the canonical Agent renderer."""

from __future__ import annotations

import hashlib
import json


OBSERVATION_VERSION = "shopping-observation-v2"
HIDDEN_INFORMATION_ACTIONS = {
    "description",
    "features",
    "reviews",
    "attributes",
}


def stable_option_id(asin: object, axis: object, value: object) -> str:
    """Return a deterministic public identifier for one product option."""
    identity = json.dumps(
        [
            str(asin or "").strip().upper(),
            str(axis or "").strip().casefold(),
            str(value or "").strip(),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"opt_{digest}"


def available_option_records(product: dict) -> dict:
    """Expose labels for reading and stable IDs for selecting options."""
    asin = product.get("asin")
    records = {}
    for raw_axis, raw_values in (product.get("options") or {}).items():
        axis = str(raw_axis)
        values = raw_values if isinstance(raw_values, (list, tuple)) else []
        records[axis] = [
            {
                "option_id": stable_option_id(asin, axis, value),
                "label": str(value),
            }
            for value in values[:100]
            if str(value).strip()
        ]
    return records


def selected_option_records(asin: object, selected_options: object) -> dict:
    """Render selected raw values without changing Reward's internal format."""
    if not isinstance(selected_options, dict):
        return {}
    return {
        str(axis): {
            "option_id": stable_option_id(asin, axis, value),
            "label": str(value),
        }
        for axis, value in selected_options.items()
        if str(axis).strip() and str(value).strip()
    }


def _compact_list(value, limit=10):
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    return [str(item) for item in value[:limit] if str(item).strip()]


def product_summary(
    product: dict,
    *,
    rank: int | None = None,
    include_details: bool = False,
) -> dict:
    pricing = product.get("Price")
    if pricing is None:
        pricing = product.get("pricing")
    result = {
        "asin": str(product.get("asin", "")),
        "title": str(product.get("title") or product.get("Title") or ""),
        "brand": str(product.get("brand") or product.get("shop_name") or ""),
        "category": str(product.get("category") or ""),
        "price": pricing,
        "key_attributes": _compact_list(
            product.get("attribute") or product.get("Attributes") or []
        ),
    }
    if include_details:
        features = _compact_list(product.get("BulletPoints") or [], limit=20)
        attributes = _compact_list(
            product.get("Attributes") or product.get("attribute") or [],
            limit=30,
        )
        if features:
            result["features"] = features
        if attributes:
            result["attributes"] = attributes
    if rank is not None:
        result["rank"] = int(rank)
    return result


def build_observation_state(
    *,
    page_type: str,
    session: dict,
    product_item_dict: dict,
    available_actions: dict,
) -> dict:
    """Build one public state; the task goal is intentionally not accepted."""
    clickables = [
        str(value)
        for value in available_actions.get("clickables", [])
        if str(value).strip().casefold() not in HIDDEN_INFORMATION_ACTIONS
    ]
    state = {
        "observation_version": OBSERVATION_VERSION,
        "page_type": page_type,
        "search_available": bool(available_actions.get("has_search_bar")),
        "actions": clickables,
    }
    if page_type == "search_results":
        all_asins = session.get("search_result_asins") or []
        page_asins = session.get("current_page_asins") or []
        rank_by_asin = {
            asin: index for index, asin in enumerate(all_asins, start=1)
        }
        state.update(
            {
                "query": " ".join(session.get("keywords") or []),
                "normalized_query": session.get("normalized_query") or "",
                "page": int(session.get("page") or 1),
                "total_pages": int(session.get("total_pages") or 1),
                "total_results": int(session.get("total_results") or 0),
                "rank_start": min(
                    (rank_by_asin.get(asin, 0) for asin in page_asins),
                    default=0,
                ),
                "rank_end": max(
                    (rank_by_asin.get(asin, 0) for asin in page_asins),
                    default=0,
                ),
                "products": [
                    product_summary(
                        product_item_dict[asin],
                        rank=rank_by_asin[asin],
                    )
                    for asin in page_asins
                    if asin in product_item_dict and asin in rank_by_asin
                ],
            }
        )
    elif page_type in {"product_detail", "information_subpage"}:
        asin = session.get("asin")
        product = product_item_dict.get(asin, {})
        state["product"] = product_summary(product, include_details=True)
        state["selected_options"] = (
            selected_option_records(asin, session.get("options"))
            if page_type == "product_detail"
            else {}
        )
        state["available_options"] = (
            available_option_records(product)
            if page_type == "product_detail"
            else {}
        )
        if session.get("selected_price") is not None:
            state["selected_price"] = session["selected_price"]
        if page_type == "information_subpage":
            subpage = str(session.get("subpage") or "information")
            field = {
                "description": "Description",
                "features": "BulletPoints",
                "reviews": "Reviews",
                "attributes": "Attributes",
            }.get(subpage.casefold())
            state["subpage"] = subpage
            state["content"] = product.get(field, "") if field else ""
    return state


def page_type_from_name(page_name: str) -> str:
    return {
        "": "search_home",
        "search_results": "search_results",
        "item_page": "product_detail",
        "item_sub_page": "information_subpage",
        "done": "terminal",
    }.get(page_name, "unknown")
