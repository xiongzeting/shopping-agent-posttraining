#!/usr/bin/env python3
"""Select strict Teacher trajectories with corpus-level action diversity caps."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import tempfile

from shopping_grpo.collection.sft import acceptance_reasons


BANDS = ("lt10", "10to15", "15to18", "ge18")
DEFAULT_QUOTAS = {"lt10": 121, "10to15": 298, "15to18": 64, "ge18": 17}
CRITICAL_REPEAT_TOOLS = {"search_products", "open_product", "select_option", "buy_now"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--band-raw",
        action="append",
        required=True,
        metavar="BAND=PATH",
        help="Raw trajectory JSONL assigned to one frozen difficulty band.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-total", type=int, default=500)
    parser.add_argument("--sequence-cap-ratio", type=float, default=0.12)
    parser.add_argument("--eight-step-cap-ratio", type=float, default=0.30)
    parser.add_argument("--seed", default="fresh-sft-diverse-selection-v1")
    return parser.parse_args()


def _parse_band_paths(values: list[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--band-raw must be BAND=PATH: {value}")
        band, raw_path = value.split("=", 1)
        if band not in BANDS:
            raise SystemExit(f"unknown difficulty band: {band}")
        result[band].append(Path(raw_path))
    return result


def _sequence(row: dict) -> tuple[str, ...]:
    return tuple(step.get("tool_name") or "" for step in row.get("steps") or [])


def _step_parameters(step: dict) -> dict:
    parameters = step.get("parameters")
    if isinstance(parameters, dict):
        return parameters
    function = (step.get("tool_call") or {}).get("function") or {}
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return arguments if isinstance(arguments, dict) else {}


def _critical_signature(step: dict) -> str | None:
    tool_name = str(step.get("tool_name") or "")
    if tool_name not in CRITICAL_REPEAT_TOOLS:
        return None
    parameters = _step_parameters(step)
    if tool_name != "buy_now" and not parameters:
        return None
    return json.dumps(
        [tool_name, parameters],
        ensure_ascii=False,
        sort_keys=True,
    ).casefold()


def _quality_reasons(row: dict) -> list[str]:
    sequence = _sequence(row)
    reasons = []
    if not sequence or sequence[-1] != "buy_now":
        reasons.append("quality_missing_explicit_terminal_buy")
    if _has_local_repeat_loop(row):
        reasons.append("quality_repeated_critical_action")
    return reasons


def _has_local_repeat_loop(row: dict, *, window: int = 3) -> bool:
    steps = row.get("steps") or []
    initial_state = (row.get("initial_result") or {}).get("observation_state")
    previous_occurrences: dict[str, list[tuple[int, object]]] = defaultdict(list)
    for index, step in enumerate(steps):
        signature = _critical_signature(step)
        if signature is None:
            continue
        pre_state = (
            initial_state
            if index == 0
            else ((steps[index - 1].get("result") or {}).get("observation_state"))
        )
        for previous_index, previous_state in reversed(previous_occurrences[signature]):
            distance = index - previous_index
            if distance > window:
                break
            if pre_state is None or previous_state is None or pre_state == previous_state:
                return True
        previous_occurrences[signature].append((index, pre_state))
    return False


def _coverage(row: dict) -> dict[str, bool]:
    return _sequence_coverage(_sequence(row))


def _sequence_coverage(sequence: tuple[str, ...]) -> dict[str, bool]:
    return {
        "multiple_searches": sequence.count("search_products") >= 2,
        "multiple_candidates": sequence.count("open_product") >= 2,
        "back_to_search": "back_to_search" in sequence,
        "reviews": "view_reviews" in sequence,
        "attributes": "view_attributes" in sequence,
        "next_page": "next_page" in sequence,
        "multiple_options": sequence.count("select_option") >= 2,
        "longer_than_10": len(sequence) > 10,
    }


def _problem_coverage(row: dict) -> dict[str, bool]:
    sequence = _sequence(row)
    base = _coverage(row)
    evidence_actions = {
        "view_features",
        "view_description",
        "view_attributes",
        "view_reviews",
    }
    return {
        "loop_recovery": base["multiple_searches"]
        and (
            base["multiple_candidates"]
            or base["back_to_search"]
            or base["next_page"]
        ),
        "evidence_verification": any(name in evidence_actions for name in sequence),
        "variant_selection": "select_option" in sequence,
        "explicit_terminal_buy": bool(sequence) and sequence[-1] == "buy_now",
        "clean_critical_actions": not _quality_reasons(row),
    }


def _coverage_score(row: dict) -> int:
    weights = {
        "multiple_searches": 4,
        "multiple_candidates": 4,
        "back_to_search": 3,
        "reviews": 3,
        "attributes": 3,
        "next_page": 2,
        "multiple_options": 2,
        "longer_than_10": 2,
    }
    problem_weights = {
        "loop_recovery": 8,
        "evidence_verification": 5,
        "variant_selection": 4,
        "explicit_terminal_buy": 2,
        "clean_critical_actions": 2,
    }
    return sum(weights[name] for name, present in _coverage(row).items() if present) + sum(
        problem_weights[name]
        for name, present in _problem_coverage(row).items()
        if present
    )


def _stable(seed: str, row: dict) -> str:
    return _stable_values(seed, int(row["task_id"]), row.get("trajectory_id", ""))


def _stable_values(seed: str, task_id: int, trajectory_id: str) -> str:
    value = f"{seed}:{int(task_id)}:{trajectory_id}"
    return hashlib.sha256(value.encode()).hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
    temporary.replace(path)


def _selection_summary(rows: list[dict]) -> dict:
    sequences = Counter(_sequence(row) for row in rows)
    lengths = Counter(len(_sequence(row)) for row in rows)
    coverage = Counter()
    actions = Counter()
    problem_coverage = Counter()
    for row in rows:
        coverage.update(name for name, present in _coverage(row).items() if present)
        problem_coverage.update(
            name for name, present in _problem_coverage(row).items() if present
        )
        actions.update(_sequence(row))
    return {
        "rows": len(rows),
        "unique_task_ids": len({int(row["task_id"]) for row in rows}),
        "unique_sequences": len(sequences),
        "length_histogram": dict(sorted(lengths.items())),
        "eight_step_rows": lengths[8],
        "eight_step_share": round(lengths[8] / len(rows), 6) if rows else None,
        "top_sequence_share": round(sequences.most_common(1)[0][1] / len(rows), 6)
        if rows
        else None,
        "top_10_sequence_share": round(
            sum(count for _, count in sequences.most_common(10)) / len(rows), 6
        )
        if rows
        else None,
        "top_sequences": [
            {"count": count, "sequence": " -> ".join(sequence)}
            for sequence, count in sequences.most_common(10)
        ],
        "coverage": dict(sorted(coverage.items())),
        "problem_coverage": dict(sorted(problem_coverage.items())),
        "actions": dict(sorted(actions.items())),
    }


def _reference_summary(references: list[dict]) -> dict:
    sequences = Counter(reference["sequence"] for reference in references)
    lengths = Counter(reference["length"] for reference in references)
    coverage = Counter()
    actions = Counter()
    for reference in references:
        sequence = reference["sequence"]
        coverage.update(
            name
            for name, present in _sequence_coverage(sequence).items()
            if present
        )
        actions.update(sequence)
    return {
        "rows": len(references),
        "unique_task_ids": len({reference["task_id"] for reference in references}),
        "unique_sequences": len(sequences),
        "length_histogram": dict(sorted(lengths.items())),
        "eight_step_rows": lengths[8],
        "eight_step_share": round(lengths[8] / len(references), 6)
        if references
        else None,
        "top_sequence_share": round(
            sequences.most_common(1)[0][1] / len(references), 6
        )
        if references
        else None,
        "top_10_sequence_share": round(
            sum(count for _, count in sequences.most_common(10)) / len(references), 6
        )
        if references
        else None,
        "top_sequences": [
            {"count": count, "sequence": " -> ".join(sequence)}
            for sequence, count in sequences.most_common(10)
        ],
        "coverage": dict(sorted(coverage.items())),
        "actions": dict(sorted(actions.items())),
    }


def _candidate_reference(row: dict, path: Path, offset: int, seed: str) -> dict:
    sequence = _sequence(row)
    return {
        "task_id": int(row["task_id"]),
        "trajectory_id": row.get("trajectory_id", ""),
        "path": path,
        "offset": int(offset),
        "sequence": sequence,
        "length": len(sequence),
        "coverage_score": _coverage_score(row),
        "stable": _stable(seed, row),
    }


def _references_from_paths(
    paths_by_band: dict[str, list[Path]], *, seed: str
) -> dict[str, list[dict]]:
    deduplicated = {}
    for band in BANDS:
        best_by_task = {}
        for path in paths_by_band.get(band, []):
            with Path(path).open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    row = json.loads(line.decode("utf-8"))
                    if not acceptance_reasons(row)[0]:
                        continue
                    if _quality_reasons(row):
                        continue
                    candidate = _candidate_reference(row, Path(path), offset, seed)
                    task_id = candidate["task_id"]
                    incumbent = best_by_task.get(task_id)
                    candidate_key = (
                        candidate["coverage_score"],
                        -candidate["length"],
                        candidate["stable"],
                    )
                    if incumbent is None:
                        best_by_task[task_id] = candidate
                        continue
                    incumbent_key = (
                        incumbent["coverage_score"],
                        -incumbent["length"],
                        incumbent["stable"],
                    )
                    if candidate_key > incumbent_key:
                        best_by_task[task_id] = candidate
        deduplicated[band] = list(best_by_task.values())
    return deduplicated


def select_references(
    references_by_band: dict[str, list[dict]],
    *,
    quotas: dict[str, int],
    target_total: int,
    sequence_cap_ratio: float,
    eight_step_cap_ratio: float,
) -> tuple[list[dict], dict]:
    sequence_cap = max(1, math.floor(target_total * sequence_cap_ratio))
    eight_step_cap = max(1, math.floor(target_total * eight_step_cap_ratio))
    selected = []
    selected_by_band = Counter()
    sequence_counts = Counter()
    eight_step_rows = 0
    remaining = {
        band: list(references_by_band.get(band, [])) for band in BANDS
    }

    while True:
        progress = False
        for band in BANDS:
            if selected_by_band[band] >= quotas[band]:
                continue
            eligible = [
                reference
                for reference in remaining[band]
                if sequence_counts[reference["sequence"]] < sequence_cap
                and not (
                    reference["length"] == 8 and eight_step_rows >= eight_step_cap
                )
            ]
            if not eligible:
                continue
            reference = min(
                eligible,
                key=lambda item: (
                    sequence_counts[item["sequence"]],
                    item["length"] == 8,
                    -item["coverage_score"],
                    item["stable"],
                ),
            )
            remaining[band].remove(reference)
            selected.append(reference)
            selected_by_band[band] += 1
            sequence_counts[reference["sequence"]] += 1
            eight_step_rows += reference["length"] == 8
            progress = True
        if not progress or sum(selected_by_band.values()) >= target_total:
            break

    report = {
        "schema_version": "shopping-sft-diversity-selection-v1",
        "target_total": target_total,
        "quotas": quotas,
        "strict_available": {
            band: len(references_by_band.get(band, [])) for band in BANDS
        },
        "selected_by_band": dict(selected_by_band),
        "deficits": {
            band: max(quotas[band] - selected_by_band[band], 0) for band in BANDS
        },
        "caps": {
            "exact_sequence_rows": sequence_cap,
            "eight_step_rows": eight_step_cap,
            "sequence_cap_ratio": sequence_cap_ratio,
            "eight_step_cap_ratio": eight_step_cap_ratio,
        },
        "result": _reference_summary(selected),
    }
    return selected, report


def _materialize_references(references: list[dict]) -> list[dict]:
    handles = {}
    rows = []
    try:
        for reference in references:
            path = reference["path"]
            handle = handles.get(path)
            if handle is None:
                handle = Path(path).open("rb")
                handles[path] = handle
            handle.seek(reference["offset"])
            line = handle.readline()
            row = json.loads(line.decode("utf-8"))
            if int(row["task_id"]) != reference["task_id"]:
                raise RuntimeError(f"trajectory reference changed while selecting: {path}")
            rows.append(row)
    finally:
        for handle in handles.values():
            handle.close()
    return rows


def _write_references_atomic(path: Path, references: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handles = {}
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            for reference in references:
                source_path = reference["path"]
                handle = handles.get(source_path)
                if handle is None:
                    handle = Path(source_path).open("rb")
                    handles[source_path] = handle
                handle.seek(reference["offset"])
                row = json.loads(handle.readline().decode("utf-8"))
                if int(row["task_id"]) != reference["task_id"]:
                    raise RuntimeError(
                        f"trajectory reference changed while selecting: {source_path}"
                    )
                stream.write(
                    (
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    ).encode("utf-8")
                )
            stream.flush()
        temporary.replace(path)
    finally:
        for handle in handles.values():
            handle.close()


def select_rows(
    rows_by_band: dict[str, list[dict]],
    *,
    quotas: dict[str, int],
    target_total: int,
    sequence_cap_ratio: float,
    eight_step_cap_ratio: float,
    seed: str,
) -> tuple[list[dict], dict]:
    sequence_cap = max(1, math.floor(target_total * sequence_cap_ratio))
    eight_step_cap = max(1, math.floor(target_total * eight_step_cap_ratio))

    deduplicated: dict[str, list[dict]] = {}
    for band in BANDS:
        best_by_task = {}
        for row in rows_by_band.get(band, []):
            if not acceptance_reasons(row)[0]:
                continue
            if _quality_reasons(row):
                continue
            task_id = int(row["task_id"])
            incumbent = best_by_task.get(task_id)
            candidate_key = (_coverage_score(row), -len(_sequence(row)), _stable(seed, row))
            if incumbent is None:
                best_by_task[task_id] = row
                continue
            incumbent_key = (
                _coverage_score(incumbent),
                -len(_sequence(incumbent)),
                _stable(seed, incumbent),
            )
            if candidate_key > incumbent_key:
                best_by_task[task_id] = row
        deduplicated[band] = list(best_by_task.values())

    selected = []
    selected_by_band = Counter()
    sequence_counts = Counter()
    eight_step_rows = 0
    remaining = {band: list(rows) for band, rows in deduplicated.items()}

    while True:
        progress = False
        for band in BANDS:
            if selected_by_band[band] >= quotas[band]:
                continue
            eligible = []
            for row in remaining.get(band, []):
                sequence = _sequence(row)
                length = len(sequence)
                if sequence_counts[sequence] >= sequence_cap:
                    continue
                if length == 8 and eight_step_rows >= eight_step_cap:
                    continue
                eligible.append(row)
            if not eligible:
                continue
            row = min(
                eligible,
                key=lambda item: (
                    sequence_counts[_sequence(item)],
                    len(_sequence(item)) == 8,
                    -_coverage_score(item),
                    _stable(seed, item),
                ),
            )
            remaining[band].remove(row)
            selected.append(row)
            selected_by_band[band] += 1
            sequence_counts[_sequence(row)] += 1
            eight_step_rows += len(_sequence(row)) == 8
            progress = True
        if not progress or sum(selected_by_band.values()) >= target_total:
            break

    report = {
        "schema_version": "shopping-sft-diversity-selection-v1",
        "target_total": target_total,
        "quotas": quotas,
        "strict_available": {
            band: len(deduplicated.get(band, [])) for band in BANDS
        },
        "selected_by_band": dict(selected_by_band),
        "deficits": {
            band: max(quotas[band] - selected_by_band[band], 0) for band in BANDS
        },
        "caps": {
            "exact_sequence_rows": sequence_cap,
            "eight_step_rows": eight_step_cap,
            "sequence_cap_ratio": sequence_cap_ratio,
            "eight_step_cap_ratio": eight_step_cap_ratio,
        },
        "result": _selection_summary(selected),
    }
    return selected, report


def main() -> None:
    args = parse_args()
    if args.target_total < 1:
        raise SystemExit("--target-total must be positive")
    if not 0 < args.sequence_cap_ratio <= 1:
        raise SystemExit("--sequence-cap-ratio must be in (0, 1]")
    if not 0 < args.eight_step_cap_ratio <= 1:
        raise SystemExit("--eight-step-cap-ratio must be in (0, 1]")

    quota_weights = {
        band: DEFAULT_QUOTAS[band] / sum(DEFAULT_QUOTAS.values()) for band in BANDS
    }
    quotas = {
        band: math.floor(args.target_total * quota_weights[band]) for band in BANDS
    }
    for band in sorted(
        BANDS,
        key=lambda name: (
            -(args.target_total * quota_weights[name] - quotas[name]),
            BANDS.index(name),
        ),
    )[: args.target_total - sum(quotas.values())]:
        quotas[band] += 1

    references_by_band = _references_from_paths(
        _parse_band_paths(args.band_raw), seed=args.seed
    )
    selected_references, report = select_references(
        references_by_band,
        quotas=quotas,
        target_total=args.target_total,
        sequence_cap_ratio=args.sequence_cap_ratio,
        eight_step_cap_ratio=args.eight_step_cap_ratio,
    )
    _write_references_atomic(args.output, selected_references)
    _write_atomic(
        args.report,
        (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    print(json.dumps(report, ensure_ascii=False))
    if len(selected_references) != args.target_total:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
