"""Deterministic parsing of public Chinese price expressions."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


PRICE_SEMANTICS_VERSION = "shopping-price-semantics-v3"

_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_LARGE_UNITS = {"万": 10000, "亿": 100000000}
_ARABIC_AMOUNT = r"\d+(?:\.\d+)?"
_CHINESE_INTEGER = r"[零〇一二两三四五六七八九十百千万亿]+"
_CHINESE_DECIMAL = r"[零〇一二两三四五六七八九]+"
_CHINESE_AMOUNT = rf"{_CHINESE_INTEGER}(?:点{_CHINESE_DECIMAL})?"
_MIXED_SHORTHAND_AMOUNT = rf"{_ARABIC_AMOUNT}\s*[万千]\s*{_ARABIC_AMOUNT}(?:\s*千)?"
_AMOUNT = rf"(?:{_MIXED_SHORTHAND_AMOUNT}|{_ARABIC_AMOUNT}|{_CHINESE_AMOUNT})"
_AMOUNT_CONTINUATION = r"[零〇一二两三四五六七八九十百千万亿点\d.kK]"
_UNIT = r"(?:万|千|[kK])"
_CURRENCY = r"(?:元|块(?:钱)?)"
_PRICE_LABEL = r"(?:预算|价格|售价|价位|价钱)"
_SOFT_PREFIX = r"(?:大约|大概|约莫|约摸|差不多|大致|将近|接近|近|约)"
_UPPER_WORDS = r"(?:不超过|不要超过|别超过|别超|不能超过|不得超过|不可超过|不高于|最高|至多|最多|上限|封顶|低于|小于)"
_LOWER_WORDS = r"(?:不低于|至少|最低|不少于|高于|大于|超过|起码)"
_AROUND_WORDS = r"(?:左右|上下|前后|附近|差不多)"
_OPEN_ABOVE_WORDS = r"(?:多(?:一点|一些)?|出头|开外|往上|以上|起)"
_LOOSE_ABOVE_WORDS = r"(?:多|来|余)"
_PER_UNIT_PRICE = re.compile(
    r"每\s*(?P<unit>平方米|平米|㎡|包|罐|瓶|盒|袋|条|件|个|只|支|斤|公斤|千克|克|杯)"
    r"\s*(?:的)?\s*(?:价格|售价|价钱|单价)"
)


@dataclass(frozen=True)
class PriceExpression:
    kind: str
    value: float | None
    lower: float | None
    upper: float | None
    text: str
    start: int
    end: int
    approximate: bool = False
    open_ended_above: bool = False
    lower_inclusive: bool = True
    upper_inclusive: bool = True


def normalize_price_text(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).replace(",", "").replace("，", "")


def _parse_standard_chinese_integer(text: str) -> float:
    total = 0
    section = 0
    current = 0
    for char in text:
        if char in _DIGITS:
            current = _DIGITS[char]
        elif char in _SMALL_UNITS:
            section += (current or 1) * _SMALL_UNITS[char]
            current = 0
        elif char in _LARGE_UNITS:
            section += current
            current = 0
            unit = _LARGE_UNITS[char]
            if char == "亿":
                total = (total + section) * unit
            else:
                total += section * unit
            section = 0
        else:
            raise ValueError(f"unsupported Chinese amount character: {char}")
    return float(total + section + current)


def parse_chinese_amount(value: object) -> float:
    """Parse written and common colloquial Chinese amounts.

    A trailing digit after 千/万 is interpreted as the next omitted place:
    两千五=2500, 一万二=12000, 两万三千五=23500. Explicit zeros keep
    their standard positional meaning, so 一千零五 remains 1005.
    """

    text = normalize_price_text(value).strip()
    if not text:
        raise ValueError("empty Chinese amount")
    integer_text, separator, decimal_text = text.partition("点")
    result = _parse_standard_chinese_integer(integer_text)

    if (
        len(integer_text) >= 2
        and integer_text[-1] in _DIGITS
        and integer_text[-2] != "零"
    ):
        highest_earlier_unit = max(
            (
                unit
                for char, unit in {**_SMALL_UNITS, **_LARGE_UNITS}.items()
                if char in integer_text[:-1]
            ),
            default=0,
        )
        last_explicit_unit = next(
            (
                {**_SMALL_UNITS, **_LARGE_UNITS}[char]
                for char in reversed(integer_text[:-1])
                if char in {**_SMALL_UNITS, **_LARGE_UNITS}
            ),
            0,
        )
        if highest_earlier_unit >= 1000 and last_explicit_unit >= 1000:
            omitted_place = last_explicit_unit // 10
            result += _DIGITS[integer_text[-1]] * omitted_place - _DIGITS[integer_text[-1]]
        elif (
            last_explicit_unit == 100
            and highest_earlier_unit == 100
            and len(integer_text) == 3
            and integer_text[0] in _DIGITS
        ):
            # Colloquial prices commonly omit the trailing ten: 一百三=130.
            result += _DIGITS[integer_text[-1]] * 10 - _DIGITS[integer_text[-1]]

    if separator:
        if any(char not in _DIGITS for char in decimal_text):
            raise ValueError(f"unsupported Chinese decimal: {decimal_text}")
        if decimal_text:
            result += float("0." + "".join(str(_DIGITS[char]) for char in decimal_text))
    return result


def parse_amount(number: object, unit: object = None) -> float:
    text = normalize_price_text(number).strip()
    mixed = re.fullmatch(
        rf"(?P<major>{_ARABIC_AMOUNT})\s*(?P<major_unit>[万千])\s*"
        rf"(?P<tail>{_ARABIC_AMOUNT})(?:\s*(?P<tail_unit>千))?",
        text,
    )
    if mixed:
        major_unit = 10000 if mixed.group("major_unit") == "万" else 1000
        tail = float(mixed.group("tail"))
        if mixed.group("tail_unit") == "千":
            tail *= 1000
        elif tail < 10:
            tail *= major_unit / 10
        value = float(mixed.group("major")) * major_unit + tail
    else:
        value = float(text) if re.fullmatch(_ARABIC_AMOUNT, text) else parse_chinese_amount(text)
    normalized_unit = normalize_price_text(unit).casefold()
    if normalized_unit == "万":
        value *= 10000
    elif normalized_unit in {"千", "k"}:
        value *= 1000
    return value


def approximate_price_bounds(value: float) -> tuple[float, float]:
    """Return stable bounds for colloquial 'around' expressions.

    Reward v4 treats colloquial target prices as an exact plus-or-minus twenty
    percent interval so the Actor prompt and deterministic comparison agree.
    """

    tolerance = value * 0.20
    return max(0.0, value - tolerance), value + tolerance


def _price_basis(instruction: object) -> dict:
    match = _PER_UNIT_PRICE.search(normalize_price_text(instruction))
    if not match:
        return {"kind": "variant_total"}
    unit = match.group("unit")
    if unit in {"平米", "㎡"}:
        unit = "平方米"
    return {"kind": "per_unit", "unit": unit, "source_text": match.group(0)}


def _number_pattern(name: str) -> str:
    return (
        rf"(?P<{name}>{_AMOUNT})\s*(?P<{name}_unit>{_UNIT})?"
        rf"(?!{_AMOUNT_CONTINUATION})"
    )


def _amount_from(match: re.Match[str], name: str) -> float:
    return parse_amount(match.group(name), match.group(f"{name}_unit"))


def _record(
    match: re.Match[str],
    *,
    kind: str,
    value: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
    approximate: bool = False,
    open_ended_above: bool = False,
    lower_inclusive: bool = True,
    upper_inclusive: bool = True,
) -> PriceExpression:
    return PriceExpression(
        kind=kind,
        value=value,
        lower=lower,
        upper=upper,
        text=match.group(0),
        start=match.start(),
        end=match.end(),
        approximate=approximate,
        open_ended_above=open_ended_above,
        lower_inclusive=lower_inclusive,
        upper_inclusive=upper_inclusive,
    )


def extract_price_expressions(instruction: object) -> list[PriceExpression]:
    """Extract non-overlapping, query-grounded price semantics."""

    text = normalize_price_text(instruction)
    occupied: list[tuple[int, int]] = []
    results: list[PriceExpression] = []

    def previous_word(start: int, limit: int = 8) -> str:
        return text[max(0, start - limit):start]

    def excluded_cost_context(match: re.Match[str]) -> bool:
        prefix = previous_word(match.start())
        return bool(
            re.search(r"(?:设计|手工|加工|安装|配送|服务|定制|补|尾|订|定)费(?:在|为|是)?$", prefix)
            or re.search(r"(?:工费|运费|邮费|设计费|加工费|安装费|服务费|补款|尾款|订金|定金)(?:在|为|是)?$", prefix)
        )

    def starts_inside_unit_price_phrase(match: re.Match[str]) -> bool:
        suffix = text[match.end():match.end() + 20]
        return bool(
            re.fullmatch(r"(?:预算|价格|售价|价位|价钱)\s*(?:控制)?\s*(?:在|为|是)?\s*(?:\d+|[一二两三四五六七八九十百千万]+)", match.group(0))
            and re.match(r"(?:米|件|个|只|支|包|罐|斤|克|公斤|套|箱)\s*\d", suffix)
        )

    def truncated_label_amount(match: re.Match[str]) -> bool:
        if not re.search(_PRICE_LABEL, match.group(0)):
            return False
        suffix = text[match.end():match.end() + 12]
        return bool(
            re.match(
                r"(?:一共|总共|合计|共|每|单价|每米|每件|每个|每只|每支|每包|每罐|每斤)",
                suffix,
            )
        )

    def add(match: re.Match[str], expression: PriceExpression) -> None:
        if any(match.start() < end and start < match.end() for start, end in occupied):
            return
        occupied.append((match.start(), match.end()))
        results.append(expression)

    def has_price_context(match: re.Match[str]) -> bool:
        matched = match.group(0)
        return bool(re.search(_PRICE_LABEL, matched) or re.search(_CURRENCY, matched))

    lead = rf"(?:{_PRICE_LABEL}\s*(?:控制)?\s*(?:在|为|是)?\s*)?(?:{_SOFT_PREFIX}\s*)?"
    currency = rf"\s*{_CURRENCY}?"

    range_pattern = re.compile(
        rf"{lead}{_number_pattern('low')}{currency}\s*(?:-|~|～|—|至|到)\s*"
        rf"{_number_pattern('high')}{currency}(?:之间|区间|范围内|以内|之内)?"
    )
    for match in range_pattern.finditer(text):
        if not has_price_context(match) or excluded_cost_context(match):
            continue
        low = _amount_from(match, "low")
        high = _amount_from(match, "high")
        if low > 0 and high >= low:
            add(match, _record(match, kind="range", lower=low, upper=high))

    ambiguous_upper_pattern = re.compile(
        rf"(?:{_PRICE_LABEL}\s*(?:控制)?\s*(?:在|为|是)?\s*)?{_UPPER_WORDS}\s*"
        rf"{_number_pattern('amount')}\s*{_LOOSE_ABOVE_WORDS}\s*{_CURRENCY}?"
    )
    for match in ambiguous_upper_pattern.finditer(text):
        if not has_price_context(match) or excluded_cost_context(match):
            continue
        value = _amount_from(match, "amount")
        if value > 0:
            add(
                match,
                _record(
                    match,
                    kind="preference",
                    value=value,
                    approximate=True,
                    open_ended_above=True,
                ),
            )

    prefix_patterns = (
        ("upper", _UPPER_WORDS),
        ("lower", _LOWER_WORDS),
    )
    for kind, words in prefix_patterns:
        pattern = re.compile(
            rf"(?:{_PRICE_LABEL}\s*(?:控制)?\s*(?:在|为|是)?\s*)?{words}\s*"
            rf"{_number_pattern('amount')}{currency}"
        )
        for match in pattern.finditer(text):
            if not has_price_context(match) or excluded_cost_context(match):
                continue
            value = _amount_from(match, "amount")
            if value > 0:
                matched = match.group(0)
                add(
                    match,
                    _record(
                        match,
                        kind=kind,
                        value=value,
                        lower_inclusive=not bool(
                            kind == "lower" and re.search(r"高于|大于|超过", matched)
                        ),
                        upper_inclusive=not bool(
                            kind == "upper" and re.search(r"低于|小于", matched)
                        ),
                    ),
                )

    suffix_patterns = (
        ("upper", r"(?:以内|以下|之内|内)"),
        ("lower", r"(?:以上|起步|起)"),
    )
    for kind, words in suffix_patterns:
        pattern = re.compile(
            rf"{lead}{_number_pattern('amount')}{currency}\s*{words}"
        )
        for match in pattern.finditer(text):
            if not has_price_context(match) or excluded_cost_context(match):
                continue
            value = _amount_from(match, "amount")
            if value > 0:
                add(match, _record(match, kind=kind, value=value))

    open_pattern = re.compile(
        rf"{lead}{_number_pattern('amount')}\s*(?:{_LOOSE_ABOVE_WORDS}\s*{_CURRENCY}?|"
        rf"{_CURRENCY}?\s*{_OPEN_ABOVE_WORDS})(?:\s*(?:就行|即可|可以|能接受))?"
    )
    for match in open_pattern.finditer(text):
        if not has_price_context(match) or excluded_cost_context(match):
            continue
        value = _amount_from(match, "amount")
        if value > 0:
            add(
                match,
                _record(
                    match,
                    kind="preference",
                    value=value,
                    approximate=True,
                    open_ended_above=True,
                ),
            )

    around_pattern = re.compile(
        rf"{lead}{_number_pattern('amount')}{currency}\s*{_AROUND_WORDS}"
    )
    for match in around_pattern.finditer(text):
        if not has_price_context(match) or excluded_cost_context(match):
            continue
        value = _amount_from(match, "amount")
        if value > 0:
            low, high = approximate_price_bounds(value)
            add(
                match,
                _record(
                    match,
                    kind="around",
                    value=value,
                    lower=low,
                    upper=high,
                    approximate=True,
                ),
            )

    soft_prefix_pattern = re.compile(
        rf"(?:{_PRICE_LABEL}\s*(?:控制)?\s*(?:在|为|是)?\s*)?{_SOFT_PREFIX}\s*(?:在|为|是)?\s*"
        rf"{_number_pattern('amount')}{currency}(?!\s*(?:{_AROUND_WORDS}|以内|以下|之内|内|以上|起|{_OPEN_ABOVE_WORDS}))"
    )
    for match in soft_prefix_pattern.finditer(text):
        if not has_price_context(match) or excluded_cost_context(match):
            continue
        value = _amount_from(match, "amount")
        if value > 0:
            low, high = approximate_price_bounds(value)
            add(
                match,
                _record(
                    match,
                    kind="around",
                    value=value,
                    lower=low,
                    upper=high,
                    approximate=True,
                ),
            )

    labeled_exact_pattern = re.compile(
        rf"{_PRICE_LABEL}\s*(?:控制)?\s*(?:在|为|是)?\s*"
        rf"{_number_pattern('amount')}{currency}"
        rf"(?!\s*(?:[-~～—至到+]|{_LOOSE_ABOVE_WORDS}|{_AROUND_WORDS}|{_OPEN_ABOVE_WORDS}|以内|以下|之内|内|以上|起))"
    )
    for match in labeled_exact_pattern.finditer(text):
        if (
            excluded_cost_context(match)
            or starts_inside_unit_price_phrase(match)
            or truncated_label_amount(match)
        ):
            continue
        value = _amount_from(match, "amount")
        if value > 0:
            add(match, _record(match, kind="upper", value=value))

    plus_pattern = re.compile(
        rf"{lead}{_number_pattern('amount')}\s*(?:{_CURRENCY}\s*)?\+"
    )
    for match in plus_pattern.finditer(text):
        if not has_price_context(match) or excluded_cost_context(match):
            continue
        value = _amount_from(match, "amount")
        if value > 0:
            add(
                match,
                _record(
                    match,
                    kind="preference",
                    value=value,
                    approximate=True,
                    open_ended_above=True,
                ),
            )

    return sorted(results, key=lambda item: item.start)


def explicit_budget_upper(instruction: object) -> float | None:
    """Return Reward's deterministic public-query upper price gate."""

    expressions = extract_price_expressions(instruction)
    upper_values = []
    for expression in expressions:
        if expression.kind == "upper" and expression.value is not None:
            upper_values.append(expression.value)
        elif expression.kind in {"range", "around"} and expression.upper is not None:
            upper_values.append(expression.upper)
    return min(upper_values) if upper_values else None


def reward_price_constraint(instruction: object) -> dict | None:
    """Compile the public price language used by the deterministic Reward."""

    expressions = extract_price_expressions(instruction)
    basis = _price_basis(instruction)
    hard = [item for item in expressions if item.kind in {"upper", "lower", "range"}]
    around = [item for item in expressions if item.kind == "around"]
    explicit_ranges = [item for item in hard if item.kind == "range"]
    uppers = [item for item in hard if item.kind == "upper" and item.value is not None]
    lowers = [item for item in hard if item.kind == "lower" and item.value is not None]
    if explicit_ranges:
        expression = explicit_ranges[0]
    elif uppers and lowers:
        upper = min(uppers, key=lambda item: item.value)
        lower = max(lowers, key=lambda item: item.value)
        if lower.value > upper.value:
            return None
        return {
            "operator": "between",
            "min": lower.value,
            "max": upper.value,
            "min_inclusive": lower.lower_inclusive,
            "max_inclusive": upper.upper_inclusive,
            "source_text": f"{lower.text}；{upper.text}",
            "approximate": False,
            "basis": basis,
        }
    elif uppers:
        expression = min(uppers, key=lambda item: item.value)
    elif lowers:
        expression = max(lowers, key=lambda item: item.value)
    elif around:
        expression = around[-1]
    else:
        return None
    if expression.kind == "upper":
        return {
            "operator": "lte" if expression.upper_inclusive else "lt",
            "value": expression.value,
            "source_text": expression.text,
            "approximate": False,
            "basis": basis,
        }
    if expression.kind == "lower":
        return {
            "operator": "gte" if expression.lower_inclusive else "gt",
            "value": expression.value,
            "source_text": expression.text,
            "approximate": False,
            "basis": basis,
        }
    if expression.kind == "range":
        return {
            "operator": "between",
            "min": expression.lower,
            "max": expression.upper,
            "min_inclusive": expression.lower_inclusive,
            "max_inclusive": expression.upper_inclusive,
            "source_text": expression.text,
            "approximate": False,
            "basis": basis,
        }
    return {
        "operator": "approximately",
        "value": expression.value,
        "min": expression.lower,
        "max": expression.upper,
        "source_text": expression.text,
        "approximate": True,
        "basis": basis,
    }
