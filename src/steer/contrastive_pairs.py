"""Generate constraint-violation pairs without loading data or writing files.

Raw MultiWOZ records remain mappings at the input boundary; generated records
have an explicit schema. Only the target system response is edited.
"""

import random
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, TypedDict

from steer.ontology import SLOT_VALUES


class ResponseEdit(TypedDict):
    """One annotated value replacement within a system response."""

    positive_response: str
    negative_response: str
    slot: str
    positive_value: str
    negative_value: str
    allowed_values: list[str]
    span_start: int
    span_end: int


class ContrastivePair(ResponseEdit):
    """Serializable response pair with shared history and source identifiers."""

    dialogue_id: str
    turn_id: str
    prompt: str
    messages: list[dict[str, str]]


def normalize_text(text: object) -> str:
    return " ".join(str(text).casefold().split())


def is_user(turns: Mapping[str, Any], index: int) -> bool:
    return turns["speaker"][index] in (0, "USER", "user")


def extract_slot_constraints(
    turns: Mapping[str, Any], index: int
) -> dict[str, list[str]]:
    """Read the current user-state snapshot; never reuse a cleared constraint."""
    result = {}
    for state in turns["frames"][index].get("state", []):
        slots = state.get("slots_values", {})
        for name, values in zip(
            slots.get("slots_values_name", []),
            slots.get("slots_values_list", []),
            strict=True,
        ):
            if isinstance(values, str):
                values = [values]
            cleaned = [normalize_text(v) for v in values]
            # A don't-care constraint cannot establish a negative example.
            if any(v in {"dontcare", "don't care", "do not care"} for v in cleaned):
                continue
            cleaned = [v for v in cleaned if v not in {"", "none", "?"}]
            if cleaned:
                result[normalize_text(name)] = cleaned
    return result


def _pick_negative_value(
    options: Sequence[str], allowed_values: Sequence[str], rng: random.Random
) -> str | None:
    """Exclude every acceptable value, not only the observed response value."""
    choices = sorted({normalize_text(v) for v in options} - set(allowed_values))
    return rng.choice(choices) if choices else None


def _iter_response_edits(
    turns: Mapping[str, Any],
    utterance: str,
    constraints: Mapping[str, list[str]],
    index: int,
    rng: random.Random,
) -> Iterator[ResponseEdit]:
    """Yield one controlled response edit per eligible annotated span.

    This dataset uses character offsets with an exclusive span_end. Skip
    mismatched annotations rather than guessing the location of a value.
    """
    if turns["speaker"][index] not in (1, "SYSTEM", "system"):
        return
    spans = turns["dialogue_acts"][index].get("span_info", {})
    fields = ("act_type", "act_slot_name", "act_slot_value", "span_start", "span_end")
    seen = set()
    for act, name, value, start, end in zip(
        *(spans.get(field, []) for field in fields), strict=True
    ):
        domain, _, intent = normalize_text(act).partition("-")
        if intent not in {"inform", "recommend"}:
            continue
        slot = f"{domain}-{normalize_text(name)}"
        allowed = constraints.get(slot, [])
        if slot not in SLOT_VALUES or normalize_text(value) not in allowed:
            continue
        start, end = int(start), int(end)
        if not 0 <= start < end <= len(utterance):
            continue
        if normalize_text(utterance[start:end]) != normalize_text(value):
            continue
        key = (slot, start, end)
        if key in seen:
            continue
        seen.add(key)
        negative_value = _pick_negative_value(SLOT_VALUES[slot], allowed, rng)
        if negative_value is None:
            continue
        yield {
            "positive_response": utterance,
            "negative_response": utterance[:start] + negative_value + utterance[end:],
            "slot": slot,
            "positive_value": str(value),
            "negative_value": negative_value,
            "allowed_values": list(allowed),
            "span_start": start,
            "span_end": end,
        }


def create_pairs(
    row: Mapping[str, Any], rng: random.Random | None = None
) -> Iterator[ContrastivePair]:
    """Keep history identical; change only the target SYSTEM response."""
    if rng is None:
        rng = random.Random(42)
    turns = row["turns"]
    constraints = {}
    for index, utterance in enumerate(turns["utterance"]):
        if is_user(turns, index):
            constraints = extract_slot_constraints(turns, index)
            continue
        for pair in _iter_response_edits(turns, utterance, constraints, index, rng):
            history = "\n".join(
                f"{'User' if is_user(turns, j) else 'Assistant'}: {turns['utterance'][j]}"
                for j in range(index)
            )
            yield {
                "dialogue_id": row["dialogue_id"],
                "turn_id": turns["turn_id"][index],
                "prompt": history + "\nAssistant:",
                "messages": [
                    {"role": "user" if is_user(turns, j) else "assistant",
                     "content": turns["utterance"][j]}
                    for j in range(index)
                ],
                **pair,
            }
