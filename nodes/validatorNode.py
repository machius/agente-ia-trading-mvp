import re
from langchain.messages import SystemMessage

NUMBER_PATTERN = re.compile(r"-?\d[\d,]*\.?\d*")
MAX_VALIDATION_ATTEMPTS = 2
TOLERANCE = 0.01


def _parse_numbers(text: str) -> set[float]:
    numbers = set()
    for raw in NUMBER_PATTERN.findall(text):
        cleaned = raw.replace(",", "")
        try:
            numbers.add(float(cleaned))
        except ValueError:
            continue
    return numbers


def _is_verified(claimed: float, known: set[float]) -> bool:
    return any(abs(claimed - k) < TOLERANCE for k in known)


def validate_node(state: dict):
    """Checks that figures in the final answer come from actual tool results"""

    last_message = state["messages"][-1]
    claimed_numbers = _parse_numbers(last_message.content)

    known_numbers = set()
    for result in state.get("tool_results", []):
        known_numbers |= _parse_numbers(str(result["result"]))

    unverified = {n for n in claimed_numbers if not _is_verified(n, known_numbers)}
    attempts = state.get("validation_attempts", 0) + 1
    can_retry = attempts <= MAX_VALIDATION_ATTEMPTS

    if unverified and can_retry:
        correction = SystemMessage(
            content=(
                f"These figures don't match any tool result: "
                f"{', '.join(str(n) for n in sorted(unverified))}. "
                f"Rewrite your answer using only verified data, or state clearly "
                f"that the information isn't available."
            )
        )
        return {
            "messages": [correction],
            "validation_attempts": attempts,
            "needs_correction": True,
        }

    return {"validation_attempts": attempts, "needs_correction": False}