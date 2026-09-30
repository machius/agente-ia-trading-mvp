from typing import Literal
from langgraph.graph import END


def route_after_validation(state: dict) -> Literal["llm_call", END]:
    if state.get("needs_correction"):
        return "llm_call"
    return END