from typing import Literal
from state import MessagesState


def should_continue(state: MessagesState) -> Literal["tool_node", "validator"]:
    """Decide si seguir el loop de tools o pasar a validar la respuesta final"""

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tool_node"

    return "validator"