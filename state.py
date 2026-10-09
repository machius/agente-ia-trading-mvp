from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    tool_results: Annotated[list[dict], operator.add]
    validation_attempts: int
    needs_correction: bool
    user_id: str
    user_profile: dict