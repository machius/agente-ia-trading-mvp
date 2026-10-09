from langchain.messages import ToolMessage
from localModel import tools_by_name

INJECTED_ARGS = {
    "get_user_profile": lambda state: {"user_id": state.get("user_id", "")},
}


def tool_node(state: dict):
    """Performs the tool call"""

    messages = []
    results = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]

        args = tool_call["args"]
        if tool_call["name"] in INJECTED_ARGS:
            args = INJECTED_ARGS[tool_call["name"]](state)  # ignora lo que puso el LLM

        try:
            observation = tool.invoke(args)
        except Exception as e:
            if type(e).__name__ == "ValidationError":
                observation = f"Argument validation error: {str(e)}"
            else:
                observation = f"Unexpected execution error: {type(e).__name__} - {str(e)}"

        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
        results.append({
            "tool": tool_call["name"],
            "args": args,
            "result": observation,
        })
    return {"messages": messages, "tool_results": results}