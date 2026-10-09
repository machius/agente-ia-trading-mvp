from langchain.messages import ToolMessage

from localModel import tools_by_name


def tool_node(state: dict):
    """Performs the tool call"""

    messages = []
    results = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        try:
            observation = tool.invoke(tool_call["args"])
        except Exception as e:
            if type(e).__name__ == "ValidationError":
                observation = f"Argument validation error: {str(e)}"
            else:
                observation = f"Unexpected execution error: {type(e).__name__} - {str(e)}"
            
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
        results.append({
            "tool": tool_call["name"],
            "args": tool_call["args"],
            "result": observation,
        })
    return {"messages": messages, "tool_results": results}