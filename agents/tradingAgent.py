# Build workflow
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from edges.endRouter import should_continue
from edges.validationRouter import route_after_validation

from nodes.modelNode import llm_call
from nodes.toolNode import tool_node
from nodes.validatorNode import validate_node

from state import MessagesState
from dotenv import load_dotenv
load_dotenv()
agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("validator", validate_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", "validator"])
agent_builder.add_edge("tool_node", "llm_call")
agent_builder.add_conditional_edges("validator", route_after_validation, ["llm_call", END])


if __name__ == "__main__":
    from langchain.messages import HumanMessage
    from IPython.display import Image, display

    agent = agent_builder.compile()
    display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

    messages = [HumanMessage(content="")]
    messages = agent.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()