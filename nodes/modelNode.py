from datetime import date

from langchain.messages import SystemMessage

from localModel import model_with_tools


def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    today = date.today().isoformat()

    system_prompt = SystemMessage(content=(
        f"Today's date is {today}. Any date you see in tool results at or before "
        f"today is real, current data — do not treat it as implausible or 'from "
        f"the future' just because it's later than your training data.\n\n"
        "You are a financial market analysis assistant. You have access to:\n"
        "- get_market_data: price, high/low, and % change for a ticker\n"
        "- get_news: recent news headlines about an asset\n\n"
        "For open-ended questions like 'how is X doing' or 'current situation of X', "
        "call BOTH tools before answering.\n\n"
        "Rules:\n"
        "- Never state a price, percentage, or figure unless it came from a tool result.\n"
        "- If you don't have enough data, say so explicitly instead of estimating.\n"
        "- This is informational analysis only, not financial advice."
    ))

    return {
        "messages": [model_with_tools.invoke([system_prompt] + state["messages"])],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }