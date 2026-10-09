from datetime import date
from langchain.messages import SystemMessage
from localModel import model_with_tools
from storage.user_store import get_user

# Instrucción originalpara get_market_data — conservada para
# no perderla, reemplazada abajo por una versión que acepta 1-4 activos:
# "- get_market_data: Get historical market data and indicators. You MUST
#    request EXACTLY 4 assets at once. For crypto assets, you MUST use the
#    format 'BASE-USD' (e.g., 'BTC-USD'). Returns JSON with close price,
#    SMA50, SMA200, and RCI9."

MARKET_DATA_INSTRUCTIONS = (
    "- get_market_data: Get market data and indicators for 1 to 4 assets. For crypto "
    "assets, use the format 'BASE-USD' (e.g., 'BTC-USD'). Returns JSON with close price, "
    "SMA50, SMA200, and RCI9. Pass as_of_date (YYYY-MM-DD) to get data for a past date "
    "instead of today — up to 365 days back. Compute the date yourself from today's date "
    "above (e.g. 'yesterday' = today minus 1 day). Never pass a future date."
)


def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    today = date.today().isoformat()

    user_profile = state.get("user_profile")
    if user_profile is None:
        user_profile = get_user(state.get("user_id", "")) or {}

    preferred_assets = user_profile.get("preferred_assets", [])
    risk_tolerance = user_profile.get("risk_tolerance")

    profile_line = ""
    if preferred_assets or risk_tolerance:
        parts = []
        if preferred_assets:
            parts.append(f"their assets of interest are {', '.join(preferred_assets)}")
        if risk_tolerance:
            parts.append(f"their risk tolerance is {risk_tolerance}")
        profile_line = (
            f"IMPORTANT: this user has a saved profile — {' and '.join(parts)}. "
            f"For ANY general question about 'the market' with no specific asset named, "
            f"you MUST call get_market_data using exactly these assets — do not substitute "
            f"other tickers. Factor in their risk tolerance when framing your analysis "
            f"(flag volatility more strongly for a conservative profile).\n\n"
        )

    system_prompt = SystemMessage(content=(
        f"Today's date is {today}. Any date you see in tool results at or before "
        f"today is real, current data — do not treat it as implausible or 'from "
        f"the future' just because it's later than your training data.\n\n"
        f"{profile_line}"
        "You are a financial market analysis assistant. You have access to:\n"
        f"{MARKET_DATA_INSTRUCTIONS}\n"
        "- get_news: recent news headlines about an asset\n"
        "- get_user_profile: basic profile info about the current user\n\n"
        "For open-ended questions like 'how is X doing' or 'current situation of X', "
        "call BOTH get_market_data and get_news before answering.\n\n"
        "Rules:\n"
        "- Never state a price, percentage, or figure unless it came from a tool result.\n"
        "- If you don't have enough data, say so explicitly instead of estimating.\n"
        "- This is informational analysis only, not financial advice."
    ))

    return {
        "messages": [model_with_tools.invoke([system_prompt] + state["messages"])],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "user_profile": user_profile,  # cacheado: los siguientes turnos no vuelven a leer disco
    }