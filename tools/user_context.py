from langchain.tools import tool
from storage.user_store import get_user


@tool
def get_user_profile(user_id: str) -> str:
    """Get basic profile info (name, preferred assets, risk tolerance) about
    the person currently in this conversation.
    """
    user = get_user(user_id)
    if not user:
        return "No profile on file for this session."

    assets = ", ".join(user.get("preferred_assets", [])) or "ninguno especificado"
    return (
        f"Nombre: {user['name']}. "
        f"Activos de interés: {assets}. "
        f"Tolerancia al riesgo: {user['risk_tolerance']}."
    )