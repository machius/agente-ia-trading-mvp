import json
import uuid
from pathlib import Path

USERS_FILE = Path("data/users.json")


def _load() -> dict:
    if not USERS_FILE.exists():
        return {}
    with USERS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save(users: dict) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with USERS_FILE.open("w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def create_user(name: str, preferred_assets: list[str], risk_tolerance: str) -> str:
    users = _load()
    user_id = str(uuid.uuid4())
    users[user_id] = {
        "name": name,
        "preferred_assets": preferred_assets,
        "risk_tolerance": risk_tolerance,
    }
    _save(users)
    return user_id


def get_user(user_id: str) -> dict | None:
    return _load().get(user_id)