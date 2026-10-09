# Technical debt

Known issues that are intentionally NOT being fixed yet. Each entry says when it should be revisited.

## 1. Python environment does not match `requirements.txt`

**Status:** open, deliberately not resolved. Revisit **before reconnecting LangChain** (i.e. before wiring `MarketDataService` into `get_market_data`).

Observed on the development machine:

- Python **3.13.1**, installed **globally** (no virtual environment).
- `langchain` **1.0.3** installed, but `requirements.txt` asks for `langchain~=1.4.2`.
  Related installed versions: `langchain-core` 1.0.3, `langgraph` 1.0.2.
- Declared in `requirements.txt` but **not installed** in that interpreter: `yfinance`, `ddgs`, `langchain-ollama`, `python-dotenv`.
- Installed and consistent with the code: `httpx` 0.28.1, `streamlit` 1.51.0, `pydantic` 2.12.4.

Consequence: the agent graph itself has not been run against the declared dependency set. The Coinbase integration and the services layer do not import LangChain, so they are unaffected.

## 2. Development dependencies are not declared

**Status:** open. To be decided later; no reorganisation of requirements in the current phases.

- `pytest` was installed manually in the global environment (9.1.1) and is **not** listed in `requirements.txt`.
- Intended outcome: the project declares its development dependencies explicitly (e.g. a separate dev requirements file).
- Tests are run from the project root with `python -m pytest`. This is what puts the project root on `sys.path`, because the repository has no `__init__.py` files and no `pytest.ini`/`conftest.py`.

## 3. Coinbase integration: known limitations

**Status:** accepted for now.

- **HTTP 400 is ambiguous.** Coinbase answers an unknown product with `400` and an empty body, the same as any other malformed request. Only `404` maps to `INVALID_PRODUCT`; an unknown product currently surfaces as `BAD_REQUEST`. No extra product lookup is made to disambiguate.
- Candle buckets are aligned to the Unix epoch. This was verified against the live API for `ONE_DAY` (00:00 UTC) only; other granularities are assumed.
- Coinbase's public rate limit figures were not confirmed from official documentation. The client relies on `429` + `Retry-After` handling instead of a fixed number.

