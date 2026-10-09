import uuid
import pandas as pd
import streamlit as st
from langchain.messages import HumanMessage, AIMessage

from agents.tradingAgent import agent_builder
from storage.user_store import create_user, get_user
from storage.db import build_checkpointer
from services.price_history import get_price_history

st.set_page_config(page_title="AI Trading Agent", page_icon="📈")
st.title("📈 AI Trading Agent")

ASSET_OPTIONS = {
    "S&P 500": "^GSPC",
    "Oro": "GC=F",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
}


# ---------- funciones auxiliares (sin efectos secundarios de página) ----------

@st.cache_resource
def get_checkpointer():
    return build_checkpointer()


@st.cache_data(ttl=300)  # 5 min — evita golpear Coinbase/yfinance en cada rerun del chat
def load_chart_data(symbol: str) -> pd.DataFrame:
    return get_price_history(symbol)


def load_display_history(agent, config) -> list[tuple[str, str]]:
    """Reconstruye el historial visible a partir del state real del checkpointer."""
    snapshot = agent.get_state(config)
    if not snapshot.values:
        return []

    history = []
    for msg in snapshot.values.get("messages", []):
        if isinstance(msg, HumanMessage) and msg.content:
            history.append(("user", msg.content))
        elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            history.append(("assistant", msg.content))
    return history


# ---------- estado de sesión ----------

if "agent" not in st.session_state:
    st.session_state.agent = agent_builder.compile(checkpointer=get_checkpointer())
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []
    st.session_state.user_id = None


# ---------- gate de login: nada después de esto corre sin user_id ----------

if st.session_state.user_id is None:
    st.subheader("Antes de empezar")

    if st.session_state.get("pending_user_id"):
        st.success("¡Listo! Este es tu ID:")
        st.code(st.session_state.pending_user_id)
        st.warning(
            "Guárdalo — lo vas a necesitar para mantener el contexto de "
            "tus asesorías la próxima vez que entres."
        )
        if st.button("Continuar al chat"):
            st.session_state.user_id = st.session_state.pending_user_id
            st.session_state.thread_id = st.session_state.pending_user_id
            del st.session_state.pending_user_id
            st.rerun()
        st.stop()

    mode = st.radio("¿Ya tienes un ID?", ["Soy nuevo", "Ya tengo un ID"], horizontal=True)

    if mode == "Ya tengo un ID":
        with st.form("resume_form"):
            existing_id = st.text_input("Tu ID")
            resume = st.form_submit_button("Continuar")

        if resume:
            user = get_user(existing_id.strip())
            if user:
                st.session_state.user_id = existing_id.strip()
                st.session_state.thread_id = existing_id.strip()
                config = {"configurable": {"thread_id": existing_id.strip()}}
                st.session_state.history = load_display_history(st.session_state.agent, config)
                st.rerun()
            else:
                st.error("No encontramos ese ID. Revisa que esté completo.")

    else:
        with st.form("user_form"):
            name = st.text_input("Nombre")
            selected_labels = st.multiselect(
                "Activos de interés (máximo 4)", list(ASSET_OPTIONS.keys())
            )
            risk_tolerance = st.selectbox("Tolerancia al riesgo", ["Conservador", "Moderado", "Agresivo"])
            submitted = st.form_submit_button("Comenzar")

        if submitted:
            if not name:
                st.error("Ingresa un nombre.")
            elif len(selected_labels) > 4:
                st.error("Elige máximo 4 activos.")
            else:
                preferred_assets = [ASSET_OPTIONS[label] for label in selected_labels]
                st.session_state.pending_user_id = create_user(name, preferred_assets, risk_tolerance)
                st.rerun()

    st.stop()


# ---------- a partir de aquí, user_id ya existe garantizado ----------

config = {"configurable": {"thread_id": st.session_state.thread_id}}

st.subheader("📊 Tus activos")
user = get_user(st.session_state.user_id)
chart_assets = user.get("preferred_assets") or list(ASSET_OPTIONS.values())
selected_symbol = st.selectbox("Ver gráfico de", chart_assets)

chart_df = load_chart_data(selected_symbol)
if chart_df.empty:
    st.info(f"No hay datos de precio disponibles para {selected_symbol}.")
else:
    st.line_chart(chart_df.set_index("date")["close"])


# ---------- chat ----------

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

if prompt := st.chat_input("Pregunta sobre un activo (ej: ¿Qué tal se ve Bitcoin hoy?)"):
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            result = st.session_state.agent.invoke(
                {"messages": [HumanMessage(content=prompt)], "user_id": st.session_state.user_id},
                config,
            )
            answer = result["messages"][-1].content
            st.markdown(answer.replace("$", "\\$"))

            tool_results = result.get("tool_results", [])
            if tool_results:
                with st.expander("Ver detalle técnico"):
                    for tr in tool_results:
                        st.code(f"{tr['tool']}({tr['args']}) → {tr['result']}")

    st.session_state.history.append(("assistant", answer))