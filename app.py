import streamlit as st
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agents.tradingAgent import agent_builder

st.set_page_config(page_title="AI Trading Agent", page_icon="📈")
st.title("📈 AI Trading Agent")
st.caption("Análisis de mercado con IA — no es asesoría financiera")

import uuid

if "agent" not in st.session_state:
    checkpointer = MemorySaver()
    st.session_state.agent = agent_builder.compile(checkpointer=checkpointer)
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []

config = {"configurable": {"thread_id": st.session_state.thread_id}}

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
                {"messages": [HumanMessage(content=prompt)]}, config
            )
            answer = result["messages"][-1].content
            st.markdown(answer)

            tool_results = result.get("tool_results", [])
            if tool_results:
                with st.expander("Ver detalle técnico"):
                    for tr in tool_results:
                        st.code(f"{tr['tool']}({tr['args']}) → {tr['result']}")

    st.session_state.history.append(("assistant", answer))