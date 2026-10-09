from langchain_ollama import ChatOllama
from tools import market_data, user_context

model = ChatOllama(
    model="qwen3:4b",
    temperature=0
)

#aqui van los prompmts
# TODO crear, ordenar y gestionar los System prompts.
# TODO interactuar con el modelo ( telegram, UI, clod run)

tools = [market_data.get_market_data, market_data.get_news, user_context.get_user_profile]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)
