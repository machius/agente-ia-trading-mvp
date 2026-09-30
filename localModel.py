from langchain_ollama import ChatOllama
from tools import aritmetic
from tools import market_data  # nuevo

model = ChatOllama(
    model="qwen3:4b",
    temperature=0
)

#aqui van los prompmts
# TODO crear, ordenar y gestionar los System prompts.
# TODO interactuar con el modelo ( telegram, UI, clod run)

tools = [aritmetic.add, aritmetic.multiply, aritmetic.divide,market_data.get_market_data]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)
