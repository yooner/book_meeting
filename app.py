import langchain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool, tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from datetime import datetime
import json
import os
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, MessagesState, StateGraph
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(max_results=2)
langchain.debug = True
os.environ["https_proxy"] = "socks5://127.0.0.1:7897"
# load_dotenv()
# os.environ["DEEPSEEK_API_KEY"] = "asd"



def create_llm():
    """创建LLM实例"""
    return ChatOpenAI(
        openai_api_key="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",
        openai_api_base="https://openrouter.ai/api/v1",
        model_name="deepseek/deepseek-r1:free",
        temperature=0
    )
model = create_llm()
def check_weather(location: str) -> str:
     '''返回当地的天气预报'''
     return f"It's always sunny in {location}"


tools = [check_weather]
graph = create_react_agent(model, tools)
inputs = {"messages": [("user", "上海的天气")]}
for s in graph.stream(inputs, stream_mode="values"):
    message = s["messages"][-1]
    if isinstance(message, tuple):
        print(message)
    else:
        message.pretty_print()

