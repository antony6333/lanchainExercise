import os

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from env_properties import get_property_value

# Create the agent
memory = MemorySaver()
openai_api_key = get_property_value("openai_api_key")
model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
os.environ['TAVILY_API_KEY'] = get_property_value("tavily_api_key")
search = TavilySearchResults(max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="哈囉，我是安東尼，我住在台灣新北市")]} ,
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

for step in agent_executor.stream(
    {"messages": [HumanMessage(content="我住的地方有什麼天氣如何?")]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()