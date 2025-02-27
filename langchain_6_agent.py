# pip install langgraph
import os

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import chat_agent_executor

from read_properties import get_property_value

# 設定 OpenAI API 金鑰
api_key = get_property_value("openai_api_key")

# 使用 OpenAI 作為 LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key, temperature=0, max_tokens=None, timeout=None, max_retries=2)

#result = llm.invoke("台北天氣如何？")
#print(result)

# 使用Tavily AI搜索引擎
os.environ['TAVILY_API_KEY'] = "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"
tavilySearch = TavilySearchResults(max_results=1)
#search = tavilySearch.invoke("台北天氣如何？")
# tavilySearch會回傳一個list，裡面包含了網頁搜索結果
#print(search)

# 讓模型綁定Tavily搜索引擎 (模型可以自動推理是否需要調要工具(Tavily)來回答問題)
#llm_with_tavily = llm.bind_tools([tavilySearch])
#response = llm_with_tavily.invoke([HumanMessage(content="台北天氣如何？")])
# LLM無法回答的答案，會回調tavily並把代理id放到tool_calls中
#print(response.tool_calls)

agentExecutor = chat_agent_executor.create_tool_calling_executor(llm, [tavilySearch])
response = agentExecutor.invoke({"messages":[HumanMessage(content="新北市板橋區天氣如何? site:tw")]})
print(response["messages"][-1].content)
