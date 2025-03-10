from math import pi
from typing import Union

from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from env_properties import get_property_value

#範例參考: https://www.pinecone.io/learn/series/langchain/langchain-tools/

# 設定 OpenAI API 金鑰
api_key = get_property_value("openai_api_key")

# 使用 OpenAI 作為 LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key, temperature=0, max_tokens=None, timeout=None, max_retries=2)

conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=3, #只保留最近的三個對話
        return_messages=True
)

# 自定義工具:計算圓周 (name, description寫給AI看的)
class CircumferenceTool(BaseTool):
    name:str = "圓周長計算器"
    description:str = "當你需要計算一個圓的圓周長時使用這個計算器並且需要一個半徑資料做為參數"
    def _run(self, radius: Union[int, float])->float:
        return float(radius) * 2.0 * pi
    def _arun(self, radius: int):
        raise NotImplementedError("This tool does not support async")

agent = initialize_agent(
    #這個agent透過賦予模型「與自己對話」的能力，實現多步驟推理和使用工具
    agent="chat-conversational-react-description",
    tools=[CircumferenceTool()],  #可以有多個工具讓AI自己挑選
    llm=llm,
    verbose=True, #顯示更多訊息
    max_iterations=3,
    early_stopping_method="generate",
    memory=conversational_memory
)

resp = agent.invoke("我有一個圓，這個圓的半徑是7.81mm，請問這個圓的圓周長是多少？")
print(resp["output"])

