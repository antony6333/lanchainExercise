
from typing import Optional, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from read_properties import get_property_value

# 設定 OpenAI API 金鑰
api_key = get_property_value("openai_api_key")

llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0, max_tokens=None, timeout=None, max_retries=2)

class Person(BaseModel):
    """
    關於人的資料模型
    """
    name: Optional[str] = Field(None, description="表示人的名字")
    hair_color: Optional[str] = Field(None, description="表示人的頭髮顏色")
    height_in_meters: Optional[str] = Field(None, description="表示人的身高")

class PersonList(BaseModel):
    people: List[Person]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """你是一個專業的提取算法。只從未結構化文本中提取相關信息。
         只從未結構化文本中提取相關信息。
         如果你不知道要提取的屬性的值，請保持該屬性的值為None。"""),
        #如果需要歷史記錄來提取文本的話
        #MessagesPlaceholder("chat_history"),
        ("human", "{text}")
    ]
)

chain = {"text": RunnablePassthrough()} | prompt | llm.with_structured_output(PersonList)
resp = chain.invoke("公車站有一個黑髮美女，目測身高約1米7，身旁是他的男朋友Jack，比他高10公分")
# 用於非結構化文本提取出結構化資料
print(resp)
