#pip install -U langchain-chroma
import warnings
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

from read_properties import get_property_value

warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime") #windows2016 < widows10
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 設定 OpenAI API 金鑰
api_key = get_property_value("openai_api_key")

llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", openai_api_key=api_key, temperature=0, max_tokens=None, timeout=None, max_retries=2)
embedding = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

# 設定存放向量數據庫的目錄
persist_dir = "chroma_data_dir"

# 準備向量數據庫
vectorStore = Chroma(persist_directory=persist_dir, embedding_function=embedding)
# 依相似度搜尋
#result  = vectorStore.similarity_search_with_score("how do i build a RAG client")
#print(result[0])

system_msg = """
你是一個將用戶問題轉換為資料庫查詢的專家。 你可以訪問關於構建大語言模型相關應用程序的軟體教程影片資料庫。
給定一個問題，返回一個資料庫查詢優化列表，以檢索最相關的結果。

如果有你不熟悉的縮寫和單字，不要試圖改變它們。
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_msg),
        ("human", "{question}")
    ]
)

#pydantic (python data validation) 定義資料模型
#檢索指令模型
class Search(BaseModel):
    query: str = Field(None, description="針對影片字幕內容的最相似搜索關鍵字")
    publish_year: Optional[int] = Field(None, description="影片發布年份")

chain = {"question": RunnablePassthrough()} | prompt | llm.with_structured_output(Search)
question = "hello你好呀, 想請問您一個困擾我很久的問題，我該如何建立一個RAG客戶端？"
resp1 = chain.invoke(question)
# question文字轉換為成定義資料模式(Search) => 將用戶問題透過LLM轉為最能直接命中的搜索問題
print(resp1)
question = "hello你好呀, 我想找一份在2023年份的資料:RAG原理"
resp2 = chain.invoke(question)
print(resp2)

def retrieval(search: Search)-> List[Document]:
    _filter = None
    if search.publish_year:
        _filter = {"publish_year": {"$eq": search.publish_year}}
    return vectorStore.similarity_search(search.query, filter=_filter)

# 接下來再以LLM轉換過的問題進行檢索向量數據庫
new_chain = chain | retrieval
result = new_chain.invoke(question)
print([(doc.metadata["title"], doc.metadata["publish_year"]) for doc in result])
