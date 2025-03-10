# 安裝 Visual C++ Build Tools (勾選 Windows 10 SDK 和 C++ CMake 工具)
# https://visualstudio.microsoft.com/zh-hant/visual-cpp-build-tools/
# pip install --upgrade pip setuptools wheel (更新 pip setuptools wheel)
# pip install langchain chromadb
import warnings

from env_properties import get_property_value

warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime") #windows2016 < widows10
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.runnables.base import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI,OpenAIEmbeddings

# 設定 OpenAI API 金鑰
api_key = get_property_value("openai_api_key")

# 使用 OpenAI 作為 LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key, temperature=0, max_tokens=None, timeout=None, max_retries=2)

documents = [
    Document(
        page_content="Jimmy的家在台中西區，購房金額800萬，大樓管理住宅位於3樓，格局為2室1廳1衛",
        metadata={"source": "台灣大哥大同仁住宅調查"}
    ),
    Document(
        page_content="Flora的家在台中東區，購房金額1020萬，平房住宅位於2樓，格局為4室2廳2衛",
        metadata={"source": "台灣大哥大同仁住宅調查"}
    ),
    Document(
        page_content="Antony的家在新北板橋區，購房金額1200萬，大樓管理住宅位於6樓，格局為3室1廳2衛",
        metadata={"source": "台灣大哥大同仁住宅調查"}
    )
]

# 實例化向量數據空間
vector_store = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key))
# 輸出相似度(k=回傳最大筆數), 分數越低越相似
print(vector_store.similarity_search_with_score("台哥大Flora", k=3))

# 檢索器
retriever = RunnableLambda(vector_store.similarity_search).bind(k=1)
# 一次找多筆相似資料
print(retriever.batch(["Flora", "Antony"]))

# 提示模版
message = """
使用提供的上下文僅回答這個問題:
{question}
上下文:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

# RunnablePassthrough: 透過運行器傳遞訊息
chain = {"question": RunnablePassthrough(), "context": retriever} | prompt | llm

response = chain.invoke("請介紹Jimmy")

print(response.content)