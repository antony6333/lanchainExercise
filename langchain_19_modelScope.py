#pip install modelscope
#pip install addict
#pip install datasets
#pip install torch
#pip install pillow
#pip install simplejson
#pip install sortedcontainers
#pip install transformers
#第一次run會有pytorch_model.bin下載模型進度提示，
import warnings

from read_properties import get_property_value

warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")
from langchain_chroma import Chroma
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableLambda
from langchain_openai import ChatOpenAI

# 設定 AI API 金鑰
api_key = get_property_value("zhipu_api_key")

# 使用 智譜AI(zhipu) 作為 LLM
# 與openai不同的地方除了model名稱不同，參數名稱也換成api_key,base_url
llm = ChatOpenAI(model="glm-4-0520", api_key=api_key,
                 base_url="https://open.bigmodel.cn/api/paas/v4/",
                 temperature=0, max_tokens=None, timeout=None, max_retries=2)

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

#替換成魔塔社區的embedding
#https://modelscope.cn/models/iic/nlp_gte_sentence-embedding_chinese-base
embedding = ModelScopeEmbeddings(model_id="iic/nlp_gte_sentence-embedding_chinese-base")

# 實例化向量數據空間
# embedding模型會下載到本地進行運算，沒有cuda會很慢
vector_store = Chroma.from_documents(documents, embedding=embedding)
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